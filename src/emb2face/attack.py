from __future__ import annotations

import os
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .embeddings import (
    arcface_from_image,
    load_adaface_model,
    load_arcface_app,
    setup_device,
    source_embeddings,
)
from .models import LinearAdapter, MLPAdapter
from .utils import compute_eer, cosine_similarity_np, far_frr_at, l2_normalize


def ensure_arc2face_repo(cfg: dict) -> Path:
    repo_dir = Path(cfg["arc2face_local_dir"]).expanduser()
    if repo_dir.exists():
        return repo_dir
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "https://github.com/foivospar/Arc2Face.git", str(repo_dir)], check=True)
    return repo_dir


def load_arc2face_pipeline(cfg: dict, device: torch.device):
    repo_dir = ensure_arc2face_repo(cfg)
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline, UNet2DConditionModel
    from arc2face import CLIPTextModelWrapper, project_face_embs
    from huggingface_hub import hf_hub_download

    model_dir = Path(cfg["output_root"]).expanduser() / ".cache" / "arc2face_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for filename in [
        "arc2face/config.json",
        "arc2face/diffusion_pytorch_model.safetensors",
        "encoder/config.json",
        "encoder/pytorch_model.bin",
    ]:
        hf_hub_download(repo_id=cfg["arc2face_repo"], filename=filename, local_dir=str(model_dir))

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    encoder = CLIPTextModelWrapper.from_pretrained(model_dir, subfolder="encoder", torch_dtype=dtype)
    unet = UNet2DConditionModel.from_pretrained(model_dir, subfolder="arc2face", torch_dtype=dtype)
    pipeline = StableDiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        text_encoder=encoder,
        unet=unet,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(device)
    return pipeline, project_face_embs


def load_best_adapter(cfg: dict, device: torch.device):
    model_dir = cfg["output_root"] / f"models_{cfg['adapter_run_mode']}"
    candidates = sorted(model_dir.glob("best_*_adapter.pt"))
    if not candidates:
        raise FileNotFoundError(f"No adapter checkpoint found in {model_dir}")
    ckpt = torch.load(candidates[0], map_location=device)
    acfg = ckpt.get("config", {})
    atype = acfg.get("adapter_type", "linear")
    adapter = (
        LinearAdapter(512)
        if atype == "linear"
        else MLPAdapter(512, acfg.get("hidden_dim", 1024), acfg.get("dropout", 0.1))
    ).to(device)
    adapter.load_state_dict(ckpt["state_dict"])
    adapter.eval()
    return adapter, candidates[0]


def collect_identity_folder(root, exts, n_ids, n_imgs):
    rows = []
    for d in sorted([p for p in Path(root).iterdir() if p.is_dir()]):
        imgs = sorted([p for p in d.rglob("*") if p.suffix.lower() in exts])
        if len(imgs) < n_imgs:
            continue
        for p in imgs[:n_imgs]:
            rows.append({"identity": d.name, "image_path": str(p)})
        if len({r["identity"] for r in rows}) >= n_ids:
            break
    return pd.DataFrame(rows)


def build_eval_set(cfg: dict):
    runmode = cfg["runmode"].lower()
    eval_ids = cfg["debug_eval_identities"] if cfg["runmode"] == "DEBUG" else cfg["eval_identities"]
    eval_imgs = cfg["debug_eval_images_per_identity"] if cfg["runmode"] == "DEBUG" else cfg["eval_images_per_identity"]
    attack_root = cfg["output_root"] / f"attack_{runmode}"
    recon_root = attack_root / "reconstructions"
    eval_report = attack_root / "reports"
    eval_emb = attack_root / "embeddings"
    for p in (attack_root, recon_root, eval_report, eval_emb):
        p.mkdir(parents=True, exist_ok=True)

    nb1_reports = cfg["output_root"] / f"reports_{cfg['adapter_run_mode']}"

    if cfg["eval_source"] == "external":
        eval_df = collect_identity_folder(cfg["eval_dataset_root"], cfg["image_extensions"], eval_ids, eval_imgs)
        if len(eval_df) == 0:
            raise ValueError(f"No identity folders with >= {eval_imgs} images under {cfg['eval_dataset_root']}")
    else:
        splits_csv = nb1_reports / "paired_metadata_with_splits.csv"
        meta_csv = nb1_reports / "paired_metadata.csv"
        if splits_csv.exists():
            meta = pd.read_csv(splits_csv)
            test_ids = sorted(meta.loc[meta["split"] == "test", "identity"].unique())
        elif meta_csv.exists():
            from sklearn.model_selection import train_test_split

            meta = pd.read_csv(meta_csv)
            ids = sorted(meta["identity"].unique())
            _, tmp = train_test_split(ids, test_size=1 - cfg["train_id_fraction"], random_state=cfg["seed"])
            vr = cfg["val_id_fraction"] / (cfg["val_id_fraction"] + cfg["test_id_fraction"])
            _, test_ids = train_test_split(tmp, test_size=1 - vr, random_state=cfg["seed"])
            test_ids = sorted(test_ids)
        else:
            raise FileNotFoundError("Run the training pipeline first so paired metadata exists.")

        pool = test_ids if cfg["use_test_split_from_notebook1"] else sorted(meta["identity"].unique())
        rows = []
        for ident in pool:
            paths = sorted(meta[meta["identity"] == ident]["image_path"].tolist())
            if len(paths) < eval_imgs:
                continue
            for p in paths[:eval_imgs]:
                rows.append({"identity": ident, "image_path": p})
            if len({r["identity"] for r in rows}) >= eval_ids:
                break
        eval_df = pd.DataFrame(rows)

    eval_df = eval_df.reset_index(drop=True)
    eval_df["probe_id"] = eval_df.index
    eval_df.to_csv(eval_report / "eval_set.csv", index=False)
    return eval_df, attack_root, recon_root, eval_report, eval_emb


def embedding_for(exp, idx, src_arc, src_ada, adapter, device):
    if exp == "exp1_arcface_baseline":
        return src_arc[idx]
    if exp == "exp2_adaface_wrongspace":
        return src_ada[idx]
    if exp == "exp3_adapter_mapped":
        a = torch.from_numpy(src_ada[idx]).float().unsqueeze(0).to(device)
        with torch.no_grad():
            pred = adapter(a).cpu().numpy()[0]
        return l2_normalize(pred).astype(np.float32)
    raise ValueError(exp)


def generate(pipeline, project_face_embs, emb512, n, seed, device, cfg):
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    id_emb = torch.from_numpy(np.asarray(emb512, dtype=np.float32)).to(device, dtype)[None]
    id_emb = id_emb / torch.norm(id_emb, dim=1, keepdim=True)
    proj = project_face_embs(pipeline, id_emb)
    g = torch.Generator(device=device).manual_seed(seed)
    return pipeline(
        prompt_embeds=proj,
        num_inference_steps=cfg["num_inference_steps"],
        guidance_scale=cfg["guidance_scale"],
        num_images_per_prompt=n,
        generator=g,
    ).images


def pixel_metrics(recon_rgb, source_rgb, lpips_model, device, size=112):
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn
    from skimage.metrics import structural_similarity as ssim_fn

    a = cv2.resize(recon_rgb, (size, size))
    b = cv2.resize(source_rgb, (size, size))
    s = ssim_fn(a, b, channel_axis=2, data_range=255)
    p = psnr_fn(b, a, data_range=255)
    with torch.no_grad():
        l = float(lpips_model(_to_lpips(a, device), _to_lpips(b, device)).item())
    return float(s), float(p), l


def _to_lpips(img_rgb_uint8, device, size=256):
    im = cv2.resize(img_rgb_uint8, (size, size)).astype(np.float32) / 255.0
    t = torch.from_numpy(im).permute(2, 0, 1).unsqueeze(0) * 2 - 1
    return t.to(device)


def run_attack_pipeline(cfg: dict):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import lpips as lpips_lib

    device = setup_device(cfg)
    adapter, adapter_path = load_best_adapter(cfg, device)
    arc_app, _ = load_arcface_app(cfg)
    ada_model = load_adaface_model(cfg, device)
    pipeline, project_face_embs = load_arc2face_pipeline(cfg, device)
    lpips_model = lpips_lib.LPIPS(net="alex").to(device).eval()

    eval_df, attack_root, recon_root, eval_report, eval_emb = build_eval_set(cfg)
    src_arc, src_ada, src_crops, keep = [], [], [], []
    for _, r in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Source embeddings"):
        e = source_embeddings(r["image_path"], arc_app, ada_model, device)
        if e is None:
            keep.append(False)
            continue
        keep.append(True)
        src_arc.append(e["arcface"])
        src_ada.append(e["adaface"])
        src_crops.append(e["crop_rgb"])

    eval_df = eval_df[keep].reset_index(drop=True)
    eval_df["probe_id"] = eval_df.index
    src_arc = np.stack(src_arc)
    src_ada = np.stack(src_ada)
    np.save(eval_emb / "src_arc.npy", src_arc)
    np.save(eval_emb / "src_ada.npy", src_ada)

    recon_rows = []
    for exp in cfg["experiments"]:
        for _, r in tqdm(eval_df.iterrows(), total=len(eval_df), desc=exp):
            idx = int(r["probe_id"])
            out_dir = recon_root / exp / str(r["identity"])
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(r["image_path"]).stem
            needed = [out_dir / f"{stem}_r{k}.png" for k in range(cfg["num_recon_per_image"])]
            if not all(p.exists() for p in needed):
                imgs = generate(
                    pipeline,
                    project_face_embs,
                    embedding_for(exp, idx, src_arc, src_ada, adapter, device),
                    cfg["num_recon_per_image"],
                    cfg["seed"] + idx,
                    device,
                    cfg,
                )
                for k, im in enumerate(imgs):
                    im.save(needed[k])
            for k, p in enumerate(needed):
                recon_rows.append(
                    {
                        "experiment": exp,
                        "probe_id": idx,
                        "identity": r["identity"],
                        "recon_k": k,
                        "recon_path": str(p),
                    }
                )

    recon_df = pd.DataFrame(recon_rows)
    recon_df.to_csv(eval_report / "recon_index.csv", index=False)

    recon_emb = {}
    recon_face_ok = {}
    for exp in cfg["experiments"]:
        sub = recon_df[recon_df["experiment"] == exp].reset_index(drop=True)
        embs, ok = [], []
        for _, r in tqdm(sub.iterrows(), total=len(sub), desc=f"score {exp}"):
            emb = arcface_from_image(r["recon_path"], arc_app)
            if emb is None:
                embs.append(np.zeros(512, np.float32))
                ok.append(False)
            else:
                embs.append(emb)
                ok.append(True)
        recon_emb[exp] = np.stack(embs)
        recon_face_ok[exp] = np.array(ok)
        np.save(eval_emb / f"recon_arc_{exp}.npy", recon_emb[exp])

    id_to_idx = defaultdict(list)
    for i, ident in enumerate(eval_df["identity"]):
        id_to_idx[ident].append(i)
    all_idx = list(range(len(eval_df)))
    rng = random.Random(cfg["seed"])

    typeI_rows, typeI_summary, typeI_scores = [], [], []
    for exp in cfg["experiments"]:
        sub = recon_df[recon_df["experiment"] == exp].reset_index(drop=True)
        R = recon_emb[exp]
        OKMASK = recon_face_ok[exp]
        labels, scores = [], []
        for ri, r in sub.iterrows():
            pid = int(r["probe_id"])
            if not OKMASK[ri]:
                continue
            gen = cosine_similarity_np(R[ri], src_arc[pid])
            labels.append(1)
            scores.append(gen)
            typeI_scores.append({"experiment": exp, "label": 1, "score": gen})
            rec_bgr = cv2.imread(r["recon_path"])
            rec_rgb = cv2.cvtColor(rec_bgr, cv2.COLOR_BGR2RGB)
            ssim_v, psnr_v, lpips_v = pixel_metrics(rec_rgb, src_crops[pid], lpips_model, device)
            others = [j for j in all_idx if eval_df.loc[j, "identity"] != r["identity"]]
            for j in rng.sample(others, min(cfg["impostors_per_probe"], len(others))):
                sc = cosine_similarity_np(R[ri], src_arc[j])
                labels.append(0)
                scores.append(sc)
                typeI_scores.append({"experiment": exp, "label": 0, "score": sc})
            typeI_rows.append(
                {
                    "experiment": exp,
                    "probe_id": pid,
                    "identity": r["identity"],
                    "genuine_cosine": gen,
                    "ssim": ssim_v,
                    "psnr": psnr_v,
                    "lpips": lpips_v,
                }
            )
        eer, thr = compute_eer(labels, scores)
        far, frr = far_frr_at(labels, scores, thr)
        g = [s for l, s in zip(labels, scores) if l == 1]
        im = [s for l, s in zip(labels, scores) if l == 0]
        dfx = pd.DataFrame([x for x in typeI_rows if x["experiment"] == exp])
        typeI_summary.append(
            {
                "experiment": exp,
                "attack": "Type-I",
                "n_probes": len(dfx),
                "mean_genuine_cosine": float(np.mean(g)) if g else float("nan"),
                "mean_impostor_cosine": float(np.mean(im)) if im else float("nan"),
                "eer": eer,
                "eer_threshold": thr,
                "far_at_eer": far,
                "frr_at_eer": frr,
                "mean_ssim": float(dfx["ssim"].mean()),
                "mean_psnr": float(dfx["psnr"].mean()),
                "mean_lpips": float(dfx["lpips"].mean()),
            }
        )

    pd.DataFrame(typeI_scores).to_csv(eval_report / "typeI_scores.csv", index=False)
    pd.DataFrame(typeI_rows).to_csv(eval_report / "typeI_per_probe.csv", index=False)
    typeI_df = pd.DataFrame(typeI_summary)
    typeI_df.to_csv(eval_report / "typeI_summary.csv", index=False)

    typeII_summary = []
    for exp in cfg["experiments"]:
        sub = recon_df[recon_df["experiment"] == exp].reset_index(drop=True)
        R = recon_emb[exp]
        OKMASK = recon_face_ok[exp]
        labels, scores = [], []
        for ri, r in sub.iterrows():
            if not OKMASK[ri]:
                continue
            pid = int(r["probe_id"])
            ident = r["identity"]
            same = [j for j in id_to_idx[ident] if j != pid]
            for j in same:
                labels.append(1)
                scores.append(cosine_similarity_np(R[ri], src_arc[j]))
            others = [j for j in all_idx if eval_df.loc[j, "identity"] != ident]
            for j in rng.sample(others, min(cfg["impostors_per_probe"], len(others))):
                labels.append(0)
                scores.append(cosine_similarity_np(R[ri], src_arc[j]))
        eer, thr = compute_eer(labels, scores)
        far, frr = far_frr_at(labels, scores, thr)
        g = [s for l, s in zip(labels, scores) if l == 1]
        im = [s for l, s in zip(labels, scores) if l == 0]
        typeII_summary.append(
            {
                "experiment": exp,
                "attack": "Type-II",
                "n_pairs": len(scores),
                "mean_genuine_cosine": float(np.mean(g)) if g else float("nan"),
                "mean_impostor_cosine": float(np.mean(im)) if im else float("nan"),
                "eer": eer,
                "far_at_eer": far,
                "frr_at_eer": frr,
            }
        )

    typeII_df = pd.DataFrame(typeII_summary)
    typeII_df.to_csv(eval_report / "typeII_summary.csv", index=False)

    summary = pd.concat([typeI_df, typeII_df], ignore_index=True, sort=False)
    summary.to_csv(eval_report / "results_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    exps = cfg["experiments"]
    x = np.arange(len(exps))
    w = 0.35
    tI = [typeI_df.set_index("experiment").loc[e, "eer"] for e in exps]
    tII = [typeII_df.set_index("experiment").loc[e, "eer"] for e in exps]
    ax.bar(x - w / 2, tI, w, label="Type-I")
    ax.bar(x + w / 2, tII, w, label="Type-II")
    ax.set_xticks(x)
    ax.set_xticklabels([e.split("_", 1)[0] for e in exps])
    ax.set_ylabel("EER")
    ax.set_title("Attack EER by experiment")
    ax.legend()
    plt.tight_layout()
    plt.savefig(eval_report / "eer_by_experiment.png", dpi=150)
    plt.close(fig)

    scores_df = pd.read_csv(eval_report / "typeI_scores.csv")
    thr_map = typeI_df.set_index("experiment")["eer_threshold"].to_dict()
    fig, axes = plt.subplots(1, len(exps), figsize=(5 * len(exps), 4), squeeze=False)
    for ax, exp in zip(axes[0], exps):
        d = scores_df[scores_df["experiment"] == exp]
        ax.hist(d[d.label == 1]["score"], bins=30, alpha=0.6, label="genuine", color="#2a9d8f")
        ax.hist(d[d.label == 0]["score"], bins=30, alpha=0.6, label="impostor", color="#e76f51")
        t = thr_map.get(exp, float("nan"))
        if not np.isnan(t):
            ax.axvline(t, color="k", ls="--", lw=1, label="EER thr")
        ax.set_title(exp.split("_", 1)[0])
        ax.set_xlabel("cosine (antelopev2)")
        ax.legend(fontsize=8)
    fig.suptitle("Type-I: genuine vs impostor score distributions")
    plt.tight_layout()
    plt.savefig(eval_report / "typeI_distributions.png", dpi=150)
    plt.close(fig)

    n_show = min(5, len(eval_df))
    for exp in cfg["experiments"]:
        sub = recon_df[(recon_df["experiment"] == exp) & (recon_df["recon_k"] == 0)].head(n_show)
        fig, axes = plt.subplots(2, n_show, figsize=(2.2 * n_show, 4.6))
        for col, (_, r) in enumerate(sub.iterrows()):
            pid = int(r["probe_id"])
            axes[0, col].imshow(src_crops[pid])
            axes[0, col].axis("off")
            rec = cv2.cvtColor(cv2.imread(r["recon_path"]), cv2.COLOR_BGR2RGB)
            axes[1, col].imshow(rec)
            axes[1, col].axis("off")
        axes[0, 0].set_ylabel("source", fontsize=11)
        axes[1, 0].set_ylabel("recon", fontsize=11)
        fig.suptitle(exp)
        plt.tight_layout()
        plt.savefig(eval_report / f"grid_{exp}.png", dpi=150)
        plt.close(fig)

    return {
        "adapter_path": adapter_path,
        "eval_df": eval_df,
        "summary": summary,
        "typeI_df": typeI_df,
        "typeII_df": typeII_df,
    }
