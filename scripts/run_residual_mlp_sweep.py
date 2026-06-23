from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

from emb2face.config import load_config
from emb2face.train import run_training_pipeline


def _serialize_cfg(cfg: dict) -> dict:
    out = {}
    for key, value in cfg.items():
        if isinstance(value, Path):
            out[key] = str(value)
        elif isinstance(value, dict):
            out[key] = _serialize_cfg(value)
        elif isinstance(value, list):
            out[key] = [str(v) if isinstance(v, Path) else v for v in value]
        else:
            out[key] = value
    return out


def _parse_args():
    parser = argparse.ArgumentParser(description="Run the residual MLP adapter sweep")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--hidden-dims", type=str, default="1024,2048", help="Comma-separated hidden dims to run")
    return parser.parse_args()


def main():
    args = _parse_args()
    base_cfg = load_config(
        args.config,
        overrides={
            "output_root": args.output_root,
            "device": args.device,
            "seed": args.seed,
        },
    )

    hidden_dims = [int(x.strip()) for x in args.hidden_dims.split(",") if x.strip()]
    sweep_root = Path(base_cfg["output_root"]) / "sweeps" / "residual_mlp"
    sweep_root.mkdir(parents=True, exist_ok=True)

    for hidden_dim in hidden_dims:
        run_cfg = copy.deepcopy(base_cfg)
        run_cfg["adapter_type"] = "residual_mlp"
        run_cfg["hidden_dim"] = hidden_dim
        run_cfg["output_root"] = Path(base_cfg["output_root"]) / f"residual_mlp_{hidden_dim}"
        run_cfg["emb_dir"] = run_cfg["output_root"] / f"embeddings_{run_cfg['runmode'].lower()}"
        run_cfg["model_dir"] = run_cfg["output_root"] / f"models_{run_cfg['runmode'].lower()}"
        run_cfg["report_dir"] = run_cfg["output_root"] / f"reports_{run_cfg['runmode'].lower()}"
        for p in (run_cfg["output_root"], run_cfg["emb_dir"], run_cfg["model_dir"], run_cfg["report_dir"]):
            p.mkdir(parents=True, exist_ok=True)

        cfg_path = sweep_root / f"residual_mlp_{hidden_dim}.yaml"
        with open(cfg_path, "w") as f:
            yaml.safe_dump(_serialize_cfg(run_cfg), f, sort_keys=False)

        print(f"Running residual MLP sweep with hidden_dim={hidden_dim}")
        print(f"Config snapshot: {cfg_path}")
        run_training_pipeline(run_cfg)


if __name__ == "__main__":
    main()
