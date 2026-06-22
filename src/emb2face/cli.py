from __future__ import annotations

import argparse
from pathlib import Path
from .config import load_config


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="emb2face")
    parser.add_argument("command", choices=["train", "attack", "infer", "all"])
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--runmode", type=str, default=None)
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--eval-source", type=str, default=None)
    parser.add_argument("--adapter-run-mode", type=str, default=None)
    parser.add_argument("--experiments", type=str, default=None, help="Comma-separated experiment list")
    parser.add_argument("--input-dir", type=str, default=None, help="Input directory for inference")
    parser.add_argument("--input-image", type=str, default=None, help="Single image for inference")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for inference")
    parser.add_argument("--num-images-per-prompt", type=int, default=None)
    parser.add_argument("--max-images-per-identity", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    overrides = {
        "runmode": args.runmode,
        "dataset_root": args.dataset_root,
        "output_root": args.output_root,
        "device": args.device,
        "eval_source": args.eval_source,
        "adapter_run_mode": args.adapter_run_mode,
        "seed": args.seed,
    }
    if args.experiments:
        overrides["experiments"] = [x.strip() for x in args.experiments.split(",") if x.strip()]

    cfg = load_config(args.config, overrides=overrides)

    if args.command in {"train", "all"}:
        from .train import run_training_pipeline

        run_training_pipeline(cfg)
    if args.command in {"attack", "all"}:
        from .attack import run_attack_pipeline

        run_attack_pipeline(cfg)
    if args.command == "infer":
        from .inference import run_inference_pipeline, run_single_image_inference

        if args.input_image and args.input_dir:
            raise ValueError("Use only one of --input-image or --input-dir")
        if args.input_image:
            run_single_image_inference(
                cfg,
                input_image=Path(args.input_image),
                output_dir=Path(args.output_dir) if args.output_dir else None,
                num_images_per_prompt=args.num_images_per_prompt,
                seed=args.seed,
            )
        elif args.input_dir:
            run_inference_pipeline(
                cfg,
                input_dir=Path(args.input_dir),
                output_dir=Path(args.output_dir) if args.output_dir else None,
                num_images_per_prompt=args.num_images_per_prompt,
                max_images_per_identity=args.max_images_per_identity,
                seed=args.seed,
            )
        else:
            raise ValueError("Provide either --input-image or --input-dir for infer")


if __name__ == "__main__":
    main()
