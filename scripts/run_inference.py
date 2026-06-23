from __future__ import annotations

import argparse
from pathlib import Path

from emb2face.config import load_config
from emb2face.inference import run_inference_pipeline, run_single_image_inference


def _parse_args():
    parser = argparse.ArgumentParser(description="Run Arc2Face inference on a folder or single image")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--input-image", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--inference-adapter-checkpoint", type=str, default=None)
    parser.add_argument("--num-images-per-prompt", type=int, default=None)
    parser.add_argument("--num-identities", type=int, default=None)
    parser.add_argument("--images-per-identity", type=int, default=None)
    parser.add_argument("--save-comparison-figures", dest="save_comparison_figures", action="store_true")
    parser.add_argument("--no-save-comparison-figures", dest="save_comparison_figures", action="store_false")
    parser.set_defaults(save_comparison_figures=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main():
    args = _parse_args()
    cfg = load_config(args.config)
    if args.inference_adapter_checkpoint:
        cfg["inference_adapter_checkpoint"] = Path(args.inference_adapter_checkpoint).expanduser()
    if args.input_image and args.input_dir:
        raise ValueError("Use only one of --input-image or --input-dir")
    if args.input_image:
        result = run_single_image_inference(
            cfg=cfg,
            input_image=Path(args.input_image),
            output_dir=Path(args.output_dir) if args.output_dir else None,
            num_images_per_prompt=args.num_images_per_prompt,
            seed=args.seed,
        )
    elif args.input_dir:
        result = run_inference_pipeline(
            cfg=cfg,
            input_dir=Path(args.input_dir),
            output_dir=Path(args.output_dir) if args.output_dir else None,
            num_identities=args.num_identities,
            images_per_identity=args.images_per_identity,
            num_images_per_prompt=args.num_images_per_prompt,
            save_comparison_figures=args.save_comparison_figures,
            seed=args.seed,
        )
    else:
        raise ValueError("Provide either --input-image or --input-dir")
    print("Adapter:", result["adapter_path"])
    print("Results written to:", result["output_dir"])
    if "summary" in result:
        print("Summary:")
        print(result["summary"])
    print(result["results"])


if __name__ == "__main__":
    main()
