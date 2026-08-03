from __future__ import annotations

import argparse
from pathlib import Path
from .config import load_config


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="emb2face")
    parser.add_argument("command", choices=["train", "attack", "infer", "score", "all"])
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--runmode", type=str, default=None)
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--eval-source", type=str, default=None)
    parser.add_argument("--adapter-run-mode", type=str, default=None)
    parser.add_argument("--inference-adapter-checkpoint", type=str, default=None)
    parser.add_argument("--inference-adapter-checkpoints", type=str, default=None, help="Comma-separated adapter checkpoints")
    parser.add_argument("--experiments", type=str, default=None, help="Comma-separated experiment list")
    parser.add_argument("--input-dir", type=str, default=None, help="Input directory for inference")
    parser.add_argument("--input-image", type=str, default=None, help="Single image for inference")
    parser.add_argument("--input-run-dir", type=str, default=None, help="Previous inference run directory for scoring")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for inference")
    parser.add_argument("--num-images-per-prompt", type=int, default=None)
    parser.add_argument("--num-identities", type=int, default=None)
    parser.add_argument("--images-per-identity", type=int, default=None)
    parser.add_argument("--inference-max-yaw-degrees", type=float, default=None)
    parser.add_argument("--inference-pose-require-single-face", dest="inference_pose_require_single_face", action="store_true")
    parser.add_argument("--no-inference-pose-require-single-face", dest="inference_pose_require_single_face", action="store_false")
    parser.set_defaults(inference_pose_require_single_face=None)
    parser.add_argument("--score-detector-backend", type=str, default=None)
    parser.add_argument("--score-embedder-backend", type=str, default=None)
    parser.add_argument("--score-methods", type=str, default=None, help="Comma-separated reconstruction methods to score")
    parser.add_argument("--save-comparison-figures", dest="save_comparison_figures", action="store_true")
    parser.add_argument("--no-save-comparison-figures", dest="save_comparison_figures", action="store_false")
    parser.set_defaults(save_comparison_figures=None)
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
        "inference_adapter_checkpoint": args.inference_adapter_checkpoint,
        "inference_max_yaw_degrees": args.inference_max_yaw_degrees,
        "inference_pose_require_single_face": args.inference_pose_require_single_face,
        "score_detector_backend": args.score_detector_backend,
        "score_embedder_backend": args.score_embedder_backend,
        "seed": args.seed,
    }
    if args.inference_adapter_checkpoints:
        overrides["inference_adapter_checkpoints"] = [x.strip() for x in args.inference_adapter_checkpoints.split(",") if x.strip()]
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

        input_dir = Path(args.input_dir) if args.input_dir else cfg.get("inference_input_dir")
        input_image = Path(args.input_image) if args.input_image else None
        output_dir = Path(args.output_dir) if args.output_dir else cfg.get("inference_output_dir")

        if input_image and input_dir:
            raise ValueError("Use only one of --input-image or --input-dir")
        if input_image:
            run_single_image_inference(
                cfg,
                input_image=input_image,
                output_dir=output_dir,
                num_images_per_prompt=args.num_images_per_prompt,
                seed=args.seed,
            )
        elif input_dir:
            run_inference_pipeline(
                cfg,
                input_dir=Path(input_dir),
                output_dir=output_dir,
                num_identities=args.num_identities,
                images_per_identity=args.images_per_identity,
                num_images_per_prompt=args.num_images_per_prompt,
                save_comparison_figures=args.save_comparison_figures,
                seed=args.seed,
            )
        else:
            raise ValueError("Provide either --input-image or --input-dir, or set inference_input_dir in the config")
    if args.command == "score":
        from .score_run import run_score_pipeline

        input_run_dir = Path(args.input_run_dir) if args.input_run_dir else cfg.get("score_input_run_dir")
        selected_methods = None
        if args.score_methods:
            selected_methods = [x.strip() for x in args.score_methods.split(",") if x.strip()]
        result = run_score_pipeline(
            cfg,
            input_run_dir=Path(input_run_dir) if input_run_dir else None,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            selected_methods=selected_methods,
        )
        print("Results written to:", result["output_dir"])
        print(result["summary"])


if __name__ == "__main__":
    main()
