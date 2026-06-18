from __future__ import annotations

import argparse
from .config import load_config


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="emb2face")
    parser.add_argument("command", choices=["train", "attack", "all"])
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--runmode", type=str, default=None)
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--eval-source", type=str, default=None)
    parser.add_argument("--adapter-run-mode", type=str, default=None)
    parser.add_argument("--experiments", type=str, default=None, help="Comma-separated experiment list")
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


if __name__ == "__main__":
    main()
