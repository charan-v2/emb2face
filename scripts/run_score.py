from __future__ import annotations

import argparse
from pathlib import Path

from emb2face.config import load_config
from emb2face.score_run import run_score_pipeline


def _parse_args():
    parser = argparse.ArgumentParser(description="Score an existing inference run folder")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--input-run-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--score-detector-backend", type=str, default=None)
    parser.add_argument("--score-embedder-backend", type=str, default=None)
    parser.add_argument("--score-methods", type=str, default=None)
    return parser.parse_args()


def main():
    args = _parse_args()
    cfg = load_config(args.config)
    if args.score_detector_backend:
        cfg["score_detector_backend"] = args.score_detector_backend
    if args.score_embedder_backend:
        cfg["score_embedder_backend"] = args.score_embedder_backend
    selected_methods = None
    if args.score_methods:
        selected_methods = [x.strip() for x in args.score_methods.split(",") if x.strip()]
    result = run_score_pipeline(
        cfg=cfg,
        input_run_dir=Path(args.input_run_dir),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        selected_methods=selected_methods,
    )
    print("Results written to:", result["output_dir"])
    print(result["summary"])


if __name__ == "__main__":
    main()
