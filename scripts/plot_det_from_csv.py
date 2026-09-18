#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from statistics import NormalDist

import pandas as pd

REQUIRED_COLUMNS = {"far", "fnmr", "method", "comparison_type"}
DEFAULT_TYPE_I_FILENAME = "typeI_det_curve.png"
DEFAULT_TYPE_II_FILENAME = "typeII_det_curve.png"
COMPARISON_LABELS = {
    "type_i": "Type I DET Curve",
    "type_ii": "Type II DET Curve",
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Type I/Type II DET curve plots from an existing det_curve.csv file."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("det_curve.csv"),
        help="Path to det_curve.csv (default: det_curve.csv)",
    )
    parser.add_argument(
        "--mode",
        choices=("standard", "log", "current"),
        default="current",
        help=(
            "Plot mode: standard=probit DET transform, log=log-scale FAR/FNMR axes, "
            "current=linear FAR/FNMR axes."
        ),
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-6,
        help=(
            "Small positive value used when mode is 'log' (zero-safe clipping) and "
            "'standard' (clip to [epsilon, 1-epsilon] before inverse normal CDF)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for output plots (default: current directory).",
    )
    parser.add_argument(
        "--type-i-output",
        type=Path,
        default=None,
        help=f"Optional explicit output path for Type I plot (default: <output-dir>/{DEFAULT_TYPE_I_FILENAME}).",
    )
    parser.add_argument(
        "--type-ii-output",
        type=Path,
        default=None,
        help=f"Optional explicit output path for Type II plot (default: <output-dir>/{DEFAULT_TYPE_II_FILENAME}).",
    )
    return parser.parse_args(argv)


def _validate_required_columns(det_curve_df: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(det_curve_df.columns))
    if missing:
        raise ValueError(f"CSV is missing required column(s): {', '.join(missing)}")


def _validate_epsilon(epsilon: float) -> None:
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    if epsilon >= 0.5:
        raise ValueError("epsilon must be < 0.5")


def _comparison_subset(det_curve_df: pd.DataFrame, comparison_type: str) -> pd.DataFrame:
    subset = det_curve_df[det_curve_df["comparison_type"] == comparison_type].copy()
    if subset.empty:
        available = sorted(det_curve_df["comparison_type"].dropna().astype(str).unique().tolist())
        raise ValueError(
            f"No rows found for comparison_type='{comparison_type}'. "
            f"Available comparison_type values: {available}"
        )
    return subset


def _sort_method_points(method_df: pd.DataFrame) -> pd.DataFrame:
    if "thresholds" in method_df.columns:
        return method_df.sort_values(["thresholds", "far", "fnmr"], kind="mergesort")
    return method_df.sort_values(["far", "fnmr"], kind="mergesort")


def _probit(series: pd.Series, epsilon: float) -> pd.Series:
    normal = NormalDist()
    clipped = series.clip(lower=epsilon, upper=1 - epsilon)
    return clipped.map(normal.inv_cdf)


def _plot_det_curve(
    det_curve_df: pd.DataFrame,
    output_path: Path,
    title: str,
    mode: str,
    epsilon: float,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for method, method_df in det_curve_df.groupby("method"):
        ordered = _sort_method_points(method_df)
        x_values = ordered["far"]
        y_values = ordered["fnmr"]
        if mode == "standard":
            x_values = _probit(x_values, epsilon)
            y_values = _probit(y_values, epsilon)
        elif mode == "log":
            x_values = x_values.clip(lower=epsilon)
            y_values = y_values.clip(lower=epsilon)
        ax.plot(x_values, y_values, label=method)

    if mode == "standard":
        ax.set_xlabel("FAR (probit)")
        ax.set_ylabel("FNMR (probit)")
    elif mode == "log":
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("FAR (log scale)")
        ax.set_ylabel("FNMR (log scale)")
    else:
        ax.set_xlabel("FAR")
        ax.set_ylabel("FNMR")

    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close(fig)


def _resolve_output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    type_i_output = args.type_i_output or (args.output_dir / DEFAULT_TYPE_I_FILENAME)
    type_ii_output = args.type_ii_output or (args.output_dir / DEFAULT_TYPE_II_FILENAME)
    return Path(type_i_output), Path(type_ii_output)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    _validate_epsilon(args.epsilon)

    det_curve_df = pd.read_csv(args.csv)
    _validate_required_columns(det_curve_df)

    type_i_df = _comparison_subset(det_curve_df, "type_i")
    type_ii_df = _comparison_subset(det_curve_df, "type_ii")
    type_i_output, type_ii_output = _resolve_output_paths(args)

    _plot_det_curve(
        type_i_df,
        output_path=type_i_output,
        title=COMPARISON_LABELS["type_i"],
        mode=args.mode,
        epsilon=args.epsilon,
    )
    _plot_det_curve(
        type_ii_df,
        output_path=type_ii_output,
        title=COMPARISON_LABELS["type_ii"],
        mode=args.mode,
        epsilon=args.epsilon,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"Error: {exc}") from exc
