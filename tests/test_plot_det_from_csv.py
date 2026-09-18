from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")


def _load_plot_script_module():
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "plot_det_from_csv.py"
    spec = importlib.util.spec_from_file_location("plot_det_from_csv", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


plot_det_from_csv = _load_plot_script_module()


class PlotDetFromCsvTests(unittest.TestCase):
    def test_validate_required_columns_raises_for_missing_columns(self):
        det_curve_df = pd.DataFrame([{"far": 0.1, "fnmr": 0.2, "comparison_type": "type_i"}])

        with self.assertRaisesRegex(ValueError, "method"):
            plot_det_from_csv._validate_required_columns(det_curve_df)

    def test_comparison_subset_raises_when_type_data_is_missing(self):
        det_curve_df = pd.DataFrame(
            [
                {"far": 0.1, "fnmr": 0.2, "method": "m1", "comparison_type": "type_i"},
            ]
        )

        with self.assertRaisesRegex(ValueError, "type_ii"):
            plot_det_from_csv._comparison_subset(det_curve_df, "type_ii")

    def test_main_creates_default_type_plot_files(self):
        det_curve_df = pd.DataFrame(
            [
                {"far": 0.1, "fnmr": 0.2, "method": "m1", "comparison_type": "type_i", "thresholds": 0.5},
                {"far": 0.2, "fnmr": 0.1, "method": "m1", "comparison_type": "type_i", "thresholds": 0.6},
                {"far": 0.15, "fnmr": 0.3, "method": "m1", "comparison_type": "type_ii", "thresholds": 0.5},
                {"far": 0.25, "fnmr": 0.25, "method": "m1", "comparison_type": "type_ii", "thresholds": 0.6},
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            csv_path = tmp_path / "det_curve.csv"
            output_dir = tmp_path / "plots"
            det_curve_df.to_csv(csv_path, index=False)

            plot_det_from_csv.main(
                [
                    "--csv",
                    str(csv_path),
                    "--output-dir",
                    str(output_dir),
                    "--mode",
                    "current",
                ]
            )

            type_i_plot = output_dir / "typeI_det_curve.png"
            type_ii_plot = output_dir / "typeII_det_curve.png"
            self.assertTrue(type_i_plot.exists())
            self.assertTrue(type_ii_plot.exists())
            self.assertGreater(type_i_plot.stat().st_size, 0)
            self.assertGreater(type_ii_plot.stat().st_size, 0)

    def test_main_log_mode_handles_zero_values_with_epsilon(self):
        det_curve_df = pd.DataFrame(
            [
                {"far": 0.0, "fnmr": 0.2, "method": "m1", "comparison_type": "type_i"},
                {"far": 0.2, "fnmr": 0.0, "method": "m1", "comparison_type": "type_i"},
                {"far": 0.0, "fnmr": 0.3, "method": "m1", "comparison_type": "type_ii"},
                {"far": 0.3, "fnmr": 0.0, "method": "m1", "comparison_type": "type_ii"},
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            csv_path = tmp_path / "det_curve.csv"
            output_dir = tmp_path / "plots"
            det_curve_df.to_csv(csv_path, index=False)

            plot_det_from_csv.main(
                [
                    "--csv",
                    str(csv_path),
                    "--output-dir",
                    str(output_dir),
                    "--mode",
                    "log",
                    "--epsilon",
                    "1e-4",
                ]
            )

            self.assertTrue((output_dir / "typeI_det_curve.png").exists())
            self.assertTrue((output_dir / "typeII_det_curve.png").exists())

    def test_main_standard_mode_handles_boundary_rates(self):
        det_curve_df = pd.DataFrame(
            [
                {"far": 0.0, "fnmr": 1.0, "method": "m1", "comparison_type": "type_i"},
                {"far": 1.0, "fnmr": 0.0, "method": "m1", "comparison_type": "type_i"},
                {"far": 0.0, "fnmr": 1.0, "method": "m1", "comparison_type": "type_ii"},
                {"far": 1.0, "fnmr": 0.0, "method": "m1", "comparison_type": "type_ii"},
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            csv_path = tmp_path / "det_curve.csv"
            output_dir = tmp_path / "plots"
            det_curve_df.to_csv(csv_path, index=False)

            plot_det_from_csv.main(
                [
                    "--csv",
                    str(csv_path),
                    "--output-dir",
                    str(output_dir),
                    "--mode",
                    "standard",
                ]
            )

            self.assertTrue((output_dir / "typeI_det_curve.png").exists())
            self.assertTrue((output_dir / "typeII_det_curve.png").exists())


if __name__ == "__main__":
    unittest.main()
