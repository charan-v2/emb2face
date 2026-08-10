from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from emb2face import score_run


class _FakeFace:
    def __init__(self, embedding, face_count=1, confidence=0.9):
        self.embedding = embedding
        self.face_count = face_count
        self.confidence = confidence


class ScoreRunTests(unittest.TestCase):
    def test_extract_face_rows_treats_multi_face_value_error_as_failed_row(self):
        rows = [
            {"source_path": "/tmp/source-ok.jpg", "identity": "alice"},
            {"source_path": "/tmp/source-multi.jpg", "identity": "bob"},
        ]
        cache: dict[str, object] = {}

        original_extract = score_run.extract_face_embedding

        def fake_extract_face_embedding(path_str, *, detector, embedder, require_single_face):
            if path_str.endswith("source-multi.jpg"):
                raise ValueError("Expected exactly one face but detected 2")
            return _FakeFace(embedding=[1.0, 0.0, 0.0], face_count=1, confidence=0.8)

        try:
            score_run.extract_face_embedding = fake_extract_face_embedding
            valid_df, failed_df = score_run._extract_face_rows(
                rows,
                path_key="source_path",
                role="source",
                detector=object(),
                embedder=object(),
                require_single_face=True,
                cache=cache,
            )
        finally:
            score_run.extract_face_embedding = original_extract

        self.assertEqual(len(valid_df), 1)
        self.assertEqual(len(failed_df), 1)
        self.assertEqual(failed_df.iloc[0]["reason"], "source_face_extraction_failed")
        self.assertIn("Expected exactly one face but detected 2", str(failed_df.iloc[0]["error"]))

        self.assertIsInstance(valid_df, pd.DataFrame)

    def test_build_type_i_pairs_uses_recon_embedding_column(self):
        source_valid_df = pd.DataFrame(
            [
                {
                    "source_row_index": 0,
                    "identity": "alice",
                    "source_path": "/tmp/source-ok.jpg",
                    "source_embedding": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
                }
            ]
        )
        valid_recon_df = pd.DataFrame(
            [
                {
                    "source_row_index": 0,
                    "recon_row_index": 0,
                    "identity": "alice",
                    "recon_path": "/tmp/recon-ok.jpg",
                    "recon_embedding": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
                }
            ]
        )

        source_pos_by_row_index, _ = score_run._build_source_lookup(source_valid_df)
        pair_df = score_run._build_type_i_pairs(
            valid_recon_df,
            source_valid_df,
            source_pos_by_row_index,
            impostors_by_source_row={0: []},
        )

        self.assertEqual(len(pair_df), 1)
        self.assertEqual(pair_df.iloc[0]["comparison_type"], "type_i_genuine")
        self.assertAlmostEqual(float(pair_df.iloc[0]["score"]), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
