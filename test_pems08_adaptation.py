"""Functional smoke tests for the PEMS08 QK adaptation."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import util
from dataset_config import get_dataset_config
from model_ST_Transformer_adaptive import STTransformerAdaptive
from train_transformer_ablation import load_graph, parse_args, validate_args


class PEMS08AdaptationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset_dir = Path("data/pems08")
        with (cls.dataset_dir / "manifest.json").open(encoding="utf-8") as file:
            cls.manifest = json.load(file)

    def test_dataset_config_is_explicit(self) -> None:
        config = get_dataset_config("pems08")
        self.assertEqual(config.num_nodes, 170)
        self.assertEqual(config.time_slots, 288)
        self.assertEqual(config.dataset_path, self.dataset_dir)
        self.assertEqual(config.loss_space, "standardized")
        self.assertEqual(config.recommended_training["batch_size"], 512)
        self.assertNotIn("adaptive", config.graphs)

    def test_chronological_splits_are_disjoint(self) -> None:
        previous_end = 0
        for name in ("train", "val", "test"):
            split = self.manifest["splits"][name]
            self.assertEqual(split["raw_start_index"], previous_end)
            self.assertLess(split["last_target_index"], split["raw_end_index_exclusive"])
            previous_end = split["raw_end_index_exclusive"]
        self.assertEqual(previous_end, self.manifest["source_shape"][0])
        self.assertFalse(
            self.manifest["leakage_checks"][
                "input_or_target_timestamp_shared_across_splits"
            ]
        )

    def test_window_values_and_time_features(self) -> None:
        source = np.load("data/st_data/pems08/pems08.npz")["data"][..., 0]
        with np.load(self.dataset_dir / "train.npz") as archive:
            x = archive["x"]
            y = archive["y"]
            sample_start = archive["sample_start"]
            target_start = archive["target_start"]
            target_end = archive["target_end"]
        self.assertEqual(x.shape, (10633, 12, 170, 3))
        self.assertEqual(y.shape, (10633, 12, 170, 1))
        np.testing.assert_array_equal(x[0, :, :, 0], source[:12])
        np.testing.assert_array_equal(y[0, :, :, 0], source[12:24])
        np.testing.assert_array_equal(x[-1, :, :, 0], source[10632:10644])
        np.testing.assert_array_equal(y[-1, :, :, 0], source[10644:10656])
        np.testing.assert_allclose(x[0, :, 0, 1], np.arange(12) / 288)
        np.testing.assert_array_equal(x[0, :, 0, 2], np.full(12, 4))
        self.assertEqual(sample_start[0], 0)
        self.assertEqual(target_start[0], 12)
        self.assertEqual(target_end[-1], 10655)

    def test_loader_scaler_matches_manifest(self) -> None:
        dataset = util.load_dataset(
            str(self.dataset_dir),
            batch_size=64,
            valid_batch_size=64,
            test_batch_size=64,
            expected_num_nodes=170,
            expected_input_len=12,
            expected_output_len=12,
            expected_input_dim=3,
        )
        expected = self.manifest["scaler"]
        self.assertEqual(dataset["val_loader"].size, 3433)
        self.assertEqual(dataset["test_loader"].size, 3721)
        self.assertAlmostEqual(float(dataset["scaler"].mean), expected["mean"], places=4)
        self.assertAlmostEqual(float(dataset["scaler"].std), expected["std"], places=4)
        restored = dataset["scaler"].inverse_transform(
            dataset["x_test"][0, :, :, 0]
        )
        self.assertTrue(np.isfinite(restored).all())

    def test_qk_rejects_graph_and_graph_mode_requires_type(self) -> None:
        with mock.patch.object(
            sys,
            "argv",
            ["train_transformer_ablation.py", "--data", "pems08"],
        ):
            qk_args = parse_args()
        validate_args(qk_args)
        self.assertEqual(load_graph(qk_args), (None, None, None))

        qk_args.graph_type = "physical"
        with self.assertRaisesRegex(ValueError, "qk mode does not accept"):
            validate_args(qk_args)
        qk_args.attention_mode = "graph"
        qk_args.graph_type = None
        with self.assertRaisesRegex(ValueError, "--graph_type is required"):
            validate_args(qk_args)

    def test_npy_graph_and_qk_forward_shapes(self) -> None:
        graph = util.load_graph_data("data/st_data/pems08/pems08_adj.npy")
        self.assertEqual(graph.shape, (170, 170))
        model = STTransformerAdaptive(
            adaptive_graph=None,
            time_slots=288,
            input_dim=3,
            num_nodes=170,
            input_len=12,
            output_len=12,
            attention_mode="qk",
            d_model=48,
            num_heads=6,
            num_layers=1,
            ffn_dim=96,
            embedding_dim=16,
            dropout=0.0,
        )
        history = torch.zeros(2, 3, 170, 12)
        history[:, 1] = torch.arange(12).view(1, 1, 12) / 288
        history[:, 2] = 4
        with torch.no_grad():
            output = model(history)
        self.assertEqual(tuple(output.shape), (2, 12, 170, 1))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
