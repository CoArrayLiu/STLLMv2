from __future__ import annotations

import math
import unittest

import pandas as pd
import torch

from train_pems_joint_gradient_conflict import (
    DATASETS,
    JointPEMSQKTransformer,
    Trainer,
    gradient_conflicts,
    gradient_group,
    parse_args,
    summarize_conflicts,
)
from util import StandardScaler


class PEMSJointGradientConflictTest(unittest.TestCase):
    @staticmethod
    def default_args():
        args = parse_args([])
        args.device = "cpu"
        args.precision = "fp32"
        return args

    def test_default_model_is_within_8m_budget(self) -> None:
        model = JointPEMSQKTransformer(self.default_args())
        self.assertEqual(model.param_num(), 7_999_920)
        self.assertEqual(model.count_trainable_params(), 7_999_920)
        shared = sum(p.numel() for _, p in model.shared_named_parameters())
        task_specific = sum(
            p.numel()
            for name, p in model.named_parameters()
            if not name.startswith("backbone.")
        )
        self.assertEqual(shared, 7_948_696)
        self.assertEqual(task_specific, 51_224)
        self.assertEqual(shared + task_specific, model.param_num())

    def test_every_shared_parameter_has_an_architecture_group(self) -> None:
        model = JointPEMSQKTransformer(self.default_args())
        groups = {
            gradient_group(name) for name, _ in model.shared_named_parameters()
        }
        self.assertNotIn("other_shared", groups)
        self.assertIn("input.value_projection", groups)
        self.assertIn("input.temporal_embedding", groups)
        self.assertIn("input.fusion_projection", groups)
        self.assertIn("output.final_norm", groups)
        for layer in range(6):
            prefix = f"layer_{layer:02d}."
            self.assertIn(prefix + "attention.query", groups)
            self.assertIn(prefix + "attention.key", groups)
            self.assertIn(prefix + "attention.value", groups)
            self.assertIn(prefix + "attention.output", groups)
            self.assertIn(prefix + "normalization", groups)
            self.assertIn(prefix + "ffn", groups)

    def test_forward_uses_dataset_specific_heads(self) -> None:
        args = self.default_args()
        args.dropout = 0.0
        model = JointPEMSQKTransformer(args).eval()
        inputs = torch.randn(1, 3, 170, 12)
        inputs[:, 1].uniform_(0, 1)
        inputs[:, 2].random_(0, 7)
        with torch.no_grad():
            outputs = {dataset: model(dataset, inputs) for dataset in DATASETS}
        self.assertEqual(tuple(outputs["pems03"].shape), (1, 12, 170, 1))
        self.assertEqual(tuple(outputs["pems04"].shape), (1, 12, 170, 1))
        self.assertIsNot(
            model.prediction_heads["pems03"].weight,
            model.prediction_heads["pems04"].weight,
        )

    def test_gradient_conflict_probe_is_finite_on_small_model(self) -> None:
        args = self.default_args()
        args.d_model = 32
        args.num_heads = 4
        args.num_layers = 2
        args.ffn_dim = 64
        args.embedding_dim = 16
        args.dropout = 0.0
        scalers = {
            dataset: StandardScaler(mean=0.0, std=1.0) for dataset in DATASETS
        }
        trainer = Trainer(args, scalers, torch.device("cpu"))
        paired = {}
        for dataset in DATASETS:
            inputs = torch.randn(1, 3, 170, 12)
            inputs[:, 1].uniform_(0, 1)
            inputs[:, 2].random_(0, 7)
            targets = torch.rand(1, 170, 12) + 0.1
            paired[dataset] = (inputs, targets)
        rows = gradient_conflicts(trainer, [paired], epoch=0)
        self.assertEqual(len(rows), 16)
        self.assertTrue(all(math.isfinite(float(row["cosine"])) for row in rows))
        self.assertTrue(
            all(0.0 <= float(row["conflict_mass"]) <= 1.0 for row in rows)
        )

    def test_conflict_summary_ranks_lowest_cosine_first(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "epoch": 1,
                    "probe_batch": 0,
                    "group": "less_conflict",
                    "parameters": 2,
                    "cosine": 0.5,
                    "negative_cosine": 0,
                    "sign_conflict_rate": 0.2,
                    "conflict_mass": 0.1,
                    "pems03_grad_norm": 1.0,
                    "pems04_grad_norm": 1.0,
                },
                {
                    "epoch": 1,
                    "probe_batch": 0,
                    "group": "more_conflict",
                    "parameters": 2,
                    "cosine": -0.5,
                    "negative_cosine": 1,
                    "sign_conflict_rate": 0.8,
                    "conflict_mass": 0.9,
                    "pems03_grad_norm": 1.0,
                    "pems04_grad_norm": 1.0,
                },
            ]
        )
        summary = summarize_conflicts(frame)
        self.assertEqual(summary.iloc[0]["group"], "more_conflict")


if __name__ == "__main__":
    unittest.main()

