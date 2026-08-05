from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import pandas as pd
import torch

import train_pems_joint_gradient_conflict as core
from util import StandardScaler


THREE_DATASETS = ("pems03", "pems04", "pems08")


class PEMS030408GradientConflictTest(unittest.TestCase):
    @staticmethod
    def default_args():
        args = core.parse_args([])
        args.device = "cpu"
        args.precision = "fp32"
        return args

    def test_three_task_model_parameter_count(self) -> None:
        with patch.object(core, "DATASETS", THREE_DATASETS):
            model = core.JointPEMSQKTransformer(self.default_args())
            self.assertEqual(model.param_num(), 8_025_532)
            shared = sum(p.numel() for _, p in model.shared_named_parameters())
            task_specific = model.param_num() - shared
            self.assertEqual(shared, 7_948_696)
            self.assertEqual(task_specific, 76_836)
            self.assertEqual(set(model.prediction_heads), set(THREE_DATASETS))
            self.assertEqual(set(model.node_embeddings), set(THREE_DATASETS))

    def test_three_task_probe_emits_all_three_pairs(self) -> None:
        args = self.default_args()
        args.d_model = 32
        args.num_heads = 4
        args.num_layers = 2
        args.ffn_dim = 64
        args.embedding_dim = 16
        args.dropout = 0.0
        with patch.object(core, "DATASETS", THREE_DATASETS):
            scalers = {
                dataset: StandardScaler(mean=0.0, std=1.0)
                for dataset in THREE_DATASETS
            }
            trainer = core.Trainer(args, scalers, torch.device("cpu"))
            probe = {}
            for dataset in THREE_DATASETS:
                inputs = torch.randn(1, 3, 170, 12)
                inputs[:, 1].uniform_(0, 1)
                inputs[:, 2].random_(0, 7)
                targets = torch.rand(1, 170, 12) + 0.1
                probe[dataset] = (inputs, targets)
            rows = core.gradient_conflicts(trainer, [probe], epoch=3)

        self.assertEqual(len(rows), 48)
        self.assertEqual(
            {row["task_pair"] for row in rows},
            {"pems03__pems04", "pems03__pems08", "pems04__pems08"},
        )
        self.assertTrue(all(math.isfinite(float(row["cosine"])) for row in rows))

    def test_training_step_captures_exact_task_gradients_and_updates(self) -> None:
        args = self.default_args()
        args.d_model = 32
        args.num_heads = 4
        args.num_layers = 2
        args.ffn_dim = 64
        args.embedding_dim = 16
        args.dropout = 0.1
        with patch.object(core, "DATASETS", THREE_DATASETS):
            scalers = {
                dataset: StandardScaler(mean=0.0, std=1.0)
                for dataset in THREE_DATASETS
            }
            trainer = core.Trainer(args, scalers, torch.device("cpu"))
            batches = {}
            for dataset in THREE_DATASETS:
                inputs = torch.randn(1, 3, 170, 12)
                inputs[:, 1].uniform_(0, 1)
                inputs[:, 2].random_(0, 7)
                targets = torch.rand(1, 170, 12) + 0.1
                batches[dataset] = (inputs, targets)
            before = trainer.model.backbone.start_conv.weight.detach().clone()
            _, _, _, rows = trainer.train_step(
                batches,
                conflict_context=(2, 0, 3, 10),
            )
            after = trainer.model.backbone.start_conv.weight.detach()

        self.assertEqual(len(rows), 48)
        self.assertTrue(all(row["source"] == "train" for row in rows))
        self.assertTrue(all(int(row["epoch_step"]) == 3 for row in rows))
        self.assertTrue(all(int(row["global_step"]) == 10 for row in rows))
        self.assertTrue(
            all(0.0 <= float(row["tug_of_war_strength"]) <= 1.0 for row in rows)
        )
        self.assertFalse(torch.equal(before, after))


    def test_frequency_summaries_have_clear_denominators(self) -> None:
        rows = []
        pairs = (
            ("pems03", "pems04"),
            ("pems03", "pems08"),
            ("pems04", "pems08"),
        )
        for probe in range(2):
            for pair_index, (task_a, task_b) in enumerate(pairs):
                conflict = int(probe == 0 and pair_index == 0)
                rows.append(
                    {
                        "epoch": 5,
                        "probe_batch": probe,
                        "group": "layer_00.attention.query",
                        "parameters": 10,
                        "task_a": task_a,
                        "task_b": task_b,
                        "task_pair": f"{task_a}__{task_b}",
                        "cosine": -0.2 if conflict else 0.2,
                        "negative_cosine": conflict,
                        "sign_conflict_rate": 0.5,
                        "conflict_mass": 0.5,
                        "task_a_grad_norm": 1.0,
                        "task_b_grad_norm": 1.0,
                    }
                )
        frame = pd.DataFrame(rows)
        module = core.summarize_conflicts(frame).iloc[0]
        pairwise = core.summarize_pairwise_conflicts(frame)

        self.assertEqual(int(module["probes"]), 2)
        self.assertEqual(int(module["pair_comparisons"]), 6)
        self.assertAlmostEqual(float(module["any_pair_conflict_rate"]), 0.5)
        self.assertAlmostEqual(float(module["pairwise_conflict_rate"]), 1 / 6)
        first_pair = pairwise[pairwise["task_pair"] == "pems03__pems04"].iloc[0]
        self.assertAlmostEqual(float(first_pair["conflict_frequency"]), 0.5)


if __name__ == "__main__":
    unittest.main()
