from __future__ import annotations

import json
import math
import unittest
from itertools import combinations
from unittest.mock import patch

import torch

import train_all_datasets_20m_gradient_conflict as entry
import train_pems_joint_gradient_conflict as core
from dataset_config import get_dataset_config


class AllDatasets20MGradientConflictTest(unittest.TestCase):
    @staticmethod
    def model_args():
        args = core.parse_args([])
        args.device = "cpu"
        args.precision = "fp32"
        args.d_model = 512
        args.num_heads = 8
        args.num_layers = 6
        args.ffn_dim = 2064
        args.embedding_dim = 200
        args.dropout = 0.0
        return args

    def test_every_local_dataset_is_registered_and_prepared(self) -> None:
        self.assertEqual(len(entry.DATASETS), 10)
        for dataset in entry.DATASETS:
            config = get_dataset_config(dataset)
            for split in ("train", "val", "test"):
                self.assertTrue((config.dataset_path / f"{split}.npz").is_file())
            if dataset in {"urbanev", "shenzhen", "sd"}:
                with (config.dataset_path / "manifest.json").open() as file:
                    manifest = json.load(file)
                self.assertEqual(manifest["num_nodes"], config.num_nodes)
                self.assertEqual(manifest["time_slots"], config.time_slots)
                self.assertTrue(
                    manifest["leakage_checks"][
                        "input_or_target_timestamp_shared_across_splits"
                    ]
                    is False
                )

    def test_existing_architecture_has_exact_20m_budget(self) -> None:
        with patch.object(core, "DATASETS", entry.DATASETS):
            model = core.JointPEMSQKTransformer(self.model_args())
        self.assertEqual(model.param_num(), 19_996_800)
        shared = sum(p.numel() for _, p in model.shared_named_parameters())
        self.assertEqual(shared, 19_387_840)
        self.assertEqual(model.param_num() - shared, 608_960)
        self.assertEqual(model.backbone.time_slots, 288)
        self.assertEqual(set(model.node_embeddings), set(entry.DATASETS))
        self.assertEqual(set(model.prediction_heads), set(entry.DATASETS))

    def test_representative_node_counts_forward(self) -> None:
        args = self.model_args()
        with patch.object(core, "DATASETS", entry.DATASETS):
            model = core.JointPEMSQKTransformer(args).eval()
            for dataset in ("pems03", "urbanev", "sd"):
                nodes = get_dataset_config(dataset).num_nodes
                inputs = torch.randn(1, 3, nodes, 12)
                inputs[:, 1].uniform_(0, 1)
                inputs[:, 2].random_(0, 7)
                with torch.no_grad():
                    output = model(dataset, inputs)
                self.assertEqual(tuple(output.shape), (1, 12, nodes, 1))

    def test_all_45_task_pairs_are_analyzed(self) -> None:
        parameter = torch.nn.Parameter(torch.zeros(5))
        named_parameters = [("backbone.start_conv.weight", parameter)]
        task_gradients = {
            dataset: [torch.randn_like(parameter)] for dataset in entry.DATASETS
        }
        task_losses = {
            dataset: float(index + 1) for index, dataset in enumerate(entry.DATASETS)
        }
        with patch.object(core, "DATASETS", entry.DATASETS):
            rows = core.conflict_rows_from_task_gradients(
                named_parameters,
                task_gradients,
                task_losses,
                epoch=1,
                probe_batch=0,
                epoch_step=2,
                global_step=2,
            )
            norms = core.training_task_gradient_norms(
                __import__("pandas").DataFrame(rows)
            )
        expected_pairs = {
            f"{first}__{second}"
            for first, second in combinations(entry.DATASETS, 2)
        }
        self.assertEqual(len(rows), math.comb(10, 2))
        self.assertEqual({row["task_pair"] for row in rows}, expected_pairs)
        self.assertEqual(len(norms), 10)


if __name__ == "__main__":
    unittest.main()
