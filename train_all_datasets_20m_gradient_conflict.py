#!/usr/bin/env python3
"""Train the existing 20M QK Transformer jointly on every local dataset.

The architecture is unchanged: one shared QK Transformer backbone plus
per-dataset node embeddings and prediction heads.  Training-gradient conflict
statistics are collected from real optimizer steps in every epoch.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import train_pems_joint_gradient_conflict as core


DATASETS = (
    "bike_drop",
    "bike_pick",
    "taxi_drop",
    "taxi_pick",
    "pems03",
    "pems04",
    "pems08",
    "urbanev",
    "shenzhen",
    "sd",
)
MODEL_ARGUMENTS = (
    "--d_model", "512",
    "--num_heads", "8",
    "--num_layers", "6",
    "--ffn_dim", "2064",
    "--embedding_dim", "200",
    "--target_parameters", "19996800",
    "--parameter_tolerance", "1000",
    "--gradient_probe_every", "1",
    "--proportional_task_batches",
)


def has_option(name: str) -> bool:
    return any(
        argument == name or argument.startswith(name + "=")
        for argument in sys.argv[1:]
    )


def main() -> None:
    core.DATASETS = DATASETS
    core.__doc__ = __doc__
    sys.argv.extend(MODEL_ARGUMENTS)
    if not has_option("--save_dir"):
        output_dir = Path("logs") / time.strftime(
            "all10_20m_gradient_conflict_%Y-%m-%d-%H%M%S"
        )
        sys.argv.extend(["--save_dir", str(output_dir)])
    core.main()


if __name__ == "__main__":
    main()
