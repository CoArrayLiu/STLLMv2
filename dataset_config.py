"""Dataset metadata shared by training and preprocessing entry points."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


GRAPH_TYPES = ("adaptive", "physical", "semantic")


@dataclass(frozen=True)
class DatasetConfig:
    """Static properties required to construct an experiment."""

    num_nodes: int
    dataset_path: Path
    time_slots: int
    graphs: Mapping[str, Path] = field(default_factory=dict)
    loss_space: str = "original"
    recommended_training: Mapping[str, object] = field(default_factory=dict)


DATASET_CONFIG = {
    "bike_drop": DatasetConfig(
        num_nodes=250,
        dataset_path=Path("data/bike_drop"),
        time_slots=48,
        graphs={"adaptive": Path("adp/bd/adaptive_adj_mx.pkl")},
    ),
    "bike_pick": DatasetConfig(
        num_nodes=250,
        dataset_path=Path("data/bike_pick"),
        time_slots=48,
        graphs={"adaptive": Path("adp/bp/adaptive_adj_mx.pkl")},
    ),
    "taxi_drop": DatasetConfig(
        num_nodes=266,
        dataset_path=Path("data/taxi_drop"),
        time_slots=48,
        graphs={"adaptive": Path("adp/td/adaptive_adj_mx.pkl")},
    ),
    "taxi_pick": DatasetConfig(
        num_nodes=266,
        dataset_path=Path("data/taxi_pick"),
        time_slots=48,
        graphs={"adaptive": Path("adp/tp/adaptive_adj_mx.pkl")},
    ),
    "pems08": DatasetConfig(
        num_nodes=170,
        dataset_path=Path("data/pems08"),
        time_slots=288,
        graphs={
            "physical": Path("data/st_data/pems08/pems08_adj.npy"),
            "semantic": Path("data/st_data/pems08/cached_dist_matrix.npy"),
        },
        loss_space="standardized",
        recommended_training={
            "precision": "bf16",
            "batch_size": 512,
            "eval_batch_size": 256,
            "lrate": 1e-3,
            "grad_clip": 10.0,
            "epochs": 60,
            "min_epochs": 15,
            "es_patience": 12,
            "lr_scheduler": "plateau",
            "lr_patience": 3,
            "lr_factor": 0.5,
            "min_lrate": 1e-5,
        },
    ),
}


def get_dataset_config(name: str) -> DatasetConfig:
    try:
        return DATASET_CONFIG[name]
    except KeyError as error:
        available = ", ".join(sorted(DATASET_CONFIG))
        raise ValueError(
            f"Unknown dataset {name!r}; available datasets: {available}"
        ) from error
