#!/usr/bin/env python3
"""Compare dense and true Top-10 adaptive-graph Transformer ablations.

The script is intentionally read-only with respect to training artifacts.  It
writes derived tables/figures only below the Top-10 experiment root.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATASETS = ("taxi_drop", "taxi_pick", "bike_drop", "bike_pick")
MODES = ("qk", "graph", "qk_graph")
METRICS = ("mae", "rmse", "mape", "wmape")
LOCAL_STLLM = {
    "taxi_drop": {"mae": 5.2032, "rmse": 9.1257, "wmape": 0.1986},
    "taxi_pick": {"mae": 5.3259, "rmse": 9.3011, "wmape": 0.2015},
    "bike_drop": {"mae": 1.9146, "rmse": 2.8702, "wmape": 0.3871},
    "bike_pick": {"mae": 2.0152, "rmse": 3.1371, "wmape": 0.4059},
}
LABELS = {
    "local_stllm": "STLLM+ local",
    "qk_dense": "QK (graph-free)",
    "graph_dense": "Graph dense",
    "qk_graph_dense": "QK+Graph dense",
    "graph_topk10": "Graph Top-10",
    "qk_graph_topk10": "QK+Graph Top-10",
}
COLORS = {
    "local_stllm": "#777777",
    "qk_dense": "#377eb8",
    "graph_dense": "#ff9f1c",
    "qk_graph_dense": "#4daf4a",
    "graph_topk10": "#e41a1c",
    "qk_graph_topk10": "#984ea3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--topk-root",
        type=Path,
        default=Path("logs/topk10_transformer_ablation_seed6666"),
    )
    parser.add_argument(
        "--dense-taxi-drop-root",
        type=Path,
        default=Path("logs/taxi_drop_transformer_ablation_seed6666"),
    )
    parser.add_argument(
        "--dense-remaining-root",
        type=Path,
        default=Path("logs/remaining_transformer_ablation_seed6666"),
    )
    parser.add_argument("--seed", type=int, default=6666)
    return parser.parse_args()


def run_dir(root: Path, dataset: str, mode: str, seed: int) -> Path:
    return root / dataset / mode / f"seed_{seed}"


def dense_root(args: argparse.Namespace, dataset: str) -> Path:
    if dataset == "taxi_drop":
        return args.dense_taxi_drop_root
    return args.dense_remaining_root


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def relative_percent(value: float, reference: float) -> float:
    return 100.0 * (value - reference) / reference


def load_method(
    key: str, path: Path, graph_kind: str, dataset: str, dense_peer: dict | None = None
) -> dict:
    summary = read_json(path / "summary.json")
    test = summary["test_average"]
    row = {
        "dataset": dataset,
        "method": key,
        "label": LABELS[key],
        "graph_kind": graph_kind,
        "best_epoch": summary["best_epoch"],
        "best_validation_mae": summary["best_validation_mae"],
        "parameters": summary["model_parameters"],
        "average_train_seconds": summary["average_train_seconds"],
        "average_validation_seconds": summary["average_validation_seconds"],
        **{metric: test[metric] for metric in METRICS},
        "graph_alphas": summary.get("graph_alphas", []),
    }
    baseline = LOCAL_STLLM[dataset]
    for metric in ("mae", "rmse", "wmape"):
        row[f"vs_local_stllm_{metric}_percent"] = relative_percent(
            row[metric], baseline[metric]
        )
        row[f"vs_dense_peer_{metric}_percent"] = (
            relative_percent(row[metric], dense_peer[metric]) if dense_peer else None
        )
    return row


def load_dataset(args: argparse.Namespace, dataset: str) -> tuple[list[dict], dict]:
    topk_paths = {
        mode: run_dir(args.topk_root, dataset, mode, args.seed)
        for mode in ("graph", "qk_graph")
    }
    missing = [
        str(path / "summary.json")
        for path in topk_paths.values()
        if not (path / "summary.json").is_file()
    ]
    if missing:
        raise FileNotFoundError("; ".join(missing))

    droot = dense_root(args, dataset)
    dense = {
        mode: load_method(
            f"{mode}_dense",
            run_dir(droot, dataset, mode, args.seed),
            "none" if mode == "qk" else "dense",
            dataset,
        )
        for mode in MODES
    }
    topk = {
        mode: load_method(
            f"{mode}_topk10",
            topk_paths[mode],
            "topk10",
            dataset,
            dense[mode],
        )
        for mode in ("graph", "qk_graph")
    }

    baseline = LOCAL_STLLM[dataset]
    local = {
        "dataset": dataset,
        "method": "local_stllm",
        "label": LABELS["local_stllm"],
        "graph_kind": "STLLM+",
        "best_epoch": None,
        "best_validation_mae": None,
        "parameters": None,
        "average_train_seconds": None,
        "average_validation_seconds": None,
        "mae": baseline["mae"],
        "rmse": baseline["rmse"],
        "mape": None,
        "wmape": baseline["wmape"],
        "graph_alphas": [],
    }
    for metric in ("mae", "rmse", "wmape"):
        local[f"vs_local_stllm_{metric}_percent"] = 0.0
        local[f"vs_dense_peer_{metric}_percent"] = None

    rows = [local, dense["qk"], dense["graph"], dense["qk_graph"], topk["graph"], topk["qk_graph"]]
    summary = {
        "dataset": dataset,
        "topk_graph_stats": {
            mode: read_json(topk_paths[mode] / "graph_stats.json")
            for mode in ("graph", "qk_graph")
        },
        "methods": {row["method"]: row for row in rows},
    }
    return rows, summary


def plot_convergence(args: argparse.Namespace, dataset: str, out_dir: Path) -> None:
    specs = (
        ("qk_dense", dense_root(args, dataset), "qk"),
        ("graph_dense", dense_root(args, dataset), "graph"),
        ("qk_graph_dense", dense_root(args, dataset), "qk_graph"),
        ("graph_topk10", args.topk_root, "graph"),
        ("qk_graph_topk10", args.topk_root, "qk_graph"),
    )
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    for key, root, mode in specs:
        records = read_csv(run_dir(root, dataset, mode, args.seed) / "train.csv")
        ax.plot(
            [int(row["epoch"]) for row in records],
            [float(row["valid_loss"]) for row in records],
            label=LABELS[key],
            color=COLORS[key],
            linewidth=1.35,
            alpha=0.9,
        )
    ax.set(title=f"{dataset}: validation MAE convergence", xlabel="Epoch", ylabel="Validation MAE")
    ax.grid(alpha=0.22)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"topk10_convergence.{suffix}", dpi=220)
    plt.close(fig)


def plot_metrics(dataset: str, rows: list[dict], out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    for metric, ax in zip(METRICS, axes.flat):
        shown = [row for row in rows if row[metric] is not None]
        labels = [row["label"] for row in shown]
        values = [row[metric] for row in shown]
        ax.bar(range(len(shown)), values, color=[COLORS[row["method"]] for row in shown])
        ax.set_title(metric.upper())
        ax.set_xticks(range(len(shown)), labels, rotation=24, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(f"{dataset}: test metrics (lower is better)")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"topk10_test_metrics.{suffix}", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.topk_root.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    completed: list[str] = []
    incomplete: dict[str, str] = {}

    for dataset in DATASETS:
        try:
            rows, summary = load_dataset(args, dataset)
        except FileNotFoundError as error:
            incomplete[dataset] = str(error)
            continue
        out_dir = args.topk_root / dataset
        write_csv(out_dir / "topk10_comparison.csv", rows)
        with (out_dir / "topk10_analysis_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        plot_convergence(args, dataset, out_dir)
        plot_metrics(dataset, rows, out_dir)
        all_rows.extend(rows)
        completed.append(dataset)

    if all_rows:
        write_csv(args.topk_root / "cross_dataset_comparison.csv", all_rows)
    audit = {"completed_datasets": completed, "incomplete_datasets": incomplete}
    with (args.topk_root / "analysis_status.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
