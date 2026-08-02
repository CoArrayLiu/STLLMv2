#!/usr/bin/env python3
"""Analyze joint QK attention distributions and adaptive-graph alignment."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr

import util
from dataset_config import get_dataset_config
from train_qk_joint import DATASETS, JointQKTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "logs/qk_joint_separate_heads_seed6666/best_model.pth"
        ),
    )
    parser.add_argument(
        "--training_config",
        type=Path,
        default=Path(
            "logs/qk_joint_separate_heads_seed6666/config.json"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("logs/qk_joint_attention_seed6666_n64"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--precision",
        choices=("fp32", "bf16", "fp16"),
        default="bf16",
    )
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=10)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")
    if not args.training_config.is_file():
        raise FileNotFoundError(
            f"Training config does not exist: {args.training_config}"
        )
    for name in ("num_samples", "batch_size", "top_k"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name} must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {args.output_dir}"
        )
    if args.precision != "fp32" and not torch.cuda.is_available():
        raise ValueError(f"--precision {args.precision} requires CUDA")


def build_model(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[JointQKTransformer, dict]:
    config = json.loads(args.training_config.read_text(encoding="utf-8"))
    model_args = SimpleNamespace(
        input_dim=int(config["input_dim"]),
        input_len=int(config["input_len"]),
        output_len=int(config["output_len"]),
        d_model=int(config["d_model"]),
        num_heads=int(config["num_heads"]),
        num_layers=int(config["num_layers"]),
        ffn_dim=int(config["ffn_dim"]),
        embedding_dim=int(config["embedding_dim"]),
        dropout=float(config["dropout"]),
    )
    model = JointQKTransformer(model_args)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()
    return model, config


def load_selected_test_data(
    dataset: str,
    num_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, util.StandardScaler]:
    config = get_dataset_config(dataset)
    train_path = config.dataset_path / "train.npz"
    test_path = config.dataset_path / "test.npz"
    with np.load(train_path) as archive:
        train_values = np.asarray(archive["x"][..., 0])
        mean = float(train_values.mean())
        std = float(train_values.std())
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError(
            f"Invalid scaler for {dataset}: mean={mean}, std={std}"
        )

    with np.load(test_path) as archive:
        test_size = int(archive["x"].shape[0])
        count = min(num_samples, test_size)
        indices = np.linspace(0, test_size - 1, count, dtype=np.int64)
        x = np.asarray(archive["x"][indices], dtype=np.float32)
        y = np.asarray(archive["y"][indices], dtype=np.float32)
    x[..., 0] = (x[..., 0] - mean) / std
    return x, y, indices, util.StandardScaler(mean, std)


def load_adaptive_graph(dataset: str) -> tuple[np.ndarray, Path]:
    config = get_dataset_config(dataset)
    try:
        path = config.graphs["adaptive"]
    except KeyError as error:
        raise ValueError(
            f"Dataset {dataset} has no adaptive graph configured"
        ) from error
    graph = np.asarray(util.load_graph_data(path), dtype=np.float64)
    expected = (config.num_nodes, config.num_nodes)
    if graph.shape != expected:
        raise ValueError(
            f"Adaptive graph shape mismatch for {dataset}: "
            f"expected {expected}, got {graph.shape}"
        )
    if not np.isfinite(graph).all() or np.any(graph < 0):
        raise ValueError(f"Invalid adaptive graph values: {path}")
    row_sum = graph.sum(axis=1, keepdims=True)
    if np.any(row_sum <= 0):
        raise ValueError(f"Adaptive graph contains zero-sum rows: {path}")
    return graph / row_sum, path


def topk_indices(
    matrix: np.ndarray,
    top_k: int,
    exclude_diagonal: bool = True,
) -> np.ndarray:
    work = np.asarray(matrix, dtype=np.float64).copy()
    if exclude_diagonal:
        np.fill_diagonal(work, -np.inf)
    return np.argpartition(work, -top_k, axis=1)[:, -top_k:]


def row_js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    epsilon = np.finfo(np.float64).tiny
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left / left.sum(axis=1, keepdims=True)
    right = right / right.sum(axis=1, keepdims=True)
    middle = 0.5 * (left + right)
    divergence = 0.5 * (
        np.sum(left * np.log((left + epsilon) / (middle + epsilon)), axis=1)
        + np.sum(
            right * np.log((right + epsilon) / (middle + epsilon)),
            axis=1,
        )
    )
    return float(divergence.mean())


def histogram_js(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left / left.sum()
    right = right / right.sum()
    middle = 0.5 * (left + right)
    mask_left = left > 0
    mask_right = right > 0
    value = 0.5 * np.sum(
        left[mask_left] * np.log(left[mask_left] / middle[mask_left])
    )
    value += 0.5 * np.sum(
        right[mask_right] * np.log(right[mask_right] / middle[mask_right])
    )
    return float(value)


def safe_correlation(
    left: np.ndarray,
    right: np.ndarray,
    method: str,
) -> float:
    if method == "pearson":
        value = pearsonr(left, right)[0]
    elif method == "spearman":
        value = spearmanr(left, right)[0]
    else:
        raise ValueError(method)
    return float(value)


def topk_overlap(
    left_indices: np.ndarray,
    right_indices: np.ndarray,
) -> float:
    overlaps = [
        len(set(left_row).intersection(right_row)) / left_indices.shape[1]
        for left_row, right_row in zip(left_indices, right_indices)
    ]
    return float(np.mean(overlaps))


class AttentionCollector:
    def __init__(
        self,
        model: JointQKTransformer,
        num_nodes: int,
        top_k: int,
        example_positions: set[int],
    ) -> None:
        self.model = model
        self.num_nodes = num_nodes
        self.top_k = top_k
        self.example_positions = example_positions
        self.num_heads = model.backbone.layers[0].mixer.num_heads
        self.num_layers = len(model.backbone.layers)
        self.layer_sums = [
            np.zeros((num_nodes, num_nodes), dtype=np.float64)
            for _ in range(self.num_layers)
        ]
        self.layer_count = np.zeros(self.num_layers, dtype=np.int64)
        self.layer_rows: list[list[dict]] = [
            [] for _ in range(self.num_layers)
        ]
        self.hist_edges = np.linspace(-8.0, 0.0, 161)
        self.histograms = [
            np.zeros(len(self.hist_edges) - 1, dtype=np.float64)
            for _ in range(self.num_layers)
        ]
        self.example_sums = {
            position: np.zeros((num_nodes, num_nodes), dtype=np.float64)
            for position in example_positions
        }
        self.current_positions: list[int] = []
        self.current_indices: list[int] = []
        self.handles = [
            layer.mixer.register_forward_pre_hook(self._make_hook(layer_index))
            for layer_index, layer in enumerate(model.backbone.layers)
        ]

    def set_batch(
        self,
        positions: np.ndarray,
        sample_indices: np.ndarray,
    ) -> None:
        self.current_positions = [int(value) for value in positions]
        self.current_indices = [int(value) for value in sample_indices]

    def _make_hook(self, layer_index: int):
        def hook(module, inputs) -> None:
            x = inputs[0].detach().float()
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
                query = F.linear(
                    x,
                    module.q_proj.weight.float(),
                    (
                        module.q_proj.bias.float()
                        if module.q_proj.bias is not None
                        else None
                    ),
                )
                key = F.linear(
                    x,
                    module.k_proj.weight.float(),
                    (
                        module.k_proj.bias.float()
                        if module.k_proj.bias is not None
                        else None
                    ),
                )
                batch_size, num_nodes, _ = query.shape
                query = query.view(
                    batch_size,
                    num_nodes,
                    module.num_heads,
                    module.head_dim,
                ).transpose(1, 2)
                key = key.view(
                    batch_size,
                    num_nodes,
                    module.num_heads,
                    module.head_dim,
                ).transpose(1, 2)
                scores = torch.matmul(query, key.transpose(-2, -1))
                scores = scores / math.sqrt(module.head_dim)
                attention = torch.softmax(scores, dim=-1)

            self.layer_sums[layer_index] += (
                attention.sum(dim=(0, 1)).cpu().double().numpy()
            )
            self.layer_count[layer_index] += (
                attention.shape[0] * attention.shape[1]
            )

            entropy = -(
                attention
                * torch.log(attention.clamp_min(torch.finfo(torch.float32).tiny))
            ).sum(dim=-1)
            sample_entropy = entropy.mean(dim=(1, 2))
            normalized_entropy = sample_entropy / math.log(self.num_nodes)
            effective_neighbors = torch.exp(sample_entropy)
            top_mass = torch.topk(
                attention,
                k=self.top_k,
                dim=-1,
            ).values.sum(dim=-1).mean(dim=(1, 2))
            self_mass = torch.diagonal(
                attention,
                dim1=-2,
                dim2=-1,
            ).mean(dim=(1, 2))
            maximum = attention.max(dim=-1).values.mean(dim=(1, 2))

            log_attention = torch.log10(attention.clamp_min(1e-8))
            histogram = torch.histc(
                log_attention,
                bins=len(self.hist_edges) - 1,
                min=-8.0,
                max=0.0,
            )
            self.histograms[layer_index] += histogram.cpu().double().numpy()

            for local_index, (position, sample_index) in enumerate(
                zip(self.current_positions, self.current_indices)
            ):
                self.layer_rows[layer_index].append(
                    {
                        "sample_position": position,
                        "sample_index": sample_index,
                        "layer": layer_index + 1,
                        "normalized_entropy": float(
                            normalized_entropy[local_index].item()
                        ),
                        "effective_neighbors": float(
                            effective_neighbors[local_index].item()
                        ),
                        "top10_mass": float(top_mass[local_index].item()),
                        "self_mass": float(self_mass[local_index].item()),
                        "mean_row_max": float(maximum[local_index].item()),
                    }
                )
                if position in self.example_sums:
                    self.example_sums[position] += (
                        attention[local_index]
                        .mean(dim=0)
                        .cpu()
                        .double()
                        .numpy()
                    )

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def layer_attention(self) -> np.ndarray:
        return np.stack(
            [
                matrix / count
                for matrix, count in zip(self.layer_sums, self.layer_count)
            ]
        )

    def example_attention(self) -> tuple[np.ndarray, np.ndarray]:
        positions = np.asarray(sorted(self.example_sums), dtype=np.int64)
        matrices = np.stack(
            [
                self.example_sums[int(position)] / self.num_layers
                for position in positions
            ]
        )
        return positions, matrices


def graph_alignment_metrics(
    attention: np.ndarray,
    graph: np.ndarray,
    top_k: int,
) -> dict[str, float]:
    num_nodes = attention.shape[0]
    off_diagonal = ~np.eye(num_nodes, dtype=bool)
    attention_flat = attention[off_diagonal]
    graph_flat = graph[off_diagonal]
    attention_top = topk_indices(attention, top_k)
    graph_top = topk_indices(graph, top_k)
    rows = np.arange(num_nodes)[:, None]
    mass_on_graph_top = attention[rows, graph_top].sum(axis=1).mean()
    graph_mass_on_attention_top = graph[rows, attention_top].sum(axis=1).mean()
    cosine = float(
        np.dot(attention_flat, graph_flat)
        / (
            np.linalg.norm(attention_flat)
            * np.linalg.norm(graph_flat)
            + np.finfo(np.float64).tiny
        )
    )
    return {
        "pearson_offdiag": safe_correlation(
            attention_flat,
            graph_flat,
            "pearson",
        ),
        "spearman_offdiag": safe_correlation(
            attention_flat,
            graph_flat,
            "spearman",
        ),
        "cosine_offdiag": cosine,
        "row_js_divergence": row_js_divergence(attention, graph),
        "top10_neighbor_overlap": topk_overlap(attention_top, graph_top),
        "attention_mass_on_graph_top10": float(mass_on_graph_top),
        "attention_mass_lift_vs_uniform": float(
            mass_on_graph_top / (top_k / num_nodes)
        ),
        "graph_mass_on_attention_top10": float(graph_mass_on_attention_top),
    }


def matrix_pair_metrics(
    left: np.ndarray,
    right: np.ndarray,
    top_k: int,
) -> dict[str, float]:
    num_nodes = left.shape[0]
    mask = ~np.eye(num_nodes, dtype=bool)
    left_flat = left[mask]
    right_flat = right[mask]
    return {
        "pearson_offdiag": safe_correlation(
            left_flat,
            right_flat,
            "pearson",
        ),
        "spearman_offdiag": safe_correlation(
            left_flat,
            right_flat,
            "spearman",
        ),
        "row_js_divergence": row_js_divergence(left, right),
        "top10_neighbor_overlap": topk_overlap(
            topk_indices(left, top_k),
            topk_indices(right, top_k),
        ),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def show_log_matrix(ax, matrix: np.ndarray, title: str) -> None:
    image = ax.imshow(
        np.log10(np.asarray(matrix) + 1e-8),
        cmap="magma",
        aspect="auto",
    )
    ax.set_title(title)
    ax.set_xlabel("Key node")
    ax.set_ylabel("Query node")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def plot_dataset_overview(
    output_dir: Path,
    dataset: str,
    graph: np.ndarray,
    layer_attention: np.ndarray,
    layer_rows: list[dict],
    graph_rows: list[dict],
) -> None:
    mean_attention = layer_attention.mean(axis=0)
    figure, axes = plt.subplots(2, 3, figsize=(18, 10))
    show_log_matrix(axes[0, 0], graph, "Adaptive graph: log10(weight)")
    show_log_matrix(
        axes[0, 1],
        layer_attention[0],
        "QK attention layer 1: log10(weight)",
    )
    show_log_matrix(
        axes[0, 2],
        layer_attention[-1],
        "QK attention layer 6: log10(weight)",
    )
    show_log_matrix(
        axes[1, 0],
        mean_attention,
        "Mean QK attention: log10(weight)",
    )

    mask = ~np.eye(graph.shape[0], dtype=bool)
    axes[1, 1].hexbin(
        np.log10(graph[mask] + 1e-8),
        np.log10(mean_attention[mask] + 1e-8),
        gridsize=55,
        bins="log",
        mincnt=1,
        cmap="viridis",
    )
    axes[1, 1].set(
        title="Off-diagonal graph vs attention",
        xlabel="log10 adaptive-graph weight",
        ylabel="log10 mean-attention weight",
    )

    layers = [row["layer"] for row in graph_rows]
    overlap = [row["top10_neighbor_overlap"] for row in graph_rows]
    lift = [row["attention_mass_lift_vs_uniform"] for row in graph_rows]
    entropy_by_layer = []
    for layer in layers:
        values = [
            row["normalized_entropy"]
            for row in layer_rows
            if row["layer"] == layer
        ]
        entropy_by_layer.append(float(np.mean(values)))
    axes[1, 2].plot(layers, overlap, marker="o", label="Top-10 overlap")
    axes[1, 2].plot(layers, entropy_by_layer, marker="s", label="Entropy")
    axes[1, 2].plot(layers, lift, marker="^", label="Graph mass lift")
    axes[1, 2].set(
        title="Layerwise attention statistics",
        xlabel="Transformer layer",
    )
    axes[1, 2].grid(alpha=0.25)
    axes[1, 2].legend()

    figure.suptitle(dataset)
    figure.tight_layout()
    figure.savefig(
        output_dir / f"{dataset}_attention_graph_overview.png",
        dpi=180,
    )
    plt.close(figure)


def plot_example_attention(
    output_dir: Path,
    dataset: str,
    graph: np.ndarray,
    example_indices: np.ndarray,
    example_attention: np.ndarray,
) -> None:
    columns = 1 + len(example_indices)
    figure, axes = plt.subplots(1, columns, figsize=(5 * columns, 4.8))
    show_log_matrix(axes[0], graph, "Adaptive graph")
    for column, (sample_index, matrix) in enumerate(
        zip(example_indices, example_attention),
        start=1,
    ):
        show_log_matrix(
            axes[column],
            matrix,
            f"Test sample {int(sample_index)}",
        )
    figure.suptitle(f"{dataset}: graph and sample QK attention")
    figure.tight_layout()
    figure.savefig(
        output_dir / f"{dataset}_sample_attention.png",
        dpi=180,
    )
    plt.close(figure)


def plot_global_summary(
    output_dir: Path,
    dataset_rows: list[dict],
    pairwise_rows: list[dict],
) -> None:
    labels = [row["dataset"] for row in dataset_rows]
    x = np.arange(len(labels))
    figure, axes = plt.subplots(1, 3, figsize=(17, 5))
    axes[0].bar(
        x,
        [row["normalized_entropy"] for row in dataset_rows],
        color="#377eb8",
    )
    axes[0].set_title("Normalized attention entropy")
    axes[1].bar(
        x,
        [row["top10_mass"] for row in dataset_rows],
        color="#ff9f1c",
    )
    axes[1].set_title("Attention Top-10 mass")
    axes[2].bar(
        x,
        [row["graph_top10_mass_lift"] for row in dataset_rows],
        color="#4daf4a",
    )
    axes[2].set_title("Mass on graph Top-10 / uniform")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=22, ha="right")
        ax.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "attention_dataset_summary.png", dpi=180)
    plt.close(figure)

    size = len(DATASETS)
    matrix = np.zeros((size, size), dtype=np.float64)
    for row in pairwise_rows:
        left = DATASETS.index(row["left_dataset"])
        right = DATASETS.index(row["right_dataset"])
        matrix[left, right] = row["histogram_js_divergence"]
        matrix[right, left] = row["histogram_js_divergence"]
    figure, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(size), DATASETS, rotation=25, ha="right")
    ax.set_yticks(range(size), DATASETS)
    ax.set_title("Pairwise attention-weight histogram JS divergence")
    for row_index in range(size):
        for column in range(size):
            ax.text(
                column,
                row_index,
                f"{matrix[row_index, column]:.4f}",
                ha="center",
                va="center",
            )
    plt.colorbar(image, ax=ax)
    figure.tight_layout()
    figure.savefig(output_dir / "attention_pairwise_js.png", dpi=180)
    plt.close(figure)


def analyze_dataset(
    args: argparse.Namespace,
    model: JointQKTransformer,
    dataset: str,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> dict:
    x, y, sample_indices, scaler = load_selected_test_data(
        dataset,
        args.num_samples,
    )
    graph, graph_path = load_adaptive_graph(dataset)
    num_samples = len(sample_indices)
    example_positions = {
        0,
        num_samples // 2,
        num_samples - 1,
    }
    collector = AttentionCollector(
        model=model,
        num_nodes=x.shape[2],
        top_k=args.top_k,
        example_positions=example_positions,
    )
    sample_prediction_rows = []
    try:
        for start in range(0, num_samples, args.batch_size):
            end = min(num_samples, start + args.batch_size)
            positions = np.arange(start, end, dtype=np.int64)
            indices = sample_indices[start:end]
            collector.set_batch(positions, indices)
            model_input = torch.as_tensor(
                x[start:end],
                dtype=torch.float32,
                device=device,
            ).transpose(1, 3)
            target = torch.as_tensor(
                y[start:end],
                dtype=torch.float32,
                device=device,
            )
            with torch.no_grad(), torch.cuda.amp.autocast(
                enabled=amp_enabled,
                dtype=amp_dtype,
            ):
                output = model(dataset, model_input)
            prediction = scaler.inverse_transform(output.float())
            prediction = prediction.squeeze(-1)
            real = target.squeeze(-1)
            for local_index, sample_index in enumerate(indices):
                mask = real[local_index] > 0
                sample_mae = torch.abs(
                    prediction[local_index][mask]
                    - real[local_index][mask]
                ).mean()
                sample_prediction_rows.append(
                    {
                        "dataset": dataset,
                        "sample_index": int(sample_index),
                        "sample_mae": float(sample_mae.item()),
                    }
                )
    finally:
        collector.close()

    layer_attention = collector.layer_attention()
    overall_attention = layer_attention.mean(axis=0)
    example_positions_array, example_attention = (
        collector.example_attention()
    )
    example_indices = sample_indices[example_positions_array]
    layer_rows = [
        {"dataset": dataset, **row}
        for rows in collector.layer_rows
        for row in rows
    ]

    graph_rows = []
    for layer_index, attention in enumerate(layer_attention):
        metrics = graph_alignment_metrics(
            attention,
            graph,
            args.top_k,
        )
        graph_rows.append(
            {
                "dataset": dataset,
                "layer": layer_index + 1,
                **metrics,
            }
        )
    overall_graph = graph_alignment_metrics(
        overall_attention,
        graph,
        args.top_k,
    )

    graph_entropy = -np.sum(
        graph * np.log(graph + np.finfo(np.float64).tiny),
        axis=1,
    )
    graph_top = topk_indices(graph, args.top_k, exclude_diagonal=False)
    rows = np.arange(graph.shape[0])[:, None]
    graph_top_mass = float(graph[rows, graph_top].sum(axis=1).mean())

    dataset_row = {
        "dataset": dataset,
        "num_samples": num_samples,
        "num_nodes": x.shape[2],
        "sample_mae": float(
            np.mean([row["sample_mae"] for row in sample_prediction_rows])
        ),
        "normalized_entropy": float(
            np.mean([row["normalized_entropy"] for row in layer_rows])
        ),
        "effective_neighbors": float(
            np.mean([row["effective_neighbors"] for row in layer_rows])
        ),
        "top10_mass": float(
            np.mean([row["top10_mass"] for row in layer_rows])
        ),
        "self_mass": float(
            np.mean([row["self_mass"] for row in layer_rows])
        ),
        "mean_row_max": float(
            np.mean([row["mean_row_max"] for row in layer_rows])
        ),
        "graph_normalized_entropy": float(
            graph_entropy.mean() / math.log(graph.shape[0])
        ),
        "graph_top10_mass": graph_top_mass,
        "graph_pearson": overall_graph["pearson_offdiag"],
        "graph_spearman": overall_graph["spearman_offdiag"],
        "graph_cosine": overall_graph["cosine_offdiag"],
        "graph_row_js": overall_graph["row_js_divergence"],
        "graph_top10_overlap": overall_graph["top10_neighbor_overlap"],
        "attention_mass_on_graph_top10": overall_graph[
            "attention_mass_on_graph_top10"
        ],
        "graph_top10_mass_lift": overall_graph[
            "attention_mass_lift_vs_uniform"
        ],
        "adaptive_graph_path": str(graph_path),
    }

    histogram = np.sum(np.stack(collector.histograms), axis=0)
    np.savez_compressed(
        args.output_dir / f"{dataset}_attention_matrices.npz",
        layer_attention=layer_attention.astype(np.float32),
        overall_attention=overall_attention.astype(np.float32),
        graph=graph.astype(np.float32),
        sample_indices=sample_indices,
        example_indices=example_indices,
        example_attention=example_attention.astype(np.float32),
        histogram_edges=collector.hist_edges,
        histogram_counts=histogram,
    )
    plot_dataset_overview(
        args.output_dir,
        dataset,
        graph,
        layer_attention,
        layer_rows,
        graph_rows,
    )
    plot_example_attention(
        args.output_dir,
        dataset,
        graph,
        example_indices,
        example_attention,
    )
    return {
        "dataset_row": dataset_row,
        "layer_rows": layer_rows,
        "graph_rows": graph_rows,
        "prediction_rows": sample_prediction_rows,
        "layer_attention": layer_attention,
        "overall_attention": overall_attention,
        "graph": graph,
        "histogram": histogram,
    }


def write_report(
    path: Path,
    dataset_rows: list[dict],
    pairwise_rows: list[dict],
    same_topology_rows: list[dict],
    args: argparse.Namespace,
) -> None:
    lines = [
        "# 联合 QK 注意力与自适应图分析",
        "",
        f"从每个测试集等间隔抽取 {args.num_samples} 个样本，"
        "提取 6 层、12 个头的 QK softmax 注意力。",
        "",
        "## 数据集汇总",
        "",
        "| 数据集 | 归一化熵 | 有效邻居数 | Top-10质量 | 自注意质量 | "
        "图Spearman | 图Top-10重合 | 图Top-10注意力提升 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in dataset_rows:
        lines.append(
            f"| {row['dataset']} | {row['normalized_entropy']:.4f} | "
            f"{row['effective_neighbors']:.1f} | {row['top10_mass']:.4f} | "
            f"{row['self_mass']:.4f} | {row['graph_spearman']:.4f} | "
            f"{row['graph_top10_overlap']:.4f} | "
            f"{row['graph_top10_mass_lift']:.3f}× |"
        )

    lines.extend(
        [
            "",
            "归一化熵越接近 1 表示越接近均匀注意力；图 Top-10 "
            "注意力提升以均匀分配的 10/N 为基准。",
            "",
            "## 注意力权重分布的两两差异",
            "",
            "| 数据集 A | 数据集 B | 直方图 JS 散度 |",
            "|---|---|---:|",
        ]
    )
    for row in pairwise_rows:
        lines.append(
            f"| {row['left_dataset']} | {row['right_dataset']} | "
            f"{row['histogram_js_divergence']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## 相同节点系统的平均注意力比较",
            "",
            "| 节点系统 | 任务 A | 任务 B | 注意力 Pearson | "
            "注意力 Top-10 重合 | 图 Pearson | 图 Top-10 重合 |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in same_topology_rows:
        lines.append(
            f"| {row['topology']} | {row['left_dataset']} | "
            f"{row['right_dataset']} | "
            f"{row['attention_pearson_offdiag']:.4f} | "
            f"{row['attention_top10_neighbor_overlap']:.4f} | "
            f"{row['graph_pearson_offdiag']:.4f} | "
            f"{row['graph_top10_neighbor_overlap']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model, training_config = build_model(args, device)
    amp_enabled = args.precision != "fp32"
    amp_dtype = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[args.precision]
    summary_path = args.checkpoint.parent / "summary.json"
    analysis_config = {
        **{
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "checkpoint_best_epoch": json.loads(
            summary_path.read_text(encoding="utf-8")
        )["best_epoch"],
        "model_layers": int(training_config["num_layers"]),
        "model_heads": int(training_config["num_heads"]),
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(analysis_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    results = {}
    for dataset in DATASETS:
        print(f"Analyzing {dataset}...", flush=True)
        results[dataset] = analyze_dataset(
            args,
            model,
            dataset,
            device,
            amp_enabled,
            amp_dtype,
        )
        row = results[dataset]["dataset_row"]
        print(
            f"{dataset}: entropy={row['normalized_entropy']:.4f}, "
            f"top10={row['top10_mass']:.4f}, "
            f"graph_spearman={row['graph_spearman']:.4f}, "
            f"graph_top10_overlap={row['graph_top10_overlap']:.4f}",
            flush=True,
        )

    dataset_rows = [results[name]["dataset_row"] for name in DATASETS]
    layer_rows = [
        row for name in DATASETS for row in results[name]["layer_rows"]
    ]
    graph_rows = [
        row for name in DATASETS for row in results[name]["graph_rows"]
    ]
    prediction_rows = [
        row for name in DATASETS for row in results[name]["prediction_rows"]
    ]

    pairwise_rows = []
    for left_index, left in enumerate(DATASETS):
        for right in DATASETS[left_index + 1 :]:
            pairwise_rows.append(
                {
                    "left_dataset": left,
                    "right_dataset": right,
                    "histogram_js_divergence": histogram_js(
                        results[left]["histogram"],
                        results[right]["histogram"],
                    ),
                }
            )

    same_topology_rows = []
    for topology, left, right in (
        ("taxi", "taxi_drop", "taxi_pick"),
        ("bike", "bike_drop", "bike_pick"),
    ):
        attention_metrics = matrix_pair_metrics(
            results[left]["overall_attention"],
            results[right]["overall_attention"],
            args.top_k,
        )
        graph_metrics = matrix_pair_metrics(
            results[left]["graph"],
            results[right]["graph"],
            args.top_k,
        )
        same_topology_rows.append(
            {
                "topology": topology,
                "left_dataset": left,
                "right_dataset": right,
                **{
                    f"attention_{key}": value
                    for key, value in attention_metrics.items()
                },
                **{
                    f"graph_{key}": value
                    for key, value in graph_metrics.items()
                },
            }
        )

    write_csv(args.output_dir / "dataset_summary.csv", dataset_rows)
    write_csv(args.output_dir / "sample_attention_metrics.csv", layer_rows)
    write_csv(args.output_dir / "sample_prediction_metrics.csv", prediction_rows)
    write_csv(args.output_dir / "graph_alignment_by_layer.csv", graph_rows)
    write_csv(args.output_dir / "attention_pairwise_js.csv", pairwise_rows)
    write_csv(
        args.output_dir / "same_topology_alignment.csv",
        same_topology_rows,
    )
    plot_global_summary(args.output_dir, dataset_rows, pairwise_rows)
    write_report(
        args.output_dir / "analysis.md",
        dataset_rows,
        pairwise_rows,
        same_topology_rows,
        args,
    )
    print(f"Analysis written to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
