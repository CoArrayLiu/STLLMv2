#!/usr/bin/env python3
"""Jointly train an 8M QK Transformer and measure training-time task conflicts.

The backbone, temporal encoder, and final normalization are shared.  Each
dataset owns its node embedding and prediction head.  Gradient conflict is
measured on shared parameters from the exact task losses and full batches
used by sampled optimizer steps.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import util
from dataset_config import get_dataset_config
from model_ST_Transformer_adaptive import STTransformerAdaptive
from ranger21 import Ranger
from train_qk_joint import aggregate_metrics, evaluate_test_set, prepare_batch


DATASETS = ("pems03", "pems04")
METRICS = ("mae", "mape", "rmse", "wmape")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument(
        "--proportional_task_batches",
        action="store_true",
        help="Scale task batches by training-set size so each epoch covers each task about once",
    )
    parser.add_argument(
        "--gradient_batch_size",
        type=int,
        default=64,
        help="Deprecated compatibility option; exact probes use the full training batch",
    )
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--disable_tf32", action="store_true")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--min_epochs", type=int, default=15)
    parser.add_argument("--es_patience", type=int, default=12)
    parser.add_argument("--lrate", type=float, default=1e-3)
    parser.add_argument("--wdecay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=10.0)
    parser.add_argument("--lr_patience", type=int, default=3)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--min_lrate", type=float, default=1e-5)

    parser.add_argument("--input_dim", type=int, default=3)
    parser.add_argument("--input_len", type=int, default=12)
    parser.add_argument("--output_len", type=int, default=12)
    parser.add_argument("--d_model", type=int, default=320)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--ffn_dim", type=int, default=1380)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--target_parameters", type=int, default=8_000_000)
    parser.add_argument("--parameter_tolerance", type=int, default=1_000)

    parser.add_argument("--gradient_probe_every", type=int, default=5)
    parser.add_argument("--gradient_probe_batches", type=int, default=2)
    parser.add_argument(
        "--max_steps_per_epoch",
        type=int,
        default=0,
        help="0 uses the full longer dataset; positive values are for smoke tests",
    )
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--save_dir", type=Path, default=None)
    parser.add_argument("--print_model", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        "batch_size", "eval_batch_size", "gradient_batch_size", "epochs",
        "es_patience", "input_dim", "input_len", "output_len", "d_model",
        "num_heads", "num_layers", "ffn_dim", "embedding_dim",
        "target_parameters", "parameter_tolerance", "gradient_probe_every",
        "gradient_probe_batches",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name} must be positive")
    if args.d_model % args.num_heads:
        raise ValueError("--d_model must be divisible by --num_heads")
    if args.min_epochs < 0 or args.max_steps_per_epoch < 0:
        raise ValueError("--min_epochs and --max_steps_per_epoch cannot be negative")
    if not 0 <= args.dropout < 1:
        raise ValueError("--dropout must be in [0, 1)")
    if args.precision != "fp32" and not torch.cuda.is_available():
        raise ValueError(f"--precision {args.precision} requires CUDA")
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("The selected CUDA device does not support BF16")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class JointPEMSQKTransformer(nn.Module):
    """Shared QK backbone with dataset-specific node embeddings and heads."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        configs = {name: get_dataset_config(name) for name in DATASETS}
        time_slots = max(config.time_slots for config in configs.values())
        self.backbone = STTransformerAdaptive(
            adaptive_graph=None,
            time_slots=time_slots,
            input_dim=args.input_dim,
            num_nodes=max(config.num_nodes for config in configs.values()),
            input_len=args.input_len,
            output_len=args.output_len,
            attention_mode="qk",
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            ffn_dim=args.ffn_dim,
            embedding_dim=args.embedding_dim,
            dropout=args.dropout,
            learn_node_embedding=False,
            learn_prediction_head=False,
        )
        self.node_embeddings = nn.ParameterDict(
            {
                name: nn.Parameter(torch.empty(config.num_nodes, args.embedding_dim))
                for name, config in configs.items()
            }
        )
        for embedding in self.node_embeddings.values():
            nn.init.xavier_uniform_(embedding)
        self.prediction_heads = nn.ModuleDict(
            {
                name: nn.Conv2d(args.d_model, args.output_len, kernel_size=(1, 1))
                for name in DATASETS
            }
        )

    def forward(self, dataset: str, history_data: torch.Tensor) -> torch.Tensor:
        if dataset not in DATASETS:
            raise ValueError(f"Unknown PEMS task: {dataset}")
        return self.backbone(
            history_data,
            node_embedding_override=self.node_embeddings[dataset],
            prediction_head_override=self.prediction_heads[dataset],
        )

    def param_num(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def shared_named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        return [
            (name, parameter)
            for name, parameter in self.named_parameters()
            if name.startswith("backbone.") and parameter.requires_grad
        ]


def gradient_group(parameter_name: str) -> str:
    """Map a shared parameter to a fine-grained architectural component."""
    if parameter_name.startswith("backbone.start_conv"):
        return "input.value_projection"
    if parameter_name.startswith("backbone.temporal_embedding"):
        return "input.temporal_embedding"
    if parameter_name.startswith("backbone.input_projection"):
        return "input.fusion_projection"
    if parameter_name.startswith("backbone.final_norm"):
        return "output.final_norm"
    parts = parameter_name.split(".")
    if len(parts) >= 4 and parts[1] == "layers":
        layer = f"layer_{int(parts[2]):02d}"
        tail = ".".join(parts[3:])
        if tail.startswith(("norm1", "norm2")):
            component = "normalization"
        elif tail.startswith("mixer.q_proj"):
            component = "attention.query"
        elif tail.startswith("mixer.k_proj"):
            component = "attention.key"
        elif tail.startswith("mixer.v_proj"):
            component = "attention.value"
        elif tail.startswith("mixer.out_proj"):
            component = "attention.output"
        elif tail.startswith("ffn"):
            component = "ffn"
        else:
            component = "other"
        return f"{layer}.{component}"
    return "other_shared"


def conflict_rows_from_task_gradients(
    named_parameters: list[tuple[str, nn.Parameter]],
    task_gradients: dict[str, list[torch.Tensor | None]],
    task_losses: dict[str, float],
    *,
    epoch: int,
    probe_batch: int,
    epoch_step: int,
    global_step: int,
) -> list[dict[str, float | int | str]]:
    """Build module-level pairwise metrics from one real joint training step."""
    groups = [gradient_group(name) for name, _ in named_parameters]
    parameters = [parameter for _, parameter in named_parameters]
    by_group: dict[str, dict[str, list[torch.Tensor]]] = defaultdict(
        lambda: {dataset: [] for dataset in DATASETS}
    )
    for parameter_index, (group, parameter) in enumerate(zip(groups, parameters)):
        for dataset in DATASETS:
            gradient = task_gradients[dataset][parameter_index]
            flattened = (
                torch.zeros(
                    parameter.numel(),
                    dtype=torch.float32,
                    device=parameter.device,
                )
                if gradient is None
                else gradient.reshape(-1)
            )
            by_group[group][dataset].append(flattened)

    rows: list[dict[str, float | int | str]] = []
    for group, task_parts in sorted(by_group.items()):
        vectors = {dataset: torch.cat(task_parts[dataset]) for dataset in DATASETS}
        for task_a, task_b in combinations(DATASETS, 2):
            first, second = vectors[task_a], vectors[task_b]
            first_norm = torch.linalg.vector_norm(first)
            second_norm = torch.linalg.vector_norm(second)
            norm_sum = first_norm + second_norm
            denominator = first_norm * second_norm
            cosine = (
                float(torch.dot(first, second) / denominator)
                if denominator > 0
                else math.nan
            )
            combined_norm = torch.linalg.vector_norm(first + second)
            tug_of_war = (
                float(1.0 - combined_norm / norm_sum)
                if norm_sum > 0
                else math.nan
            )
            products = first * second
            product_mass = products.abs().sum()
            conflict_mass = (
                float((-products.clamp_max(0)).sum() / product_mass)
                if product_mass > 0
                else math.nan
            )
            active = (first != 0) & (second != 0)
            sign_conflict = (
                float((products[active] < 0).float().mean())
                if active.any()
                else math.nan
            )
            rows.append(
                {
                    "source": "train",
                    "epoch": epoch,
                    "probe_batch": probe_batch,
                    "epoch_step": epoch_step,
                    "global_step": global_step,
                    "group": group,
                    "parameters": int(first.numel()),
                    "task_a": task_a,
                    "task_b": task_b,
                    "task_pair": f"{task_a}__{task_b}",
                    "task_a_loss_contribution": task_losses[task_a],
                    "task_b_loss_contribution": task_losses[task_b],
                    "task_a_grad_norm": float(first_norm),
                    "task_b_grad_norm": float(second_norm),
                    "cosine": cosine,
                    "negative_cosine": (
                        int(cosine < 0) if math.isfinite(cosine) else 0
                    ),
                    "sign_conflict_rate": sign_conflict,
                    "conflict_mass": conflict_mass,
                    "tug_of_war_strength": tug_of_war,
                }
            )
    return rows


class Trainer:
    def __init__(
        self,
        args: argparse.Namespace,
        scalers: dict[str, util.StandardScaler],
        device: torch.device,
    ) -> None:
        self.model = JointPEMSQKTransformer(args).to(device)
        self.optimizer = Ranger(
            self.model.parameters(), lr=args.lrate, weight_decay=args.wdecay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=args.min_lrate,
        )
        self.scalers = scalers
        self.grad_clip = args.grad_clip
        self.amp_enabled = args.precision != "fp32"
        self.amp_dtype = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[args.precision]
        self.loss_scaler = torch.cuda.amp.GradScaler(enabled=args.precision == "fp16")

    def metrics(
        self, dataset: str, model_input: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.model(dataset, model_input).transpose(1, 3)
        prediction = self.scalers[dataset].inverse_transform(output.float())
        real = target.unsqueeze(1)
        return (
            util.MAE_torch(prediction, real, 0.0),
            util.MAPE_torch(prediction, real, 0.0),
            util.RMSE_torch(prediction, real, 0.0),
            util.WMAPE_torch(prediction, real, 0.0),
        )

    def normalized_loss(
        self, dataset: str, model_input: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return self.metrics(dataset, model_input, target)[0] / float(
            self.scalers[dataset].std
        )

    def train_step(
        self,
        batches: dict[str, tuple[torch.Tensor, torch.Tensor]],
        conflict_context: tuple[int, int, int, int] | None = None,
    ) -> tuple[
        dict[str, tuple[float, ...]],
        float,
        float,
        list[dict[str, float | int | str]],
    ]:
        """Update once and optionally inspect the exact per-task training gradients."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        values = {}
        named_parameters = (
            self.model.shared_named_parameters()
            if conflict_context is not None
            else []
        )
        shared_parameters = [parameter for _, parameter in named_parameters]
        task_gradients: dict[str, list[torch.Tensor | None]] = {}
        task_losses: dict[str, float] = {}

        for dataset in DATASETS:
            model_input, target = batches[dataset]
            with torch.cuda.amp.autocast(
                enabled=self.amp_enabled, dtype=self.amp_dtype
            ):
                metrics = self.metrics(dataset, model_input, target)
                loss = metrics[0] / float(self.scalers[dataset].std) / len(DATASETS)
            if conflict_context is not None:
                gradients = torch.autograd.grad(
                    loss,
                    shared_parameters,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True,
                )
                task_gradients[dataset] = [
                    None if gradient is None else gradient.detach().float()
                    for gradient in gradients
                ]
                task_losses[dataset] = float(loss.detach().item())
            self.loss_scaler.scale(loss).backward()
            values[dataset] = (
                *(metric.detach().item() for metric in metrics),
                model_input.shape[0],
            )

        self.loss_scaler.unscale_(self.optimizer)
        maximum = self.grad_clip if self.grad_clip > 0 else float("inf")
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), maximum)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"Non-finite joint gradient: {grad_norm.item()}")
        norm_value = float(grad_norm.item())
        clipped = float(self.grad_clip > 0 and norm_value > self.grad_clip)
        self.loss_scaler.step(self.optimizer)
        self.loss_scaler.update()

        conflict_rows: list[dict[str, float | int | str]] = []
        if conflict_context is not None:
            epoch, probe_batch, epoch_step, global_step = conflict_context
            conflict_rows = conflict_rows_from_task_gradients(
                named_parameters,
                task_gradients,
                task_losses,
                epoch=epoch,
                probe_batch=probe_batch,
                epoch_step=epoch_step,
                global_step=global_step,
            )
        return values, norm_value, clipped, conflict_rows

    @torch.no_grad()
    def eval_batch(
        self, dataset: str, model_input: torch.Tensor, target: torch.Tensor
    ) -> tuple[float, ...]:
        self.model.eval()
        with torch.cuda.amp.autocast(enabled=self.amp_enabled, dtype=self.amp_dtype):
            metrics = self.metrics(dataset, model_input, target)
        return (*(metric.item() for metric in metrics), model_input.shape[0])


def _gradient_conflicts_two_task(
    trainer: Trainer,
    paired_batches: list[dict[str, tuple[torch.Tensor, torch.Tensor]]],
    epoch: int,
) -> list[dict[str, float | int | str]]:
    """Compare two task gradients without modifying model or optimizer state."""
    model = trainer.model
    model.eval()
    named_parameters = model.shared_named_parameters()
    parameters = [parameter for _, parameter in named_parameters]
    groups = [gradient_group(name) for name, _ in named_parameters]
    rows: list[dict[str, float | int | str]] = []

    for batch_index, batches in enumerate(paired_batches):
        task_gradients: dict[str, list[torch.Tensor | None]] = {}
        task_losses = {}
        for dataset in DATASETS:
            model_input, target = batches[dataset]
            with torch.cuda.amp.autocast(enabled=False):
                loss = trainer.normalized_loss(dataset, model_input, target)
            gradients = torch.autograd.grad(
                loss, parameters, retain_graph=False, create_graph=False,
                allow_unused=True
            )
            task_gradients[dataset] = [
                None if grad is None else grad.detach().float().cpu()
                for grad in gradients
            ]
            task_losses[dataset] = float(loss.detach().item())

        by_group: dict[str, dict[str, list[torch.Tensor]]] = defaultdict(
            lambda: {dataset: [] for dataset in DATASETS}
        )
        for group, parameter, grad03, grad04 in zip(
            groups,
            parameters,
            task_gradients[DATASETS[0]],
            task_gradients[DATASETS[1]],
        ):
            for dataset, gradient in ((DATASETS[0], grad03), (DATASETS[1], grad04)):
                if gradient is None:
                    gradient = torch.zeros(parameter.numel(), dtype=torch.float32)
                else:
                    gradient = gradient.reshape(-1)
                by_group[group][dataset].append(gradient)

        for group, task_parts in sorted(by_group.items()):
            first = torch.cat(task_parts[DATASETS[0]])
            second = torch.cat(task_parts[DATASETS[1]])
            first_norm = torch.linalg.vector_norm(first)
            second_norm = torch.linalg.vector_norm(second)
            denominator = first_norm * second_norm
            cosine = float(torch.dot(first, second) / denominator) if denominator > 0 else math.nan
            products = first * second
            product_mass = products.abs().sum()
            conflict_mass = (
                float((-products.clamp_max(0)).sum() / product_mass)
                if product_mass > 0 else math.nan
            )
            active = (first != 0) & (second != 0)
            sign_conflict = (
                float(((first[active] * second[active]) < 0).float().mean())
                if active.any() else math.nan
            )
            rows.append(
                {
                    "epoch": epoch,
                    "probe_batch": batch_index,
                    "group": group,
                    "parameters": int(first.numel()),
                    "pems03_loss": task_losses["pems03"],
                    "pems04_loss": task_losses["pems04"],
                    "pems03_grad_norm": float(first_norm),
                    "pems04_grad_norm": float(second_norm),
                    "cosine": cosine,
                    "negative_cosine": int(cosine < 0) if math.isfinite(cosine) else 0,
                    "sign_conflict_rate": sign_conflict,
                    "conflict_mass": conflict_mass,
                }
            )
        del task_gradients
    trainer.optimizer.zero_grad(set_to_none=True)
    return rows


def make_probe_batches(
    data: dict[str, dict], args: argparse.Namespace, device: torch.device
) -> list[dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    required = args.gradient_probe_batches * args.gradient_batch_size
    for dataset in DATASETS:
        available = int(data[dataset]["val_loader"].size)
        if required > available:
            raise ValueError(
                f"Gradient probe needs {required} distinct validation samples for "
                f"{dataset}, but only {available} are available; reduce "
                "--gradient_probe_batches or --gradient_batch_size"
            )
    result = []
    for batch_index in range(args.gradient_probe_batches):
        probe = {}
        start = batch_index * args.gradient_batch_size
        stop = start + args.gradient_batch_size
        for dataset in DATASETS:
            loader = data[dataset]["val_loader"]
            probe[dataset] = prepare_batch(
                loader.xs[start:stop],
                loader.ys[start:stop],
                device,
            )
        result.append(probe)
    return result


def _summarize_conflicts_two_task(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby("group", as_index=False)
        .agg(
            parameters=("parameters", "first"),
            probes=("cosine", "count"),
            mean_cosine=("cosine", "mean"),
            min_cosine=("cosine", "min"),
            negative_cosine_rate=("negative_cosine", "mean"),
            mean_sign_conflict_rate=("sign_conflict_rate", "mean"),
            mean_conflict_mass=("conflict_mass", "mean"),
            mean_pems03_grad_norm=("pems03_grad_norm", "mean"),
            mean_pems04_grad_norm=("pems04_grad_norm", "mean"),
        )
        .sort_values(["mean_cosine", "mean_conflict_mass"], ascending=[True, False])
    )


def gradient_conflicts(
    trainer: Trainer,
    paired_batches: list[dict[str, tuple[torch.Tensor, torch.Tensor]]],
    epoch: int,
) -> list[dict[str, float | int | str]]:
    """Measure pairwise conflicts between every task on shared parameters."""
    if len(DATASETS) < 2:
        raise ValueError("Gradient conflict analysis requires at least two tasks")
    model = trainer.model
    model.eval()
    named_parameters = model.shared_named_parameters()
    parameters = [parameter for _, parameter in named_parameters]
    groups = [gradient_group(name) for name, _ in named_parameters]
    rows: list[dict[str, float | int | str]] = []

    for batch_index, batches in enumerate(paired_batches):
        task_gradients: dict[str, list[torch.Tensor | None]] = {}
        task_losses: dict[str, float] = {}
        for dataset in DATASETS:
            model_input, target = batches[dataset]
            with torch.cuda.amp.autocast(enabled=False):
                loss = trainer.normalized_loss(dataset, model_input, target)
            gradients = torch.autograd.grad(
                loss,
                parameters,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
            task_gradients[dataset] = [
                None if gradient is None else gradient.detach().float().cpu()
                for gradient in gradients
            ]
            task_losses[dataset] = float(loss.detach().item())

        by_group: dict[str, dict[str, list[torch.Tensor]]] = defaultdict(
            lambda: {dataset: [] for dataset in DATASETS}
        )
        for parameter_index, (group, parameter) in enumerate(zip(groups, parameters)):
            for dataset in DATASETS:
                gradient = task_gradients[dataset][parameter_index]
                flattened = (
                    torch.zeros(parameter.numel(), dtype=torch.float32)
                    if gradient is None
                    else gradient.reshape(-1)
                )
                by_group[group][dataset].append(flattened)

        for group, task_parts in sorted(by_group.items()):
            vectors = {
                dataset: torch.cat(task_parts[dataset]) for dataset in DATASETS
            }
            for task_a, task_b in combinations(DATASETS, 2):
                first, second = vectors[task_a], vectors[task_b]
                first_norm = torch.linalg.vector_norm(first)
                second_norm = torch.linalg.vector_norm(second)
                denominator = first_norm * second_norm
                cosine = (
                    float(torch.dot(first, second) / denominator)
                    if denominator > 0
                    else math.nan
                )
                products = first * second
                product_mass = products.abs().sum()
                conflict_mass = (
                    float((-products.clamp_max(0)).sum() / product_mass)
                    if product_mass > 0
                    else math.nan
                )
                active = (first != 0) & (second != 0)
                sign_conflict = (
                    float((products[active] < 0).float().mean())
                    if active.any()
                    else math.nan
                )
                rows.append(
                    {
                        "epoch": epoch,
                        "probe_batch": batch_index,
                        "group": group,
                        "parameters": int(first.numel()),
                        "task_a": task_a,
                        "task_b": task_b,
                        "task_pair": f"{task_a}__{task_b}",
                        "task_a_loss": task_losses[task_a],
                        "task_b_loss": task_losses[task_b],
                        "task_a_grad_norm": float(first_norm),
                        "task_b_grad_norm": float(second_norm),
                        "cosine": cosine,
                        "negative_cosine": (
                            int(cosine < 0) if math.isfinite(cosine) else 0
                        ),
                        "sign_conflict_rate": sign_conflict,
                        "conflict_mass": conflict_mass,
                    }
                )
        del task_gradients
    trainer.optimizer.zero_grad(set_to_none=True)
    return rows


def summarize_conflicts(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize module conflict frequency across task pairs and probe batches."""
    frame = frame.copy()
    if "tug_of_war_strength" not in frame:
        frame["tug_of_war_strength"] = math.nan
    aggregate = (
        frame.groupby("group", as_index=False)
        .agg(
            parameters=("parameters", "first"),
            pair_comparisons=("cosine", "count"),
            mean_cosine=("cosine", "mean"),
            min_cosine=("cosine", "min"),
            pairwise_conflict_rate=("negative_cosine", "mean"),
            mean_sign_conflict_rate=("sign_conflict_rate", "mean"),
            mean_conflict_mass=("conflict_mass", "mean"),
            mean_tug_of_war_strength=("tug_of_war_strength", "mean"),
            max_tug_of_war_strength=("tug_of_war_strength", "max"),
        )
    )
    per_probe = (
        frame.groupby(["epoch", "probe_batch", "group"], as_index=False)
        .agg(any_pair_conflict=("negative_cosine", "max"))
        .groupby("group", as_index=False)
        .agg(
            probes=("any_pair_conflict", "size"),
            any_pair_conflict_rate=("any_pair_conflict", "mean"),
        )
    )
    return (
        aggregate.merge(per_probe, on="group", how="left")
        .sort_values(
            ["any_pair_conflict_rate", "pairwise_conflict_rate", "mean_cosine"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )


def summarize_pairwise_conflicts(frame: pd.DataFrame) -> pd.DataFrame:
    """Report conflict frequency separately for each module and task pair."""
    frame = frame.copy()
    if "tug_of_war_strength" not in frame:
        frame["tug_of_war_strength"] = math.nan
    return (
        frame.groupby(["group", "task_pair", "task_a", "task_b"], as_index=False)
        .agg(
            parameters=("parameters", "first"),
            probes=("cosine", "count"),
            conflict_frequency=("negative_cosine", "mean"),
            mean_cosine=("cosine", "mean"),
            min_cosine=("cosine", "min"),
            mean_sign_conflict_rate=("sign_conflict_rate", "mean"),
            mean_conflict_mass=("conflict_mass", "mean"),
            mean_tug_of_war_strength=("tug_of_war_strength", "mean"),
            max_tug_of_war_strength=("tug_of_war_strength", "max"),
            mean_task_a_grad_norm=("task_a_grad_norm", "mean"),
            mean_task_b_grad_norm=("task_b_grad_norm", "mean"),
        )
        .sort_values(
            ["conflict_frequency", "mean_cosine", "mean_conflict_mass"],
            ascending=[False, True, False],
        )
        .reset_index(drop=True)
    )


def training_task_gradient_norms(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one non-duplicated module-gradient norm row per task and probe."""
    metadata = [
        "source",
        "epoch",
        "probe_batch",
        "epoch_step",
        "global_step",
        "group",
        "parameters",
    ]
    left = frame[
        metadata
        + ["task_a", "task_a_loss_contribution", "task_a_grad_norm"]
    ].rename(
        columns={
            "task_a": "dataset",
            "task_a_loss_contribution": "loss_contribution",
            "task_a_grad_norm": "grad_norm",
        }
    )
    right = frame[
        metadata
        + ["task_b", "task_b_loss_contribution", "task_b_grad_norm"]
    ].rename(
        columns={
            "task_b": "dataset",
            "task_b_loss_contribution": "loss_contribution",
            "task_b_grad_norm": "grad_norm",
        }
    )
    return (
        pd.concat([left, right], ignore_index=True)
        .drop_duplicates(
            ["epoch", "probe_batch", "group", "dataset"],
            keep="first",
        )
        .sort_values(["epoch", "probe_batch", "group", "dataset"])
        .reset_index(drop=True)
    )


def summarize_task_gradient_norms(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["group", "dataset"], as_index=False)
        .agg(
            parameters=("parameters", "first"),
            probes=("grad_norm", "count"),
            mean_grad_norm=("grad_norm", "mean"),
            median_grad_norm=("grad_norm", "median"),
            max_grad_norm=("grad_norm", "max"),
            mean_loss_contribution=("loss_contribution", "mean"),
        )
        .sort_values(["group", "mean_grad_norm"], ascending=[True, False])
        .reset_index(drop=True)
    )


def summarize_conflicts_by_epoch(frame: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for epoch, epoch_frame in frame.groupby("epoch", sort=True):
        summary = summarize_conflicts(epoch_frame)
        summary.insert(0, "epoch", int(epoch))
        summaries.append(summary)
    return pd.concat(summaries, ignore_index=True)


def summarize_pairwise_conflicts_by_epoch(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(
            ["epoch", "group", "task_pair", "task_a", "task_b"],
            as_index=False,
        )
        .agg(
            parameters=("parameters", "first"),
            probes=("cosine", "count"),
            conflict_frequency=("negative_cosine", "mean"),
            mean_cosine=("cosine", "mean"),
            min_cosine=("cosine", "min"),
            mean_conflict_mass=("conflict_mass", "mean"),
            mean_tug_of_war_strength=("tug_of_war_strength", "mean"),
            max_tug_of_war_strength=("tug_of_war_strength", "max"),
        )
        .sort_values(
            ["epoch", "conflict_frequency", "mean_tug_of_war_strength"],
            ascending=[True, False, False],
        )
        .reset_index(drop=True)
    )


def write_training_conflict_outputs(
    rows: list[dict[str, float | int | str]],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Persist raw training probes and every useful aggregation."""
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "training_gradient_conflicts.csv", index=False)

    module_summary = summarize_conflicts(frame)
    module_summary.to_csv(
        output_dir / "gradient_conflict_summary.csv", index=False
    )
    pair_summary = summarize_pairwise_conflicts(frame)
    pair_summary.to_csv(
        output_dir / "gradient_conflict_pair_summary.csv", index=False
    )
    summarize_conflicts_by_epoch(frame).to_csv(
        output_dir / "gradient_conflict_epoch_summary.csv", index=False
    )
    summarize_pairwise_conflicts_by_epoch(frame).to_csv(
        output_dir / "gradient_conflict_pair_epoch_summary.csv", index=False
    )

    task_norms = training_task_gradient_norms(frame)
    task_norms.to_csv(
        output_dir / "training_gradient_task_norms.csv", index=False
    )
    summarize_task_gradient_norms(task_norms).to_csv(
        output_dir / "gradient_task_norm_summary.csv", index=False
    )
    return module_summary, pair_summary


def serializable_args(args: argparse.Namespace) -> dict:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        tf32 = not args.disable_tf32
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        if tf32:
            torch.set_float32_matmul_precision("high")

    dataset_tag = "_".join(dataset.replace("pems", "") for dataset in DATASETS)
    output_dir = args.save_dir or Path("logs") / time.strftime(
        f"pems{dataset_tag}_8m_gradient_conflict_%Y-%m-%d-%H%M%S"
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(serializable_args(args), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    data = {}
    for dataset in DATASETS:
        config = get_dataset_config(dataset)
        data[dataset] = util.load_dataset(
            str(config.dataset_path),
            args.batch_size,
            args.eval_batch_size,
            args.eval_batch_size,
            expected_num_nodes=config.num_nodes,
            expected_input_len=args.input_len,
            expected_output_len=args.output_len,
            expected_input_dim=args.input_dim,
        )
    task_batch_sizes = {dataset: args.batch_size for dataset in DATASETS}
    if args.proportional_task_batches:
        largest_samples = max(
            int(data[dataset]["x_train"].shape[0]) for dataset in DATASETS
        )
        target_steps = math.ceil(largest_samples / args.batch_size)
        for dataset in DATASETS:
            samples = int(data[dataset]["x_train"].shape[0])
            task_batch_size = max(1, math.ceil(samples / target_steps))
            task_batch_sizes[dataset] = task_batch_size
            data[dataset]["train_loader"] = util.DataLoader(
                data[dataset]["x_train"],
                data[dataset]["y_train"],
                task_batch_size,
            )
    scalers = {dataset: data[dataset]["scaler"] for dataset in DATASETS}
    trainer = Trainer(args, scalers, device)
    parameters = trainer.model.param_num()
    shared_parameter_count = sum(
        parameter.numel()
        for _, parameter in trainer.model.shared_named_parameters()
    )
    if abs(parameters - args.target_parameters) > args.parameter_tolerance:
        raise ValueError(
            f"Model has {parameters:,} parameters, target is {args.target_parameters:,} "
            f"± {args.parameter_tolerance:,}"
        )
    if args.print_model:
        print(trainer.model)
    print(f"Output directory: {output_dir}")
    print(f"Model parameters: {parameters:,}")
    print(f"Trainable parameters: {trainer.model.count_trainable_params():,}")
    print(
        f"Shared/task-specific parameters: {shared_parameter_count:,}/"
        f"{parameters - shared_parameter_count:,}"
    )
    print(
        f"Loss: equal-weight standardized MAE; {', '.join(DATASETS)} use separate heads and node embeddings"
    )

    batch_counts = {
        dataset: data[dataset]["train_loader"].num_batch for dataset in DATASETS
    }
    steps_per_epoch = max(batch_counts.values())
    if args.max_steps_per_epoch:
        steps_per_epoch = min(steps_per_epoch, args.max_steps_per_epoch)
    print(f"Train batches: {batch_counts}; balanced steps per epoch: {steps_per_epoch}")
    print(f"Task batch sizes: {task_batch_sizes}")
    print(
        "Training-gradient probes: epoch 1 and every "
        f"{args.gradient_probe_every} epochs; "
        f"{min(args.gradient_probe_batches, steps_per_epoch)} evenly spaced "
        "full-batch optimizer steps"
    )

    best_score = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history = []
    conflict_rows = []
    global_step = 0
    checkpoint = output_dir / "best_model.pth"

    for epoch in range(1, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for dataset in DATASETS:
            data[dataset]["train_loader"].shuffle()
        iterators = {
            dataset: data[dataset]["train_loader"].get_iterator()
            for dataset in DATASETS
        }
        train_values = {dataset: [] for dataset in DATASETS}
        grad_norms, clipped = [], []
        started = time.time()
        probe_step_to_index: dict[int, int] = {}
        if epoch == 1 or epoch % args.gradient_probe_every == 0:
            probe_count = min(args.gradient_probe_batches, steps_per_epoch)
            if probe_count == 1:
                selected_steps = [0]
            else:
                selected_steps = [
                    round(index * (steps_per_epoch - 1) / (probe_count - 1))
                    for index in range(probe_count)
                ]
            probe_step_to_index = {
                step: probe_index
                for probe_index, step in enumerate(selected_steps)
            }
        for epoch_step_index in range(steps_per_epoch):
            batches = {}
            for dataset in DATASETS:
                try:
                    x, y = next(iterators[dataset])
                except StopIteration:
                    data[dataset]["train_loader"].shuffle()
                    iterators[dataset] = data[dataset]["train_loader"].get_iterator()
                    x, y = next(iterators[dataset])
                batches[dataset] = prepare_batch(x, y, device)
            global_step += 1
            probe_index = probe_step_to_index.get(epoch_step_index)
            conflict_context = (
                (epoch, probe_index, epoch_step_index + 1, global_step)
                if probe_index is not None
                else None
            )
            values, norm, was_clipped, step_conflicts = trainer.train_step(
                batches,
                conflict_context=conflict_context,
            )
            conflict_rows.extend(step_conflicts)
            for dataset in DATASETS:
                train_values[dataset].append(values[dataset])
            grad_norms.append(norm)
            clipped.append(was_clipped)
        if probe_step_to_index:
            write_training_conflict_outputs(conflict_rows, output_dir)
        train_seconds = time.time() - started

        validation_values = {dataset: [] for dataset in DATASETS}
        for dataset in DATASETS:
            for x, y in data[dataset]["val_loader"].get_iterator():
                model_input, target = prepare_batch(x, y, device)
                validation_values[dataset].append(
                    trainer.eval_batch(dataset, model_input, target)
                )
        train_metrics = {
            dataset: aggregate_metrics(train_values[dataset]) for dataset in DATASETS
        }
        valid_metrics = {
            dataset: aggregate_metrics(validation_values[dataset])
            for dataset in DATASETS
        }
        normalized_mae = {
            dataset: valid_metrics[dataset]["mae"] / float(scalers[dataset].std)
            for dataset in DATASETS
        }
        score = float(np.mean(list(normalized_mae.values())))
        trainer.scheduler.step(score)
        current_lr = float(trainer.optimizer.param_groups[0]["lr"])


        peak_gib = (
            torch.cuda.max_memory_allocated(device) / 1024**3
            if device.type == "cuda" else 0.0
        )
        row = {
            "epoch": epoch,
            "macro_valid_normalized_mae": score,
            "lrate": current_lr,
            "grad_norm_mean": float(np.mean(grad_norms)),
            "grad_norm_max": float(np.max(grad_norms)),
            "grad_clip_rate": float(np.mean(clipped)),
            "train_seconds": train_seconds,
            "peak_memory_gib": peak_gib,
        }
        for dataset in DATASETS:
            for metric in METRICS:
                row[f"{dataset}_train_{metric}"] = train_metrics[dataset][metric]
                row[f"{dataset}_valid_{metric}"] = valid_metrics[dataset][metric]
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "train.csv", index=False)
        valid_text = " | ".join(
            f"{dataset.upper().replace('PEMS', 'P')} {valid_metrics[dataset]['mae']:.4f}"
            for dataset in DATASETS
        )
        print(
            f"Epoch {epoch:03d} | {valid_text} | macro-norm {score:.4f} | "
            f"{train_seconds:.1f}s | grad {np.mean(grad_norms):.2f}/"
            f"{np.max(grad_norms):.2f} | {peak_gib:.2f} GiB | lr {current_lr:.2g}",
            flush=True,
        )
        if score < best_score:
            best_score, best_epoch, stale_epochs = score, epoch, 0
            torch.save(trainer.model.state_dict(), checkpoint)
        else:
            stale_epochs += 1
        if epoch >= args.min_epochs and stale_epochs >= args.es_patience:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break

    trainer.model.load_state_dict(torch.load(checkpoint, map_location=device))
    conflict_summary, pair_summary = write_training_conflict_outputs(
        conflict_rows, output_dir
    )

    test_results = {}
    for dataset in DATASETS:
        frame, average = evaluate_test_set(
            trainer, dataset, data[dataset], device, args.output_len
        )
        frame.to_csv(output_dir / f"test_{dataset}.csv", index=False)
        test_results[dataset] = average
        print(
            f"Test {dataset}: MAE {average['mae']:.4f}, RMSE {average['rmse']:.4f}",
            flush=True,
        )

    most_frequent = conflict_summary.iloc[0].to_dict()
    strongest = conflict_summary.sort_values("mean_cosine").iloc[0].to_dict()
    strongest_tug = conflict_summary.sort_values(
        "mean_tug_of_war_strength", ascending=False
    ).iloc[0].to_dict()
    summary = {
        "best_epoch": best_epoch,
        "best_macro_validation_normalized_mae": best_score,
        "model_parameters": parameters,
        "trainable_parameters": trainer.model.count_trainable_params(),
        "shared_parameters": shared_parameter_count,
        "task_specific_parameters": parameters - shared_parameter_count,
        "datasets": list(DATASETS),
        "time_slots": trainer.model.backbone.time_slots,
        "data_protocol": "dataset-native node counts; day-aligned 60/20/20 split",
        "loss_weighting": "equal standardized MAE",
        "proportional_task_batches": args.proportional_task_batches,
        "task_batch_sizes": task_batch_sizes,
        "steps_per_epoch": steps_per_epoch,
        "gradient_probe_every": args.gradient_probe_every,
        "gradient_probe_batches": min(args.gradient_probe_batches, steps_per_epoch),
        "shared_components": "value/temporal/fusion encoders, all QK Transformer layers, final norm",
        "dataset_specific_components": "node embeddings and prediction heads (excluded from conflict analysis)",
        "gradient_probe": "exact per-task gradients from sampled joint training steps",
        "highest_conflict_frequency": most_frequent,
        "strongest_conflict_by_mean_cosine": strongest,
        "strongest_tug_of_war": strongest_tug,
        "test_average": test_results,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Most frequent shared-gradient conflict: {most_frequent['group']} "
        f"(any-pair rate {most_frequent['any_pair_conflict_rate']:.1%}, "
        f"pairwise rate {most_frequent['pairwise_conflict_rate']:.1%})",
        flush=True,
    )
    print(
        f"Lowest-cosine shared-gradient conflict: {strongest['group']} "
        f"(mean cosine {strongest['mean_cosine']:.4f}, "
        f"pairwise rate {strongest['pairwise_conflict_rate']:.1%})",
        flush=True,
    )
    print(
        f"Strongest tug-of-war: {strongest_tug['group']} "
        f"(mean strength {strongest_tug['mean_tug_of_war_strength']:.4f}, "
        f"max {strongest_tug['max_tug_of_war_strength']:.4f})",
        flush=True,
    )


if __name__ == "__main__":
    main()
