#!/usr/bin/env python3
"""Jointly train one QK Transformer on all four Taxi/Bike datasets.

The QK Transformer backbone is shared. Each dataset owns its node embedding,
prediction head, and scaler because node identities and output mappings are not
shared across traffic systems. One optimizer update accumulates one batch
from every dataset, so an epoch has the same number of optimizer updates and
per-dataset exposures as the corresponding single-dataset experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import util
from dataset_config import get_dataset_config
from model_ST_Transformer_adaptive import STTransformerAdaptive
from ranger21 import Ranger


DATASETS = ("taxi_drop", "taxi_pick", "bike_drop", "bike_pick")
METRICS = ("mae", "mape", "rmse", "wmape")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=384)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument(
        "--precision", choices=("fp32", "bf16", "fp16"), default="bf16"
    )
    parser.add_argument("--disable_tf32", action="store_true")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--min_epochs", type=int, default=200)
    parser.add_argument("--es_patience", type=int, default=100)
    parser.add_argument("--lrate", type=float, default=1e-3)
    parser.add_argument("--wdecay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument(
        "--lr_scheduler", choices=("none", "plateau"), default="none"
    )
    parser.add_argument("--lr_patience", type=int, default=5)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--min_lrate", type=float, default=1e-5)

    parser.add_argument("--input_dim", type=int, default=3)
    parser.add_argument("--input_len", type=int, default=12)
    parser.add_argument("--output_len", type=int, default=12)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--ffn_dim", type=int, default=3072)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--save_dir", type=Path, default=None)
    parser.add_argument("--single_results", type=Path, default=Path("result.md"))
    parser.add_argument("--print_model", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        "batch_size",
        "eval_batch_size",
        "epochs",
        "es_patience",
        "input_dim",
        "input_len",
        "output_len",
        "d_model",
        "num_heads",
        "num_layers",
        "ffn_dim",
        "embedding_dim",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name} must be positive")
    if args.min_epochs < 0:
        raise ValueError("--min_epochs cannot be negative")
    if args.d_model % args.num_heads:
        raise ValueError("--d_model must be divisible by --num_heads")
    if not 0 <= args.dropout < 1:
        raise ValueError("--dropout must be in [0, 1)")
    if args.precision != "fp32" and not torch.cuda.is_available():
        raise ValueError(f"--precision {args.precision} requires CUDA")
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("The selected CUDA device does not support BF16")
    if not 0 < args.lr_factor < 1:
        raise ValueError("--lr_factor must be in (0, 1)")
    if not 0 < args.min_lrate < args.lrate:
        raise ValueError("--min_lrate must be between 0 and --lrate")


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


class JointQKTransformer(nn.Module):
    """Shared QK backbone with dataset-specific node embeddings and heads."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        configs = {name: get_dataset_config(name) for name in DATASETS}
        time_slots = {config.time_slots for config in configs.values()}
        if len(time_slots) != 1:
            raise ValueError(
                "Joint training requires the same time-slot convention; got "
                f"{sorted(time_slots)}"
            )

        self.backbone = STTransformerAdaptive(
            adaptive_graph=None,
            time_slots=time_slots.pop(),
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
                name: nn.Parameter(
                    torch.empty(config.num_nodes, args.embedding_dim)
                )
                for name, config in configs.items()
            }
        )
        for embedding in self.node_embeddings.values():
            nn.init.xavier_uniform_(embedding)
        self.prediction_heads = nn.ModuleDict(
            {
                name: nn.Conv2d(
                    args.d_model, args.output_len, kernel_size=(1, 1)
                )
                for name in DATASETS
            }
        )

    def forward(self, dataset: str, history_data: torch.Tensor) -> torch.Tensor:
        try:
            prediction_head = self.prediction_heads[dataset]
            node_embedding = self.node_embeddings[dataset]
        except KeyError as error:
            raise ValueError(f"Unknown joint-training dataset: {dataset}") from error
        return self.backbone(
            history_data,
            prediction_head_override=prediction_head,
            node_embedding_override=node_embedding,
        )

    def param_num(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def count_trainable_params(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )


def prepare_batch(
    x: np.ndarray,
    y: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model_input = torch.as_tensor(x, dtype=torch.float32, device=device)
    model_input = model_input.transpose(1, 3)
    target = torch.as_tensor(y, dtype=torch.float32, device=device)
    target = target.transpose(1, 3)[:, 0, :, :]
    return model_input, target


class JointTrainer:
    def __init__(
        self,
        args: argparse.Namespace,
        scalers: dict[str, util.StandardScaler],
        device: torch.device,
    ) -> None:
        self.model = JointQKTransformer(args).to(device)
        self.optimizer = Ranger(
            self.model.parameters(),
            lr=args.lrate,
            weight_decay=args.wdecay,
        )
        self.scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=args.lr_factor,
                patience=args.lr_patience,
                min_lr=args.min_lrate,
            )
            if args.lr_scheduler == "plateau"
            else None
        )
        self.scalers = scalers
        self.grad_clip = args.grad_clip
        self.amp_enabled = args.precision != "fp32"
        self.amp_dtype = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[args.precision]
        self.loss_scaler = torch.cuda.amp.GradScaler(
            enabled=args.precision == "fp16"
        )

    def _metrics(
        self,
        dataset: str,
        model_input: torch.Tensor,
        target: torch.Tensor,
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

    def train_joint_step(
        self,
        batches: dict[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[dict[str, tuple[float, ...]], float, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        task_metrics: dict[str, tuple[float, ...]] = {}
        task_count = len(batches)

        for dataset in DATASETS:
            model_input, target = batches[dataset]
            with torch.cuda.amp.autocast(
                enabled=self.amp_enabled,
                dtype=self.amp_dtype,
            ):
                metrics = self._metrics(dataset, model_input, target)
                dataset_scale = float(self.scalers[dataset].std)
                joint_loss = (
                    metrics[0] / dataset_scale / task_count
                )
            self.loss_scaler.scale(joint_loss).backward()
            task_metrics[dataset] = (
                *(metric.detach().item() for metric in metrics),
                model_input.shape[0],
            )

        self.loss_scaler.unscale_(self.optimizer)
        max_norm = self.grad_clip if self.grad_clip > 0 else float("inf")
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=max_norm,
        )
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(
                f"Non-finite joint gradient norm: {grad_norm.item()}"
            )
        grad_norm_value = grad_norm.detach().item()
        was_clipped = float(
            self.grad_clip > 0 and grad_norm_value > self.grad_clip
        )
        self.loss_scaler.step(self.optimizer)
        self.loss_scaler.update()
        return task_metrics, grad_norm_value, was_clipped

    @torch.no_grad()
    def eval_batch(
        self,
        dataset: str,
        model_input: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[float, ...]:
        self.model.eval()
        with torch.cuda.amp.autocast(
            enabled=self.amp_enabled,
            dtype=self.amp_dtype,
        ):
            metrics = self._metrics(dataset, model_input, target)
        return (*(metric.item() for metric in metrics), model_input.shape[0])

    def step_scheduler(self, score: float) -> float:
        if self.scheduler is not None:
            self.scheduler.step(score)
        return float(self.optimizer.param_groups[0]["lr"])


def aggregate_metrics(values: list[tuple[float, ...]]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot aggregate an empty metric list")
    array = np.asarray(values, dtype=np.float64)
    weights = array[:, 4]
    return {
        "mae": float(np.average(array[:, 0], weights=weights)),
        "mape": float(np.average(array[:, 1], weights=weights)),
        "rmse": float(np.average(array[:, 2], weights=weights)),
        "wmape": float(np.average(array[:, 3], weights=weights)),
        "samples": int(weights.sum()),
    }


@torch.no_grad()
def evaluate_test_set(
    trainer: JointTrainer,
    dataset: str,
    data: dict,
    device: torch.device,
    output_len: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    trainer.model.eval()
    outputs = []
    for x, _ in data["test_loader"].get_iterator():
        model_input = torch.as_tensor(x, dtype=torch.float32, device=device)
        model_input = model_input.transpose(1, 3)
        with torch.cuda.amp.autocast(
            enabled=trainer.amp_enabled,
            dtype=trainer.amp_dtype,
        ):
            prediction = trainer.model(dataset, model_input)
        outputs.append(prediction.float().transpose(1, 3).squeeze(1))

    predicted = torch.cat(outputs, dim=0)
    real = torch.as_tensor(
        data["y_test"],
        dtype=torch.float32,
        device=device,
    ).transpose(1, 3)[:, 0, :, :]
    predicted = predicted[: real.size(0)]

    rows = []
    for horizon in range(output_len):
        horizon_prediction = data["scaler"].inverse_transform(
            predicted[:, :, horizon]
        )
        horizon_real = real[:, :, horizon]
        mae, mape, rmse, wmape = util.metric(
            horizon_prediction,
            horizon_real,
        )
        rows.append(
            {
                "horizon": horizon + 1,
                "mae": mae,
                "rmse": rmse,
                "mape": mape,
                "wmape": wmape,
            }
        )

    average = {
        metric: float(np.mean([row[metric] for row in rows]))
        for metric in METRICS
    }
    average["horizon"] = "average"
    return pd.DataFrame(rows + [average]), average


def parse_single_qk_results(path: Path) -> dict[str, float]:
    if not path.is_file():
        return {}
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("|")
    ]
    if len(lines) < 3:
        raise ValueError(f"Cannot parse Markdown result table: {path}")

    def cells(line: str) -> list[str]:
        return [cell.strip() for cell in line.strip("|").split("|")]

    header = cells(lines[0])
    try:
        dataset_index = header.index("数据集")
        qk_index = header.index("QK")
    except ValueError as error:
        raise ValueError(f"{path} must contain 数据集 and QK columns") from error

    result = {}
    for line in lines[2:]:
        row = cells(line)
        if len(row) <= max(dataset_index, qk_index):
            continue
        dataset = row[dataset_index]
        if dataset not in DATASETS:
            continue
        value = row[qk_index].replace("*", "").strip()
        result[dataset] = float(value)
    return result


def write_comparison(
    path: Path,
    single_results: dict[str, float],
    joint_results: dict[str, dict[str, float]],
) -> dict[str, dict[str, float | str]]:
    comparisons: dict[str, dict[str, float | str]] = {}
    lines = [
        "# 四数据集联合 QK-Transformer 对比",
        "",
        "| 数据集 | 单独训练 QK MAE | 联合训练 QK MAE | 变化 | 变化率 | 结论 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    relative_changes = []
    for dataset in DATASETS:
        joint = float(joint_results[dataset]["mae"])
        single = single_results.get(dataset)
        if single is None:
            lines.append(f"| {dataset} | N/A | {joint:.4f} | N/A | N/A | 无基线 |")
            continue
        delta = joint - single
        relative = 100.0 * delta / single
        conclusion = "提升" if delta < 0 else "变差" if delta > 0 else "持平"
        relative_changes.append(relative)
        comparisons[dataset] = {
            "single_mae": single,
            "joint_mae": joint,
            "delta_mae": delta,
            "delta_percent": relative,
            "conclusion": conclusion,
        }
        lines.append(
            f"| {dataset} | {single:.4f} | {joint:.4f} | "
            f"{delta:+.4f} | {relative:+.2f}% | {conclusion} |"
        )

    if relative_changes:
        macro = float(np.mean(relative_changes))
        overall = "提升" if macro < 0 else "变差" if macro > 0 else "持平"
        lines.extend(
            [
                "",
                f"四个数据集 MAE 相对变化的宏平均为 **{macro:+.2f}%**，"
                f"联合训练总体表现为 **{overall}**。",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return comparisons


def make_output_dir(args: argparse.Namespace) -> Path:
    output_dir = args.save_dir
    if output_dir is None:
        timestamp = time.strftime("%Y-%m-%d-%H%M%S")
        output_dir = Path("logs") / f"qk_joint_{timestamp}"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty experiment directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


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
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested ({args.device}) but is unavailable")
    if device.type == "cuda":
        tf32_enabled = not args.disable_tf32
        torch.backends.cuda.matmul.allow_tf32 = tf32_enabled
        torch.backends.cudnn.allow_tf32 = tf32_enabled
        if tf32_enabled:
            torch.set_float32_matmul_precision("high")

    output_dir = make_output_dir(args)
    (output_dir / "config.json").write_text(
        json.dumps(serializable_args(args), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    data: dict[str, dict] = {}
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

    scalers = {dataset: data[dataset]["scaler"] for dataset in DATASETS}
    trainer = JointTrainer(args, scalers, device)
    if args.print_model:
        print(trainer.model)
    print(f"Output directory: {output_dir}")
    print(f"Model parameters: {trainer.model.param_num():,}")
    print(f"Trainable parameters: {trainer.model.count_trainable_params():,}")
    print(
        "Protocol: one batch per dataset per optimizer step; "
        "each MAE is divided by its training-set standard deviation before "
        "the four task losses are averaged"
    )

    print(
        "Shared components: input/temporal encoder, QK Transformer, final norm; "
        "dataset-specific components: node embedding, prediction head, scaler"
    )
    batch_counts = {
        dataset: data[dataset]["train_loader"].num_batch
        for dataset in DATASETS
    }
    if len(set(batch_counts.values())) != 1:
        raise ValueError(
            "Balanced joint steps require equal train batch counts; got "
            f"{batch_counts}"
        )
    steps_per_epoch = next(iter(batch_counts.values()))

    best_score = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    train_times = []
    validation_times = []
    checkpoint_path = output_dir / "best_model.pth"

    for epoch in range(1, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for dataset in DATASETS:
            data[dataset]["train_loader"].shuffle()
        iterators = {
            dataset: data[dataset]["train_loader"].get_iterator()
            for dataset in DATASETS
        }
        training_values = {dataset: [] for dataset in DATASETS}
        grad_norms = []
        clipped = []

        started = time.time()
        for _ in range(steps_per_epoch):
            batches = {}
            for dataset in DATASETS:
                x, y = next(iterators[dataset])
                batches[dataset] = prepare_batch(x, y, device)
            task_values, grad_norm, was_clipped = trainer.train_joint_step(
                batches
            )
            for dataset, values in task_values.items():
                training_values[dataset].append(values)
            grad_norms.append(grad_norm)
            clipped.append(was_clipped)
        train_seconds = time.time() - started
        train_times.append(train_seconds)

        started = time.time()
        validation_values = {dataset: [] for dataset in DATASETS}
        for dataset in DATASETS:
            for x, y in data[dataset]["val_loader"].get_iterator():
                model_input, target = prepare_batch(x, y, device)
                validation_values[dataset].append(
                    trainer.eval_batch(dataset, model_input, target)
                )
        validation_seconds = time.time() - started
        validation_times.append(validation_seconds)

        train_metrics = {
            dataset: aggregate_metrics(training_values[dataset])
            for dataset in DATASETS
        }
        validation_metrics = {
            dataset: aggregate_metrics(validation_values[dataset])
            for dataset in DATASETS
        }
        normalized_validation_mae = {
            dataset: validation_metrics[dataset]["mae"]
            / float(scalers[dataset].std)
            for dataset in DATASETS
        }
        macro_validation_score = float(
            np.mean(list(normalized_validation_mae.values()))
        )
        current_lrate = trainer.step_scheduler(macro_validation_score)
        peak_memory_gib = (
            torch.cuda.max_memory_allocated(device) / (1024**3)
            if device.type == "cuda"
            else 0.0
        )

        row = {
            "epoch": epoch,
            "lrate": current_lrate,
            "macro_valid_normalized_mae": macro_validation_score,
            "grad_norm_mean": float(np.mean(grad_norms)),
            "grad_norm_max": float(np.max(grad_norms)),
            "grad_clip_rate": float(np.mean(clipped)),
            "train_seconds": train_seconds,
            "validation_seconds": validation_seconds,
            "peak_memory_gib": peak_memory_gib,
        }
        for dataset in DATASETS:
            for metric in METRICS:
                row[f"{dataset}_train_{metric}"] = train_metrics[dataset][metric]
                row[f"{dataset}_valid_{metric}"] = validation_metrics[dataset][metric]
            row[f"{dataset}_valid_normalized_mae"] = (
                normalized_validation_mae[dataset]
            )
        history.append(row)
        pd.DataFrame(history).to_csv(
            output_dir / "train.csv",
            index=False,
        )

        valid_text = " | ".join(
            f"{dataset} {validation_metrics[dataset]['mae']:.4f}"
            for dataset in DATASETS
        )
        print(
            f"Epoch {epoch:03d} | valid MAE {valid_text} | "
            f"macro-norm {macro_validation_score:.4f} | "
            f"{train_seconds:.2f}s train {validation_seconds:.2f}s valid | "
            f"grad {np.mean(grad_norms):.2f}/{np.max(grad_norms):.2f} | "
            f"{peak_memory_gib:.2f} GiB | lr {current_lrate:.2g}",
            flush=True,
        )

        if macro_validation_score < best_score:
            best_score = macro_validation_score
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(trainer.model.state_dict(), checkpoint_path)
            print(
                f"Saved best joint checkpoint at epoch {epoch} "
                f"(macro normalized validation MAE {best_score:.4f})",
                flush=True,
            )
        else:
            epochs_without_improvement += 1

        if (
            epoch >= args.min_epochs
            and epochs_without_improvement >= args.es_patience
        ):
            print(
                f"Early stopping after {epochs_without_improvement} epochs "
                "without macro validation improvement",
                flush=True,
            )
            break

    trainer.model.load_state_dict(
        torch.load(checkpoint_path, map_location=device)
    )
    joint_test_results: dict[str, dict[str, float]] = {}
    for dataset in DATASETS:
        test_frame, average = evaluate_test_set(
            trainer,
            dataset,
            data[dataset],
            device,
            args.output_len,
        )
        test_frame.to_csv(output_dir / f"test_{dataset}.csv", index=False)
        joint_test_results[dataset] = average
        print(
            f"Test {dataset} | MAE {average['mae']:.4f} | "
            f"RMSE {average['rmse']:.4f} | MAPE {average['mape']:.4f} | "
            f"WMAPE {average['wmape']:.4f}",
            flush=True,
        )

    single_results = parse_single_qk_results(args.single_results)
    comparisons = write_comparison(
        output_dir / "comparison.md",
        single_results,
        joint_test_results,
    )
    summary = {
        "best_epoch": best_epoch,
        "best_macro_validation_normalized_mae": best_score,
        "test_average": joint_test_results,
        "single_vs_joint": comparisons,
        "model_parameters": trainer.model.param_num(),
        "trainable_parameters": trainer.model.count_trainable_params(),
        "steps_per_epoch": steps_per_epoch,
        "datasets_per_step": len(DATASETS),
        "loss_weighting": "equal standardized MAE (original MAE / train std)",
        "shared_components": "input/temporal encoder, QK Transformer, final norm",
        "dataset_specific_components": "node embedding, prediction head, scaler",
        "average_train_seconds": float(np.mean(train_times)),
        "average_validation_seconds": float(np.mean(validation_times)),
        "peak_memory_gib": float(max(row["peak_memory_gib"] for row in history)),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Comparison written to {output_dir / 'comparison.md'}")


if __name__ == "__main__":
    main()
