"""Train the three Transformer/adaptive-graph ablations.

Examples:
    python -u train_transformer_ablation.py --data bike_drop --attention_mode qk
    python -u train_transformer_ablation.py --data bike_drop \
        --attention_mode graph --graph_type adaptive
    python -u train_transformer_ablation.py --data bike_drop \
        --attention_mode qk_graph --graph_type adaptive

This is a standalone entry point and does not import or modify train_plus.py.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

import util
from dataset_config import DATASET_CONFIG, GRAPH_TYPES, get_dataset_config
from model_ST_Transformer_adaptive import ATTENTION_MODES, STTransformerAdaptive
from ranger21 import Ranger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ST-LLM+ QK/adaptive-graph Transformer ablations"
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data", choices=sorted(DATASET_CONFIG), default="bike_drop")
    parser.add_argument(
        "--attention_mode",
        choices=ATTENTION_MODES,
        default="qk",
        help="qk: standard attention; graph: A@V; qk_graph: QK plus log(A) prior",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Override the dataset_path declared in DATASET_CONFIG",
    )
    parser.add_argument(
        "--graph_type",
        choices=GRAPH_TYPES,
        default=None,
        help="Required by graph/qk_graph; declares graph semantics explicitly",
    )
    parser.add_argument(
        "--graph_path",
        "--adaptive_adj_path",
        dest="graph_path",
        type=str,
        default=None,
        help="Override the selected graph path; supports .npy/.pkl/.pickle",
    )

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=None,
        help="Validation/test batch size; defaults to --batch_size",
    )
    parser.add_argument(
        "--precision",
        choices=("fp32", "bf16", "fp16"),
        default="fp32",
        help="Training precision; bf16 is recommended on A100-class GPUs",
    )
    parser.add_argument(
        "--disable_tf32",
        action="store_true",
        help="Disable TF32 acceleration for FP32 CUDA matrix multiplications",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lrate", type=float, default=1e-3)
    parser.add_argument("--wdecay", type=float, default=1e-4)
    parser.add_argument(
        "--loss_space",
        choices=("auto", "original", "standardized"),
        default="auto",
        help=(
            "Backward MAE space; metrics always remain in original units. "
            "auto uses the dataset recommendation."
        ),
    )
    parser.add_argument("--input_dim", type=int, default=3)
    parser.add_argument("--input_len", type=int, default=12)
    parser.add_argument("--output_len", type=int, default=12)
    parser.add_argument(
        "--lr_scheduler",
        choices=("none", "plateau"),
        default="none",
        help="Optional validation-MAE learning-rate schedule",
    )
    parser.add_argument("--lr_patience", type=int, default=5)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--min_lrate", type=float, default=1e-5)

    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--ffn_dim", type=int, default=3072)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--graph_alpha",
        type=float,
        default=1.0,
        help="Initial alpha for qk_graph; each layer learns its own non-negative value",
    )
    parser.add_argument("--graph_epsilon", type=float, default=1e-8)

    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--es_patience", type=int, default=100)
    parser.add_argument(
        "--min_epochs",
        type=int,
        default=200,
        help="Do not early-stop before this epoch",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Output root; default is a new timestamped directory under logs/",
    )
    parser.add_argument("--print_model", action="store_true")
    return parser.parse_args()


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


def validate_args(args: argparse.Namespace) -> None:
    positive_int_fields = (
        "batch_size",
        "epochs",
        "input_dim",
        "input_len",
        "output_len",
        "d_model",
        "num_heads",
        "num_layers",
        "ffn_dim",
        "embedding_dim",
    )
    for field in positive_int_fields:
        if getattr(args, field) <= 0:
            raise ValueError(f"--{field} must be positive")
    if args.d_model % args.num_heads != 0:
        raise ValueError("--d_model must be divisible by --num_heads")
    if not 0 <= args.dropout < 1:
        raise ValueError("--dropout must be in [0, 1)")
    if args.graph_alpha <= 0:
        raise ValueError("--graph_alpha must be positive")
    if args.graph_epsilon <= 0:
        raise ValueError("--graph_epsilon must be positive")
    if args.es_patience <= 0:
        raise ValueError("--es_patience must be positive")
    if args.min_epochs < 0:
        raise ValueError("--min_epochs cannot be negative")
    if args.eval_batch_size is not None and args.eval_batch_size <= 0:
        raise ValueError("--eval_batch_size must be positive")
    if args.precision != "fp32" and not torch.cuda.is_available():
        raise ValueError(f"--precision {args.precision} requires CUDA")
    if args.lr_patience <= 0:
        raise ValueError("--lr_patience must be positive")
    if not 0 < args.lr_factor < 1:
        raise ValueError("--lr_factor must be in (0, 1)")
    if args.min_lrate <= 0 or args.min_lrate >= args.lrate:
        raise ValueError("--min_lrate must be positive and less than --lrate")

    if args.attention_mode == "qk":
        if args.graph_type is not None or args.graph_path is not None:
            raise ValueError("qk mode does not accept --graph_type or --graph_path")
    elif args.graph_type is None:
        raise ValueError(
            f"--graph_type is required for attention_mode={args.attention_mode!r}; "
            "choose adaptive, physical, or semantic"
        )


def load_graph(
    args: argparse.Namespace,
) -> tuple[Optional[np.ndarray], Optional[Path], Optional[dict]]:
    if args.attention_mode == "qk":
        return None, None, None

    dataset_config = get_dataset_config(args.data)
    if args.graph_path is not None:
        graph_path = Path(args.graph_path)
    else:
        try:
            graph_path = dataset_config.graphs[args.graph_type]
        except KeyError as error:
            available = ", ".join(sorted(dataset_config.graphs)) or "none"
            raise ValueError(
                f"Dataset {args.data!r} has no configured {args.graph_type!r} graph; "
                f"available graph types: {available}. Pass --graph_path explicitly "
                "if a compatible graph has been prepared."
            ) from error
    if not graph_path.is_file():
        raise FileNotFoundError(f"Graph does not exist: {graph_path}")

    graph = np.asarray(util.load_graph_data(str(graph_path)), dtype=np.float32)
    num_nodes = dataset_config.num_nodes
    expected_shape = (num_nodes, num_nodes)
    if graph.shape != expected_shape:
        raise ValueError(
            f"Graph shape mismatch for {args.data}: expected "
            f"{expected_shape}, got {graph.shape} from {graph_path}"
        )
    if not np.isfinite(graph).all():
        raise ValueError(f"Graph contains NaN/Inf: {graph_path}")
    if np.any(graph < 0):
        raise ValueError(f"Graph contains negative weights: {graph_path}")

    row_sums = graph.sum(axis=1)
    nonzero_per_row = np.count_nonzero(graph, axis=1)
    top_k = min(10, num_nodes)
    top10_mass = np.partition(graph, -top_k, axis=1)[:, -top_k:].sum(axis=1)
    stats = {
        "graph_type": args.graph_type,
        "file_format": graph_path.suffix.lower(),
        "path": str(graph_path),
        "shape": list(graph.shape),
        "dtype": str(graph.dtype),
        "minimum": float(graph.min()),
        "maximum": float(graph.max()),
        "row_sum_min": float(row_sums.min()),
        "row_sum_max": float(row_sums.max()),
        "nonzero_per_row_min": int(nonzero_per_row.min()),
        "nonzero_per_row_max": int(nonzero_per_row.max()),
        "top10_mass_mean": float(top10_mass.mean()),
    }
    return graph, graph_path, stats


def make_output_dir(args: argparse.Namespace) -> Path:
    if args.save_dir is None:
        timestamp = time.strftime("%Y-%m-%d-%H%M%S")
        output_root = Path("logs") / f"transformer_ablation_{timestamp}"
    else:
        output_root = Path(args.save_dir)
    output_dir = output_root / args.data / args.attention_mode / f"seed_{args.seed}"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty experiment directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


class Trainer:
    def __init__(
        self,
        args: argparse.Namespace,
        scaler: util.StandardScaler,
        adaptive_graph: Optional[np.ndarray],
        device: torch.device,
    ) -> None:
        dataset_config = get_dataset_config(args.data)
        self.loss_space = (
            dataset_config.loss_space
            if args.loss_space == "auto"
            else args.loss_space
        )
        self.model = STTransformerAdaptive(
            adaptive_graph=adaptive_graph,
            time_slots=dataset_config.time_slots,
            input_dim=args.input_dim,
            num_nodes=dataset_config.num_nodes,
            input_len=args.input_len,
            output_len=args.output_len,
            attention_mode=args.attention_mode,
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            ffn_dim=args.ffn_dim,
            embedding_dim=args.embedding_dim,
            dropout=args.dropout,
            graph_alpha=args.graph_alpha,
            graph_epsilon=args.graph_epsilon,
        ).to(device)
        self.optimizer = Ranger(
            self.model.parameters(), lr=args.lrate, weight_decay=args.wdecay
        )
        self.lr_scheduler = (
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
        self.scaler = scaler
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

        print(f"Model parameters: {self.model.param_num():,}")
        print(f"Trainable parameters: {self.model.count_trainable_params():,}")
        print(f"Training precision: {args.precision}")
        print(f"Optimization loss space: {self.loss_space}")
        print(f"LR scheduler: {args.lr_scheduler}")
        if args.print_model:
            print(self.model)

    def step_lr_scheduler(self, validation_loss: float) -> float:
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(validation_loss)
        return float(self.optimizer.param_groups[0]["lr"])

    def _metrics(
        self, model_input: torch.Tensor, real_value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.model(model_input).transpose(1, 3)
        real = real_value.unsqueeze(1)
        prediction = self.scaler.inverse_transform(output.float())
        loss = util.MAE_torch(prediction, real, 0.0)
        mape = util.MAPE_torch(prediction, real, 0.0)
        rmse = util.RMSE_torch(prediction, real, 0.0)
        wmape = util.WMAPE_torch(prediction, real, 0.0)
        return loss, mape, rmse, wmape

    def train_batch(
        self, model_input: torch.Tensor, real_value: torch.Tensor
    ) -> tuple[float, float, float, float, float, float, int]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(
            enabled=self.amp_enabled, dtype=self.amp_dtype
        ):
            metrics = self._metrics(model_input, real_value)
        optimization_loss = metrics[0]
        if self.loss_space == "standardized":
            optimization_loss = optimization_loss / self.scaler.std
        self.loss_scaler.scale(optimization_loss).backward()
        self.loss_scaler.unscale_(self.optimizer)
        max_norm = self.grad_clip if self.grad_clip > 0 else float("inf")
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=max_norm
        )
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"Non-finite gradient norm: {grad_norm.item()}")
        grad_norm_value = grad_norm.detach().item()
        was_clipped = float(self.grad_clip > 0 and grad_norm_value > self.grad_clip)
        self.loss_scaler.step(self.optimizer)
        self.loss_scaler.update()
        return (
            *(metric.detach().item() for metric in metrics),
            grad_norm_value,
            was_clipped,
            model_input.shape[0],
        )

    @torch.no_grad()
    def eval_batch(
        self, model_input: torch.Tensor, real_value: torch.Tensor
    ) -> tuple[float, float, float, float, int]:
        self.model.eval()
        with torch.cuda.amp.autocast(
            enabled=self.amp_enabled, dtype=self.amp_dtype
        ):
            metrics = self._metrics(model_input, real_value)
        return (*(metric.item() for metric in metrics), model_input.shape[0])


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


def aggregate_metrics(values: list[tuple[float, ...]]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.shape[1] == 7:
        weights = array[:, 6]
    elif array.shape[1] == 5:
        weights = array[:, 4]
    else:
        weights = np.ones(len(array), dtype=np.float64)
    result = {
        "loss": float(np.average(array[:, 0], weights=weights)),
        "mape": float(np.average(array[:, 1], weights=weights)),
        "rmse": float(np.average(array[:, 2], weights=weights)),
        "wmape": float(np.average(array[:, 3], weights=weights)),
        "samples": int(weights.sum()),
    }
    if array.shape[1] == 7:
        result.update(
            {
                "grad_norm_mean": float(array[:, 4].mean()),
                "grad_norm_max": float(array[:, 4].max()),
                "grad_clip_rate": float(array[:, 5].mean()),
            }
        )
    return result


@torch.no_grad()
def evaluate_test_set(
    trainer: Trainer,
    dataloader: dict,
    device: torch.device,
    output_len: int,
) -> tuple[pd.DataFrame, dict]:
    trainer.model.eval()
    outputs = []
    for x, _ in dataloader["test_loader"].get_iterator():
        model_input = torch.as_tensor(x, dtype=torch.float32, device=device)
        model_input = model_input.transpose(1, 3)
        with torch.cuda.amp.autocast(
            enabled=trainer.amp_enabled, dtype=trainer.amp_dtype
        ):
            prediction = trainer.model(model_input)
        prediction = prediction.float().transpose(1, 3).squeeze(1)
        outputs.append(prediction)

    yhat = torch.cat(outputs, dim=0)
    real = torch.as_tensor(
        dataloader["y_test"], dtype=torch.float32, device=device
    ).transpose(1, 3)[:, 0, :, :]
    yhat = yhat[: real.size(0)]

    horizon_rows = []
    for horizon in range(output_len):
        prediction = trainer.scaler.inverse_transform(yhat[:, :, horizon])
        horizon_real = real[:, :, horizon]
        mae, mape, rmse, wmape = util.metric(prediction, horizon_real)
        row = {
            "horizon": horizon + 1,
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "wmape": wmape,
        }
        horizon_rows.append(row)
        print(
            f"Horizon {horizon + 1:02d} | MAE {mae:.4f} | RMSE {rmse:.4f} "
            f"| MAPE {mape:.4f} | WMAPE {wmape:.4f}"
        )

    averages = {
        "horizon": "average",
        "mae": float(np.mean([row["mae"] for row in horizon_rows])),
        "rmse": float(np.mean([row["rmse"] for row in horizon_rows])),
        "mape": float(np.mean([row["mape"] for row in horizon_rows])),
        "wmape": float(np.mean([row["wmape"] for row in horizon_rows])),
    }
    return pd.DataFrame(horizon_rows + [averages]), averages


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

    dataset_config = get_dataset_config(args.data)
    dataset_path = Path(args.dataset_path) if args.dataset_path else dataset_config.dataset_path
    adaptive_graph, graph_path, graph_stats = load_graph(args)
    output_dir = make_output_dir(args)
    checkpoint_path = output_dir / "best_model.pth"
    eval_batch_size = args.eval_batch_size or args.batch_size

    config = vars(args).copy()
    config.update(
        {
            "num_nodes": dataset_config.num_nodes,
            "time_slots": dataset_config.time_slots,
            "dataset_recommended_training": dict(
                dataset_config.recommended_training
            ),
            "resolved_loss_space": (
                dataset_config.loss_space
                if args.loss_space == "auto"
                else args.loss_space
            ),
            "dataset_path": str(dataset_path),
            "resolved_graph_path": (
                str(graph_path) if graph_path is not None else None
            ),
            "output_dir": str(output_dir),
            "resolved_eval_batch_size": eval_batch_size,
        }
    )
    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)
    if graph_stats is not None:
        with (output_dir / "graph_stats.json").open("w", encoding="utf-8") as file:
            json.dump(graph_stats, file, ensure_ascii=False, indent=2)

    print(f"Dataset: {args.data}")
    print(f"Attention mode: {args.attention_mode}")
    print(f"Graph type: {args.graph_type if graph_path else 'not used'}")
    print(f"Graph path: {graph_path if graph_path else 'not used'}")
    print(f"Output directory: {output_dir}")
    print(f"Train/eval batch size: {args.batch_size}/{eval_batch_size}")
    print(
        f"TF32 enabled: {device.type == 'cuda' and not args.disable_tf32}"
    )

    dataloader = util.load_dataset(
        str(dataset_path),
        args.batch_size,
        eval_batch_size,
        eval_batch_size,
        expected_num_nodes=dataset_config.num_nodes,
        expected_input_len=args.input_len,
        expected_output_len=args.output_len,
        expected_input_dim=args.input_dim,
    )
    trainer = Trainer(
        args=args,
        scaler=dataloader["scaler"],
        adaptive_graph=adaptive_graph,
        device=device,
    )

    best_validation_mae = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    train_times = []
    validation_times = []

    for epoch in range(1, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.time()
        dataloader["train_loader"].shuffle()
        train_values = []
        for x, y in dataloader["train_loader"].get_iterator():
            model_input, target = prepare_batch(x, y, device)
            train_values.append(trainer.train_batch(model_input, target))
        train_times.append(time.time() - start)
        train_throughput = dataloader["train_loader"].size / train_times[-1]
        peak_memory_gib = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            if device.type == "cuda"
            else 0.0
        )

        start = time.time()
        validation_values = []
        for x, y in dataloader["val_loader"].get_iterator():
            model_input, target = prepare_batch(x, y, device)
            validation_values.append(trainer.eval_batch(model_input, target))
        validation_times.append(time.time() - start)

        train_metrics = aggregate_metrics(train_values)
        validation_metrics = aggregate_metrics(validation_values)
        current_lrate = trainer.step_lr_scheduler(validation_metrics["loss"])
        alphas = trainer.model.graph_alphas()
        row = {
            "epoch": epoch,
            "lrate": current_lrate,
            "train_loss": train_metrics["loss"],
            "train_rmse": train_metrics["rmse"],
            "train_mape": train_metrics["mape"],
            "grad_norm_mean": train_metrics["grad_norm_mean"],
            "grad_norm_max": train_metrics["grad_norm_max"],
            "grad_clip_rate": train_metrics["grad_clip_rate"],
            "train_wmape": train_metrics["wmape"],
            "valid_loss": validation_metrics["loss"],
            "valid_rmse": validation_metrics["rmse"],
            "valid_mape": validation_metrics["mape"],
            "valid_wmape": validation_metrics["wmape"],
            "train_seconds": train_times[-1],
            "train_samples_per_second": train_throughput,
            "peak_memory_gib": peak_memory_gib,
            "validation_seconds": validation_times[-1],
            "graph_alpha_mean": float(np.mean(alphas)) if alphas else np.nan,
            "graph_alphas": json.dumps(alphas),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "train.csv", index=False)

        print(
            f"Epoch {epoch:03d} | train MAE {train_metrics['loss']:.4f} "
            f"RMSE {train_metrics['rmse']:.4f} | valid MAE "
            f"{validation_metrics['loss']:.4f} RMSE "
            f"{validation_metrics['rmse']:.4f} | "
            f"grad {train_metrics['grad_norm_mean']:.2f}/"
            f"{train_metrics['grad_norm_max']:.2f} clip "
            f"{train_metrics['grad_clip_rate']:.1%} | {train_times[-1]:.2f}s train "
            f"{validation_times[-1]:.2f}s valid | {train_throughput:.1f} sample/s "
            f"{peak_memory_gib:.2f} GiB | lr {current_lrate:.2g}"
        )
        if alphas:
            print("Graph alphas: " + ", ".join(f"{value:.4f}" for value in alphas))

        if validation_metrics["loss"] < best_validation_mae:
            best_validation_mae = validation_metrics["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(trainer.model.state_dict(), checkpoint_path)
            print(f"Saved new best checkpoint (validation MAE {best_validation_mae:.4f})")
        else:
            epochs_without_improvement += 1

        if (
            epoch >= args.min_epochs
            and epochs_without_improvement >= args.es_patience
        ):
            print(
                f"Early stopping after {epochs_without_improvement} epochs "
                "without validation improvement"
            )
            break

    print(
        f"Best epoch: {best_epoch}; best validation MAE: "
        f"{best_validation_mae:.4f}"
    )
    print(f"Average training time: {np.mean(train_times):.4f} seconds/epoch")
    print(
        f"Average validation time: {np.mean(validation_times):.4f} seconds/epoch"
    )

    trainer.model.load_state_dict(
        torch.load(checkpoint_path, map_location=device)
    )
    test_frame, test_average = evaluate_test_set(
        trainer=trainer,
        dataloader=dataloader,
        device=device,
        output_len=args.output_len,
    )
    test_frame.to_csv(output_dir / "test.csv", index=False)

    summary = {
        "best_epoch": best_epoch,
        "best_validation_mae": best_validation_mae,
        "test_average": test_average,
        "model_parameters": trainer.model.param_num(),
        "trainable_parameters": trainer.model.count_trainable_params(),
        "graph_alphas": trainer.model.graph_alphas(),
        "final_lrate": float(trainer.optimizer.param_groups[0]["lr"]),
        "mean_grad_clip_rate": float(
            np.mean([row["grad_clip_rate"] for row in history])
        ),
        "peak_memory_gib": float(max(row["peak_memory_gib"] for row in history)),
        "average_train_samples_per_second": float(
            np.mean([row["train_samples_per_second"] for row in history])
        ),
        "average_train_seconds": float(np.mean(train_times)),
        "average_validation_seconds": float(np.mean(validation_times)),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(
        "Test average | "
        f"MAE {test_average['mae']:.4f} | RMSE {test_average['rmse']:.4f} "
        f"| MAPE {test_average['mape']:.4f} | "
        f"WMAPE {test_average['wmape']:.4f}"
    )


if __name__ == "__main__":
    main()

