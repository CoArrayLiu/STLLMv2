#!/usr/bin/env python3
"""Train one QK Transformer depth-ablation variant from scratch.

The six supported variants remove the final one, two, or three Transformer
layers.  A ``reduced`` variant keeps the original width and therefore reduces
the parameter count.  A ``matched`` variant widens the remaining layers so
that the total parameter count stays close to the original six-layer model.

This is a standalone experiment entry point.  It reuses stable project
components but does not modify or overwrite previous training code or logs.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import util
from dataset_config import DATASET_CONFIG, get_dataset_config
from model_ST_Transformer_adaptive import STTransformerAdaptive
from train_transformer_ablation import (
    Trainer,
    aggregate_metrics,
    prepare_batch,
    seed_everything,
)


BASE_NUM_LAYERS = 6
BASE_D_MODEL = 768
BASE_FFN_DIM = 3072
BASE_NUM_HEADS = 12
RECOMMENDED_TRAIN_BATCH_SIZE = 652
RECOMMENDED_EVAL_BATCH_SIZE = 870


@dataclass(frozen=True)
class DepthVariant:
    name: str
    removed_layers: int
    parameter_mode: str
    num_layers: int
    d_model: int
    ffn_dim: int
    num_heads: int = BASE_NUM_HEADS


VARIANTS = {
    variant.name: variant
    for variant in (
        DepthVariant(
            "remove_last_1_reduced", 1, "reduced", 5, 768, 3072
        ),
        DepthVariant(
            "remove_last_1_matched", 1, "matched", 5, 840, 3368
        ),
        DepthVariant(
            "remove_last_2_reduced", 2, "reduced", 4, 768, 3072
        ),
        DepthVariant(
            "remove_last_2_matched", 2, "matched", 4, 936, 3784
        ),
        DepthVariant(
            "remove_last_3_reduced", 3, "reduced", 3, 768, 3072
        ),
        DepthVariant(
            "remove_last_3_matched", 3, "matched", 3, 1080, 4360
        ),
    )
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument(
        "--data",
        choices=tuple(sorted(DATASET_CONFIG)),
        required=True,
    )
    parser.add_argument("--dataset_path", type=Path, default=None)
    parser.add_argument("--save_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--precision",
        choices=("fp32", "bf16", "fp16"),
        default="bf16",
    )
    parser.add_argument("--disable_tf32", action="store_true")

    parser.add_argument(
        "--batch_size",
        type=int,
        default=RECOMMENDED_TRAIN_BATCH_SIZE,
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=RECOMMENDED_EVAL_BATCH_SIZE,
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--min_epochs", type=int, default=200)
    parser.add_argument("--es_patience", type=int, default=100)
    parser.add_argument("--lrate", type=float, default=1e-3)
    parser.add_argument("--wdecay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument(
        "--loss_space",
        choices=("auto", "original", "standardized"),
        default="auto",
    )
    parser.add_argument(
        "--lr_scheduler",
        choices=("none", "plateau"),
        default="none",
    )
    parser.add_argument("--lr_patience", type=int, default=5)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--min_lrate", type=float, default=1e-5)

    parser.add_argument("--input_dim", type=int, default=3)
    parser.add_argument("--input_len", type=int, default=12)
    parser.add_argument("--output_len", type=int, default=12)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--print_model", action="store_true")

    parser.add_argument(
        "--max_train_batches",
        type=int,
        default=None,
        help="Limit batches per epoch for smoke tests only.",
    )
    parser.add_argument(
        "--max_eval_batches",
        type=int,
        default=None,
        help="Limit validation batches for smoke tests only.",
    )
    parser.add_argument(
        "--max_test_batches",
        type=int,
        default=None,
        help="Limit test batches for smoke tests only.",
    )
    return parser.parse_args()


def resolve_variant_args(args: argparse.Namespace) -> DepthVariant:
    variant = VARIANTS[args.variant]
    args.num_layers = variant.num_layers
    args.d_model = variant.d_model
    args.ffn_dim = variant.ffn_dim
    args.num_heads = variant.num_heads

    # Fields consumed by the stable Trainer shared with the original entry
    # point.  The depth ablation is deliberately QK-only.
    args.attention_mode = "qk"
    args.graph_type = None
    args.graph_path = None
    args.graph_alpha = 1.0
    args.graph_epsilon = 1e-8
    return variant


def validate_args(args: argparse.Namespace, variant: DepthVariant) -> None:
    positive_ints = (
        "batch_size",
        "eval_batch_size",
        "epochs",
        "es_patience",
        "input_dim",
        "input_len",
        "output_len",
        "embedding_dim",
        "num_layers",
        "d_model",
        "ffn_dim",
        "num_heads",
    )
    for name in positive_ints:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name} must be positive")
    for name in (
        "max_train_batches",
        "max_eval_batches",
        "max_test_batches",
    ):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name} must be positive when provided")
    if args.min_epochs < 0:
        raise ValueError("--min_epochs cannot be negative")
    if args.d_model % args.num_heads:
        raise ValueError("Resolved d_model must be divisible by num_heads")
    if not 0 <= args.dropout < 1:
        raise ValueError("--dropout must be in [0, 1)")
    if args.grad_clip < 0:
        raise ValueError("--grad_clip cannot be negative")
    if not 0 < args.lr_factor < 1:
        raise ValueError("--lr_factor must be in (0, 1)")
    if not 0 < args.min_lrate < args.lrate:
        raise ValueError("--min_lrate must be between 0 and --lrate")
    if variant.num_layers != BASE_NUM_LAYERS - variant.removed_layers:
        raise AssertionError(f"Invalid layer count in {variant.name}")
    if args.precision != "fp32" and not torch.cuda.is_available():
        raise ValueError(f"--precision {args.precision} requires CUDA")
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("The selected CUDA device does not support BF16")


def build_reference_model(args: argparse.Namespace) -> STTransformerAdaptive:
    dataset = get_dataset_config(args.data)
    return STTransformerAdaptive(
        adaptive_graph=None,
        time_slots=dataset.time_slots,
        input_dim=args.input_dim,
        num_nodes=dataset.num_nodes,
        input_len=args.input_len,
        output_len=args.output_len,
        attention_mode="qk",
        d_model=BASE_D_MODEL,
        num_heads=BASE_NUM_HEADS,
        num_layers=BASE_NUM_LAYERS,
        ffn_dim=BASE_FFN_DIM,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
    )


def limited_batches(loader, maximum: int | None):
    for batch_index, batch in enumerate(loader.get_iterator()):
        if maximum is not None and batch_index >= maximum:
            break
        yield batch


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate_test_set(
    trainer: Trainer,
    dataloader: dict,
    device: torch.device,
    output_len: int,
    max_batches: int | None,
) -> tuple[pd.DataFrame, dict]:
    trainer.model.eval()
    predictions = []
    targets = []
    for x, y in limited_batches(dataloader["test_loader"], max_batches):
        model_input = torch.as_tensor(x, dtype=torch.float32, device=device)
        model_input = model_input.transpose(1, 3)
        with torch.cuda.amp.autocast(
            enabled=trainer.amp_enabled,
            dtype=trainer.amp_dtype,
        ):
            prediction = trainer.model(model_input)
        predictions.append(prediction.float().transpose(1, 3).squeeze(1))
        target = torch.as_tensor(y, dtype=torch.float32, device=device)
        targets.append(target.transpose(1, 3)[:, 0, :, :])

    if not predictions:
        raise RuntimeError("Test iterator produced no batches")
    yhat = torch.cat(predictions, dim=0)
    real = torch.cat(targets, dim=0)
    horizon_rows = []
    for horizon in range(output_len):
        prediction = trainer.scaler.inverse_transform(yhat[:, :, horizon])
        mae, mape, rmse, wmape = util.metric(
            prediction,
            real[:, :, horizon],
        )
        horizon_rows.append(
            {
                "horizon": horizon + 1,
                "mae": mae,
                "rmse": rmse,
                "mape": mape,
                "wmape": wmape,
            }
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
    variant = resolve_variant_args(args)
    validate_args(args, variant)
    seed_everything(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested ({args.device}) but unavailable")
    if device.type == "cuda":
        tf32_enabled = not args.disable_tf32
        torch.backends.cuda.matmul.allow_tf32 = tf32_enabled
        torch.backends.cudnn.allow_tf32 = tf32_enabled
        if tf32_enabled:
            torch.set_float32_matmul_precision("high")

    output_dir = args.save_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pth"

    dataset_config = get_dataset_config(args.data)
    dataset_path = args.dataset_path or dataset_config.dataset_path
    dataloader = util.load_dataset(
        str(dataset_path),
        args.batch_size,
        args.eval_batch_size,
        args.eval_batch_size,
        expected_num_nodes=dataset_config.num_nodes,
        expected_input_len=args.input_len,
        expected_output_len=args.output_len,
        expected_input_dim=args.input_dim,
    )

    reference_model = build_reference_model(args)
    reference_parameters = reference_model.param_num()
    del reference_model
    trainer = Trainer(
        args=args,
        scaler=dataloader["scaler"],
        adaptive_graph=None,
        device=device,
    )
    model_parameters = trainer.model.param_num()
    parameter_ratio = model_parameters / reference_parameters
    if variant.parameter_mode == "matched" and abs(parameter_ratio - 1.0) > 0.01:
        raise AssertionError(
            f"Matched variant differs from baseline by more than 1%: "
            f"ratio={parameter_ratio:.6f}"
        )
    if variant.parameter_mode == "reduced" and parameter_ratio >= 0.9:
        raise AssertionError(
            f"Reduced variant did not materially reduce parameters: "
            f"ratio={parameter_ratio:.6f}"
        )

    config = vars(args).copy()
    config.update(
        {
            "variant_spec": asdict(variant),
            "dataset_path": str(dataset_path),
            "num_nodes": dataset_config.num_nodes,
            "time_slots": dataset_config.time_slots,
            "output_dir": str(output_dir),
            "reference_parameters": reference_parameters,
            "model_parameters": model_parameters,
            "parameter_ratio": parameter_ratio,
            "parameter_difference": model_parameters - reference_parameters,
            "smoke_limited": any(
                value is not None
                for value in (
                    args.max_train_batches,
                    args.max_eval_batches,
                    args.max_test_batches,
                )
            ),
        }
    )
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2, default=str)

    print(f"Variant: {variant.name}")
    print(
        f"Architecture: {variant.num_layers} layers, d_model={variant.d_model}, "
        f"ffn_dim={variant.ffn_dim}, heads={variant.num_heads}"
    )
    print(
        f"Parameters: {model_parameters:,} vs reference "
        f"{reference_parameters:,} ({parameter_ratio:.4%})"
    )
    print(
        f"Batch sizes: train={args.batch_size}, eval={args.eval_batch_size}; "
        f"precision={args.precision}"
    )
    print(f"Output directory: {output_dir}")

    best_validation_mae = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    train_times = []
    validation_times = []
    run_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        trainer.model.train()
        dataloader["train_loader"].shuffle()
        train_values = []
        synchronize(device)
        start = time.perf_counter()
        for x, y in limited_batches(
            dataloader["train_loader"],
            args.max_train_batches,
        ):
            model_input, target = prepare_batch(x, y, device)
            train_values.append(trainer.train_batch(model_input, target))
        synchronize(device)
        train_seconds = time.perf_counter() - start
        if not train_values:
            raise RuntimeError("Training iterator produced no batches")
        train_times.append(train_seconds)
        train_metrics = aggregate_metrics(train_values)
        throughput = train_metrics["samples"] / train_seconds
        peak_memory_gib = (
            torch.cuda.max_memory_allocated(device) / (1024**3)
            if device.type == "cuda"
            else 0.0
        )

        validation_values = []
        synchronize(device)
        start = time.perf_counter()
        for x, y in limited_batches(
            dataloader["val_loader"],
            args.max_eval_batches,
        ):
            model_input, target = prepare_batch(x, y, device)
            validation_values.append(trainer.eval_batch(model_input, target))
        synchronize(device)
        validation_seconds = time.perf_counter() - start
        if not validation_values:
            raise RuntimeError("Validation iterator produced no batches")
        validation_times.append(validation_seconds)
        validation_metrics = aggregate_metrics(validation_values)
        current_lrate = trainer.step_lr_scheduler(validation_metrics["loss"])

        row = {
            "epoch": epoch,
            "lrate": current_lrate,
            "train_loss": train_metrics["loss"],
            "train_rmse": train_metrics["rmse"],
            "train_mape": train_metrics["mape"],
            "train_wmape": train_metrics["wmape"],
            "grad_norm_mean": train_metrics["grad_norm_mean"],
            "grad_norm_max": train_metrics["grad_norm_max"],
            "grad_clip_rate": train_metrics["grad_clip_rate"],
            "valid_loss": validation_metrics["loss"],
            "valid_rmse": validation_metrics["rmse"],
            "valid_mape": validation_metrics["mape"],
            "valid_wmape": validation_metrics["wmape"],
            "train_batches": len(train_values),
            "validation_batches": len(validation_values),
            "train_seconds": train_seconds,
            "validation_seconds": validation_seconds,
            "train_samples_per_second": throughput,
            "peak_memory_gib": peak_memory_gib,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "train.csv", index=False)

        print(
            f"Epoch {epoch:03d} | train MAE {train_metrics['loss']:.4f} | "
            f"valid MAE {validation_metrics['loss']:.4f} | "
            f"{train_seconds:.2f}s train, {validation_seconds:.2f}s valid | "
            f"{throughput:.1f} sample/s | {peak_memory_gib:.2f} GiB | "
            f"lr {current_lrate:.2g}"
        )

        if validation_metrics["loss"] < best_validation_mae:
            best_validation_mae = validation_metrics["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(trainer.model.state_dict(), checkpoint_path)
            print(
                f"Saved best checkpoint (validation MAE "
                f"{best_validation_mae:.4f})"
            )
        else:
            epochs_without_improvement += 1

        if (
            epoch >= args.min_epochs
            and epochs_without_improvement >= args.es_patience
        ):
            print(
                f"Early stopping after {epochs_without_improvement} "
                "epochs without improvement"
            )
            break

    trainer.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    synchronize(device)
    test_start = time.perf_counter()
    test_frame, test_average = evaluate_test_set(
        trainer,
        dataloader,
        device,
        args.output_len,
        args.max_test_batches,
    )
    synchronize(device)
    test_seconds = time.perf_counter() - test_start
    test_frame.to_csv(output_dir / "test.csv", index=False)

    summary = {
        "status": "complete",
        "variant": variant.name,
        "variant_spec": asdict(variant),
        "dataset": args.data,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_validation_mae": best_validation_mae,
        "test_average": test_average,
        "reference_parameters": reference_parameters,
        "model_parameters": model_parameters,
        "parameter_ratio": parameter_ratio,
        "parameter_difference": model_parameters - reference_parameters,
        "epochs_completed": len(history),
        "average_train_seconds": float(np.mean(train_times)),
        "average_validation_seconds": float(np.mean(validation_times)),
        "average_train_samples_per_second": float(
            np.mean([row["train_samples_per_second"] for row in history])
        ),
        "peak_memory_gib": float(
            max(row["peak_memory_gib"] for row in history)
        ),
        "test_seconds": test_seconds,
        "total_wall_seconds": time.perf_counter() - run_start,
        "smoke_limited": config["smoke_limited"],
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(
        f"Complete | best epoch {best_epoch} | test MAE "
        f"{test_average['mae']:.4f} | wall {summary['total_wall_seconds']:.1f}s"
    )


if __name__ == "__main__":
    main()
