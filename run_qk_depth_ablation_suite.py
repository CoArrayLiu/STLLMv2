#!/usr/bin/env python3
"""Run the six QK depth-ablation configurations sequentially.

Each configuration is an experiment design.  By default it is trained and
tested independently on all four Taxi/Bike datasets before the runner advances
to the next configuration.  All child processes use one GPU sequentially.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from train_qk_depth_ablation import (
    RECOMMENDED_EVAL_BATCH_SIZE,
    RECOMMENDED_TRAIN_BATCH_SIZE,
    VARIANTS,
)


DEFAULT_DATASETS = ("taxi_drop", "taxi_pick", "bike_drop", "bike_pick")
DEFAULT_VARIANTS = tuple(VARIANTS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DEFAULT_DATASETS,
        default=list(DEFAULT_DATASETS),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=DEFAULT_VARIANTS,
        default=list(DEFAULT_VARIANTS),
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("logs/qk_depth_ablation_seed6666"),
    )
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
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--skip_completed", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="One epoch and one train/validation/test batch per child run.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "batch_size",
        "eval_batch_size",
        "epochs",
        "es_patience",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name} must be positive")
    if args.min_epochs < 0:
        raise ValueError("--min_epochs cannot be negative")
    if len(set(args.datasets)) != len(args.datasets):
        raise ValueError("--datasets contains duplicates")
    if len(set(args.variants)) != len(args.variants):
        raise ValueError("--variants contains duplicates")


def child_command(
    args: argparse.Namespace,
    variant: str,
    dataset: str,
    output_dir: Path,
) -> list[str]:
    smoke = args.smoke
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).with_name("train_qk_depth_ablation.py")),
        "--variant",
        variant,
        "--data",
        dataset,
        "--save_dir",
        str(output_dir),
        "--device",
        args.device,
        "--precision",
        args.precision,
        "--batch_size",
        str(8 if smoke else args.batch_size),
        "--eval_batch_size",
        str(8 if smoke else args.eval_batch_size),
        "--epochs",
        str(1 if smoke else args.epochs),
        "--min_epochs",
        str(0 if smoke else args.min_epochs),
        "--es_patience",
        str(1 if smoke else args.es_patience),
        "--lrate",
        str(args.lrate),
        "--wdecay",
        str(args.wdecay),
        "--grad_clip",
        str(args.grad_clip),
        "--loss_space",
        args.loss_space,
        "--lr_scheduler",
        args.lr_scheduler,
        "--lr_patience",
        str(args.lr_patience),
        "--lr_factor",
        str(args.lr_factor),
        "--min_lrate",
        str(args.min_lrate),
        "--seed",
        str(args.seed),
    ]
    if args.disable_tf32:
        command.append("--disable_tf32")
    if smoke:
        command.extend(
            [
                "--max_train_batches",
                "1",
                "--max_eval_batches",
                "1",
                "--max_test_batches",
                "1",
            ]
        )
    return command


def read_summary(path: Path) -> dict:
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("status") != "complete":
        raise ValueError(f"Incomplete summary: {path}")
    return summary


def result_row(summary: dict) -> dict:
    test = summary["test_average"]
    spec = summary["variant_spec"]
    return {
        "variant": summary["variant"],
        "removed_layers": spec["removed_layers"],
        "parameter_mode": spec["parameter_mode"],
        "num_layers": spec["num_layers"],
        "d_model": spec["d_model"],
        "ffn_dim": spec["ffn_dim"],
        "dataset": summary["dataset"],
        "seed": summary["seed"],
        "parameters": summary["model_parameters"],
        "parameter_ratio": summary["parameter_ratio"],
        "best_epoch": summary["best_epoch"],
        "best_validation_mae": summary["best_validation_mae"],
        "test_mae": test["mae"],
        "test_rmse": test["rmse"],
        "test_mape": test["mape"],
        "test_wmape": test["wmape"],
        "train_samples_per_second": summary[
            "average_train_samples_per_second"
        ],
        "peak_memory_gib": summary["peak_memory_gib"],
        "total_wall_seconds": summary["total_wall_seconds"],
        "smoke_limited": summary["smoke_limited"],
    }


def write_results(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    validate_args(args)
    jobs = [
        (variant, dataset)
        for variant in args.variants
        for dataset in args.datasets
    ]
    commands = []
    for variant, dataset in jobs:
        output_dir = (
            args.output_root / variant / dataset / f"seed_{args.seed}"
        )
        commands.append(
            {
                "variant": variant,
                "dataset": dataset,
                "output_dir": str(output_dir),
                "command": child_command(
                    args,
                    variant,
                    dataset,
                    output_dir,
                ),
            }
        )

    print(
        f"Six-design QK depth suite: {len(args.variants)} variants × "
        f"{len(args.datasets)} datasets = {len(jobs)} sequential runs"
    )
    for index, item in enumerate(commands, start=1):
        print(
            f"[{index:02d}/{len(commands):02d}] "
            f"{item['variant']} / {item['dataset']}"
        )
        print("  " + shlex.join(item["command"]))
    if args.dry_run:
        return

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "running",
        "datasets": args.datasets,
        "variants": args.variants,
        "seed": args.seed,
        "smoke": args.smoke,
        "commands": [
            {**item, "command": shlex.join(item["command"])}
            for item in commands
        ],
    }
    manifest_path = args.output_root / "suite_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    rows = []
    for index, item in enumerate(commands, start=1):
        output_dir = Path(item["output_dir"])
        summary_path = output_dir / "summary.json"
        print(
            f"\n=== [{index:02d}/{len(commands):02d}] "
            f"{item['variant']} / {item['dataset']} ===",
            flush=True,
        )
        if summary_path.is_file() and args.skip_completed:
            print(f"Skipping completed run: {summary_path}", flush=True)
        else:
            if output_dir.exists() and any(output_dir.iterdir()):
                raise FileExistsError(
                    f"Refusing to overwrite partial run: {output_dir}"
                )
            subprocess.run(
                item["command"],
                cwd=Path(__file__).resolve().parent,
                env=environment,
                check=True,
            )
        rows.append(result_row(read_summary(summary_path)))
        write_results(args.output_root / "suite_results.csv", rows)

    manifest["status"] = "complete"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"\nAll runs complete. Results: "
        f"{args.output_root / 'suite_results.csv'}"
    )


if __name__ == "__main__":
    main()
