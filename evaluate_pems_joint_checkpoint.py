#!/usr/bin/env python3
"""Evaluate a completed joint PEMS checkpoint without restarting training."""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
from pathlib import Path

import pandas as pd
import torch

import util
from dataset_config import get_dataset_config
from train_qk_joint import evaluate_test_set


def load_training_module(path: Path):
    loader = importlib.machinery.SourceFileLoader("joint_gradient_training", str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:
        raise ImportError(f"Cannot load training module from {path}")
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--training-source",
        type=Path,
        default=Path(__file__).with_name("train_pems_joint_gradient_conflict.py.orig"),
    )
    parser.add_argument(
        "--probe-conflicts",
        action="store_true",
        help="Also compute one post-hoc validation gradient-conflict probe.",
    )
    cli = parser.parse_args()

    run_dir = cli.run_dir.resolve()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    config["device"] = cli.device or config["device"]
    args = argparse.Namespace(**config)
    device = torch.device(args.device)

    training = load_training_module(cli.training_source.resolve())
    training.seed_everything(args.seed)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = not args.disable_tf32
        torch.backends.cudnn.allow_tf32 = not args.disable_tf32

    data = {}
    for dataset in training.DATASETS:
        dataset_config = get_dataset_config(dataset)
        data[dataset] = util.load_dataset(
            str(dataset_config.dataset_path),
            args.batch_size,
            args.eval_batch_size,
            args.eval_batch_size,
            expected_num_nodes=dataset_config.num_nodes,
            expected_input_len=args.input_len,
            expected_output_len=args.output_len,
            expected_input_dim=args.input_dim,
        )

    scalers = {dataset: data[dataset]["scaler"] for dataset in training.DATASETS}
    trainer = training.Trainer(args, scalers, device)
    checkpoint = run_dir / "best_model.pth"
    trainer.model.load_state_dict(torch.load(checkpoint, map_location=device))

    history = pd.read_csv(run_dir / "train.csv")
    best_row = history.loc[history["macro_valid_normalized_mae"].idxmin()]
    best_epoch = int(best_row["epoch"])
    results = {}
    for dataset in training.DATASETS:
        frame, average = evaluate_test_set(
            trainer, dataset, data[dataset], device, args.output_len
        )
        frame.to_csv(run_dir / f"test_{dataset}.csv", index=False)
        results[dataset] = average
        print(
            f"Test {dataset}: MAE {average['mae']:.4f}, "
            f"RMSE {average['rmse']:.4f}, MAPE {average['mape']:.4f}, "
            f"WMAPE {average['wmape']:.4f}",
            flush=True,
        )

    recovered = {
        "best_epoch": best_epoch,
        "best_macro_validation_normalized_mae": float(
            best_row["macro_valid_normalized_mae"]
        ),
        "test_average": results,
    }

    if cli.probe_conflicts:
        args.gradient_probe_batches = 1
        probes = training.make_probe_batches(data, args, device)
        conflicts = pd.DataFrame(
            training._gradient_conflicts_two_task(trainer, probes, best_epoch)
        )
        summary = training._summarize_conflicts_two_task(conflicts)
        conflicts.to_csv(run_dir / "gradient_conflicts_best_posthoc.csv", index=False)
        summary.to_csv(
            run_dir / "gradient_conflict_summary_best_posthoc.csv", index=False
        )
        strongest = summary.iloc[0].to_dict()
        recovered["posthoc_gradient_probe"] = {
            "validation_batches": 1,
            "strongest_conflict_by_mean_cosine": strongest,
        }
        print(
            f"Post-hoc conflict: {strongest['group']} "
            f"(cosine {strongest['mean_cosine']:.4f}, "
            f"negative {strongest['negative_cosine_rate']:.1%})",
            flush=True,
        )

    (run_dir / "recovered_evaluation.json").write_text(
        json.dumps(recovered, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
