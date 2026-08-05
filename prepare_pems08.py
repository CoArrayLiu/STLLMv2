"""Create leakage-free PEMS08 windows for the Transformer experiments.

The default split is performed on complete chronological days before windowing:
37 train days, 12 validation days, and 13 test days. Consequently, no raw
timestamp can occur in more than one split as either an input or a target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np


SPLIT_NAMES = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare leakage-free spatiotemporal Transformer windows"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/st_data/pems08/pems08.npz"),
    )
    parser.add_argument("--source_key", default="data")
    parser.add_argument("--output_dir", type=Path, default=Path("data/pems08"))
    parser.add_argument("--start_time", default="2016-07-01T00:00:00")
    parser.add_argument("--interval_minutes", type=int, default=5)
    parser.add_argument("--input_len", type=int, default=12)
    parser.add_argument("--output_len", type=int, default=12)
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument(
        "--split_unit",
        choices=("day", "timestep"),
        default="day",
        help="Snap boundaries to full days by default while preserving chronology",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing train/val/test files in the output directory",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def timestamp_text(start: datetime, index: int, interval_minutes: int) -> str:
    return (start + timedelta(minutes=index * interval_minutes)).isoformat()


def split_bounds(
    length: int,
    time_slots: int,
    train_ratio: float,
    val_ratio: float,
    split_unit: str,
) -> dict[str, tuple[int, int]]:
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1:
        raise ValueError("train_ratio and val_ratio must both be in (0, 1)")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1")

    if split_unit == "day":
        if length % time_slots != 0:
            raise ValueError(
                "Day-aligned splitting requires a whole number of complete days: "
                f"length={length}, time_slots={time_slots}"
            )
        total_units = length // time_slots
        train_units = int(total_units * train_ratio)
        val_units = int(total_units * val_ratio)
        train_end = train_units * time_slots
        val_end = train_end + val_units * time_slots
    else:
        train_end = int(length * train_ratio)
        val_end = train_end + int(length * val_ratio)

    bounds = {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, length),
    }
    if any(end <= start for start, end in bounds.values()):
        raise ValueError(f"Empty chronological split produced: {bounds}")
    return bounds


def metric_summary(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    mask = target > 0
    if not np.any(mask):
        raise ValueError("Cannot compute masked metrics: target has no positive values")
    pred = prediction[mask].astype(np.float64)
    true = target[mask].astype(np.float64)
    error = pred - true
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mape": float(np.mean(np.abs(error) / true)),
        "wmape": float(np.sum(np.abs(error)) / np.sum(np.abs(true))),
    }


def baseline_summary(
    x: np.ndarray,
    y: np.ndarray,
    train_node_mean: np.ndarray,
) -> dict[str, Any]:
    persistence = np.broadcast_to(x[:, -1:, :, :1], y.shape)
    historical_mean = np.broadcast_to(
        train_node_mean.reshape(1, 1, -1, 1), y.shape
    )
    result: dict[str, Any] = {}
    for name, prediction in (
        ("last_value", persistence),
        ("train_node_mean", historical_mean),
    ):
        horizons = [
            metric_summary(prediction[:, horizon], y[:, horizon])
            for horizon in range(y.shape[1])
        ]
        result[name] = {
            "average": {
                key: float(np.mean([row[key] for row in horizons]))
                for key in horizons[0]
            },
            "horizons": horizons,
        }
    return result


def build_split(
    values: np.ndarray,
    time_of_day: np.ndarray,
    day_of_week: np.ndarray,
    global_start: int,
    input_len: int,
    output_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_samples = len(values) - input_len - output_len + 1
    if num_samples <= 0:
        raise ValueError(
            f"Split of length {len(values)} is too short for "
            f"input_len={input_len}, output_len={output_len}"
        )
    num_nodes = values.shape[1]
    x = np.empty((num_samples, input_len, num_nodes, 3), dtype=np.float32)
    y = np.empty((num_samples, output_len, num_nodes, 1), dtype=np.float32)

    for offset in range(input_len):
        x[:, offset, :, 0] = values[offset : offset + num_samples]
        x[:, offset, :, 1] = time_of_day[offset : offset + num_samples, None]
        x[:, offset, :, 2] = day_of_week[offset : offset + num_samples, None]
    for offset in range(output_len):
        begin = input_len + offset
        y[:, offset, :, 0] = values[begin : begin + num_samples]

    sample_start = global_start + np.arange(num_samples, dtype=np.int64)
    return x, y, sample_start


def validate_windows(
    x: np.ndarray,
    y: np.ndarray,
    sample_start: np.ndarray,
    raw_start: int,
    raw_end: int,
    input_len: int,
    output_len: int,
    num_nodes: int,
    time_slots: int,
) -> None:
    expected_x = (len(sample_start), input_len, num_nodes, 3)
    expected_y = (len(sample_start), output_len, num_nodes, 1)
    if x.shape != expected_x or y.shape != expected_y:
        raise ValueError(
            f"Window shape mismatch: expected x={expected_x}, y={expected_y}; "
            f"got x={x.shape}, y={y.shape}"
        )
    if x.dtype != np.float32 or y.dtype != np.float32:
        raise ValueError(f"Expected float32 arrays, got x={x.dtype}, y={y.dtype}")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Generated windows contain NaN or infinite values")
    if x[..., 1].min() < 0 or x[..., 1].max() >= 1:
        raise ValueError("time-of-day feature must be in [0, 1)")
    if not np.allclose(x[..., 1] * time_slots, np.round(x[..., 1] * time_slots)):
        raise ValueError("time-of-day feature is not aligned to discrete slots")
    if x[..., 2].min() < 0 or x[..., 2].max() > 6:
        raise ValueError("day-of-week feature must be in [0, 6]")
    if not np.array_equal(x[..., 2], np.round(x[..., 2])):
        raise ValueError("day-of-week feature must contain integer categories")
    if sample_start[0] < raw_start:
        raise ValueError("First window begins before its chronological split")
    final_target = sample_start[-1] + input_len + output_len - 1
    if final_target >= raw_end:
        raise ValueError("Final target extends beyond its chronological split")


def write_npz(
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    sample_start: np.ndarray,
    input_len: int,
    output_len: int,
) -> None:
    np.savez_compressed(
        path,
        x=x,
        y=y,
        x_offsets=np.arange(-(input_len - 1), 1, dtype=np.int64).reshape(-1, 1),
        y_offsets=np.arange(1, output_len + 1, dtype=np.int64).reshape(-1, 1),
        sample_start=sample_start,
        target_start=sample_start + input_len,
        target_end=sample_start + input_len + output_len - 1,
    )


def main() -> None:
    args = parse_args()
    if args.interval_minutes <= 0 or 1440 % args.interval_minutes != 0:
        raise ValueError("interval_minutes must be a positive divisor of 1440")
    if args.input_len <= 0 or args.output_len <= 0:
        raise ValueError("input_len and output_len must be positive")
    if not args.source.is_file():
        raise FileNotFoundError(f"Source does not exist: {args.source}")

    output_files = [args.output_dir / f"{name}.npz" for name in SPLIT_NAMES]
    output_files.extend(
        [args.output_dir / "manifest.json", args.output_dir / "baselines.json"]
    )
    existing = [path for path in output_files if path.exists()]
    if existing and not args.overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite existing prepared data: {joined}; "
            "pass --overwrite to replace it"
        )

    if args.source.suffix.lower() == ".npz":
        with np.load(args.source) as archive:
            if args.source_key not in archive:
                raise KeyError(
                    f"Source archive has no {args.source_key!r} key: {archive.files}"
                )
            raw = np.asarray(archive[args.source_key])
    elif args.source.suffix.lower() in {".h5", ".hdf5"}:
        import h5py

        with h5py.File(args.source, "r") as handle:
            if args.source_key not in handle:
                raise KeyError(f"HDF5 source has no {args.source_key!r} dataset")
            raw = np.asarray(handle[args.source_key])
    else:
        raise ValueError(f"Unsupported source format: {args.source.suffix}")
    if raw.ndim == 2:
        values = np.asarray(raw, dtype=np.float32)
    elif raw.ndim == 3 and raw.shape[-1] >= 1:
        values = np.asarray(raw[..., 0], dtype=np.float32)
    else:
        raise ValueError(
            "Expected source shape [time, nodes] or [time, nodes, features], "
            f"got {raw.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("Source contains NaN or infinite values")
    if np.any(values < 0):
        raise ValueError("Source values must be non-negative")

    start_time = datetime.fromisoformat(args.start_time)
    time_slots = 1440 // args.interval_minutes
    indices = np.arange(len(values), dtype=np.int64)
    time_of_day = (indices % time_slots).astype(np.float32) / time_slots
    day_of_week = (
        start_time.weekday() + indices // time_slots
    ).astype(np.int64) % 7
    bounds = split_bounds(
        length=len(values),
        time_slots=time_slots,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        split_unit=args.split_unit,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_start, train_end = bounds["train"]
    train_node_mean = values[train_start:train_end].mean(axis=0, dtype=np.float64)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "source": str(args.source),
        "source_sha256": sha256_file(args.source),
        "source_shape": list(raw.shape),
        "source_dtype": str(raw.dtype),
        "start_time": start_time.isoformat(),
        "interval_minutes": args.interval_minutes,
        "time_slots": time_slots,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "num_nodes": int(values.shape[1]),
        "split_unit": args.split_unit,
        "requested_ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": 1 - args.train_ratio - args.val_ratio,
        },
        "splits": {},
        "leakage_checks": {
            "split_raw_ranges_disjoint": True,
            "windows_built_after_split": True,
            "input_or_target_timestamp_shared_across_splits": False,
        },
    }
    baselines: dict[str, Any] = {
        "mask": "target > 0, matching util.metric",
        "splits": {},
    }
    scaler_mean: float | None = None
    scaler_std: float | None = None

    previous_end = 0
    for split_name in SPLIT_NAMES:
        raw_start, raw_end = bounds[split_name]
        if raw_start != previous_end:
            raise ValueError(f"Non-contiguous split boundaries: {bounds}")
        previous_end = raw_end
        x, y, sample_start = build_split(
            values=values[raw_start:raw_end],
            time_of_day=time_of_day[raw_start:raw_end],
            day_of_week=day_of_week[raw_start:raw_end],
            global_start=raw_start,
            input_len=args.input_len,
            output_len=args.output_len,
        )
        validate_windows(
            x=x,
            y=y,
            sample_start=sample_start,
            raw_start=raw_start,
            raw_end=raw_end,
            input_len=args.input_len,
            output_len=args.output_len,
            num_nodes=values.shape[1],
            time_slots=time_slots,
        )
        if split_name == "train":
            scaler_mean = float(x[..., 0].mean(dtype=np.float64))
            scaler_std = float(x[..., 0].std(dtype=np.float64))
            if not math.isfinite(scaler_std) or scaler_std <= 0:
                raise ValueError(f"Invalid training standard deviation: {scaler_std}")
        assert scaler_mean is not None and scaler_std is not None
        restored = ((x[..., 0] - scaler_mean) / scaler_std) * scaler_std + scaler_mean
        roundtrip_max_error = float(np.max(np.abs(restored - x[..., 0])))

        output_path = args.output_dir / f"{split_name}.npz"
        write_npz(
            path=output_path,
            x=x,
            y=y,
            sample_start=sample_start,
            input_len=args.input_len,
            output_len=args.output_len,
        )
        manifest["splits"][split_name] = {
            "path": str(output_path),
            "raw_start_index": raw_start,
            "raw_end_index_exclusive": raw_end,
            "raw_start_time": timestamp_text(
                start_time, raw_start, args.interval_minutes
            ),
            "raw_end_time_inclusive": timestamp_text(
                start_time, raw_end - 1, args.interval_minutes
            ),
            "raw_timesteps": raw_end - raw_start,
            "samples": len(sample_start),
            "x_shape": list(x.shape),
            "y_shape": list(y.shape),
            "first_target_index": int(sample_start[0] + args.input_len),
            "last_target_index": int(
                sample_start[-1] + args.input_len + args.output_len - 1
            ),
            "scaler_roundtrip_max_abs_error": roundtrip_max_error,
        }
        baselines["splits"][split_name] = baseline_summary(
            x=x,
            y=y,
            train_node_mean=train_node_mean,
        )
        print(
            f"{split_name}: raw=[{raw_start}, {raw_end}), samples={len(sample_start)}, "
            f"x={x.shape}, y={y.shape}, roundtrip_error={roundtrip_max_error:.3g}"
        )
        del x, y, sample_start, restored

    if previous_end != len(values):
        raise ValueError(f"Splits do not cover the complete source: {bounds}")
    manifest["scaler"] = {"mean": scaler_mean, "std": scaler_std}
    manifest["actual_ratios"] = {
        name: (end - start) / len(values) for name, (start, end) in bounds.items()
    }
    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    with (args.output_dir / "baselines.json").open("w", encoding="utf-8") as file:
        json.dump(baselines, file, ensure_ascii=False, indent=2)
    print(f"Wrote manifest and baselines to {args.output_dir}")


if __name__ == "__main__":
    main()
