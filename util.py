import os
import pickle
from pathlib import Path

import numpy as np
import torch

def load_graph_data(graph_filename):
    """Load a dense graph from a supported NumPy or pickle container."""

    graph_path = Path(graph_filename)
    suffix = graph_path.suffix.lower()
    if suffix == ".npy":
        return np.load(graph_path)
    if suffix in {".pkl", ".pickle"}:
        return load_pickle(graph_path)
    raise ValueError(
        f"Unsupported graph format {suffix!r} for {graph_path}; "
        "expected .npy, .pkl, or .pickle"
    )

def load_pickle(pickle_file):

    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int((self.size + self.batch_size - 1) // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind:end_ind, ...]
                y_i = self.ys[start_ind:end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def load_dataset(
    dataset_dir,
    batch_size,
    valid_batch_size=None,
    test_batch_size=None,
    expected_num_nodes=None,
    expected_input_len=None,
    expected_output_len=None,
    expected_input_dim=None,
):
    data = {}
    for category in ["train", "val", "test"]:
        dataset_path = os.path.join(dataset_dir, category + ".npz")
        if not os.path.isfile(dataset_path):
            raise FileNotFoundError(f"Dataset split does not exist: {dataset_path}")
        with np.load(dataset_path) as cat_data:
            missing = {"x", "y"} - set(cat_data.files)
            if missing:
                raise KeyError(
                    f"Dataset split {dataset_path} is missing keys: {sorted(missing)}"
                )
            x = cat_data["x"]
            y = cat_data["y"]
        if x.ndim != 4 or y.ndim != 4:
            raise ValueError(
                f"{category} arrays must be rank 4; got x={x.shape}, y={y.shape}"
            )
        if x.shape[0] == 0 or x.shape[0] != y.shape[0]:
            raise ValueError(
                f"{category} sample count mismatch or empty arrays: "
                f"x={x.shape}, y={y.shape}"
            )
        checks = (
            ("nodes", x.shape[2], expected_num_nodes),
            ("target nodes", y.shape[2], expected_num_nodes),
            ("input length", x.shape[1], expected_input_len),
            ("output length", y.shape[1], expected_output_len),
            ("input features", x.shape[3], expected_input_dim),
        )
        for label, actual, expected in checks:
            if expected is not None and actual != expected:
                raise ValueError(
                    f"{category} {label} mismatch: expected {expected}, got {actual}"
                )
        if y.shape[3] < 1:
            raise ValueError(f"{category} targets require at least one feature")
        if not np.issubdtype(x.dtype, np.floating) or not np.issubdtype(
            y.dtype, np.floating
        ):
            raise ValueError(
                f"{category} arrays must be floating; got x={x.dtype}, y={y.dtype}"
            )
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f"{category} arrays contain NaN or infinite values")
        if expected_input_dim is not None and expected_input_dim >= 3:
            time_of_day = x[..., 1]
            day_of_week = x[..., 2]
            if time_of_day.min() < 0 or time_of_day.max() >= 1:
                raise ValueError(
                    f"{category} time-of-day values must be in [0, 1)"
                )
            if day_of_week.min() < 0 or day_of_week.max() > 6:
                raise ValueError(
                    f"{category} day-of-week values must be in [0, 6]"
                )
            if not np.array_equal(day_of_week, np.round(day_of_week)):
                raise ValueError(
                    f"{category} day-of-week values must be integer categories"
                )
        data["x_" + category] = x
        data["y_" + category] = y
    scaler = StandardScaler(
        mean=data["x_train"][..., 0].mean(), std=data["x_train"][..., 0].std()
    )
    # Data format
    if not np.isfinite(scaler.mean) or not np.isfinite(scaler.std) or scaler.std <= 0:
        raise ValueError(
            f"Invalid training scaler: mean={scaler.mean}, std={scaler.std}"
        )
    for category in ["train", "val", "test"]:
        data["x_" + category][..., 0] = scaler.transform(data["x_" + category][..., 0])

    print("Perform shuffle on the dataset")
    random_train = torch.arange(int(data["x_train"].shape[0]))
    random_train = torch.randperm(random_train.size(0))
    data["x_train"] = data["x_train"][random_train, ...]
    data["y_train"] = data["y_train"][random_train, ...]

    random_val = torch.arange(int(data["x_val"].shape[0]))
    random_val = torch.randperm(random_val.size(0))
    data["x_val"] = data["x_val"][random_val, ...]
    data["y_val"] = data["y_val"][random_val, ...]

    # random_test = torch.arange(int(data['x_test'].shape[0]))
    # random_test = torch.randperm(random_test.size(0))
    # data['x_test'] =  data['x_test'][random_test,...]
    # data['y_test'] =  data['y_test'][random_test,...]

    data["train_loader"] = DataLoader(data["x_train"], data["y_train"], batch_size)
    data["val_loader"] = DataLoader(
        data["x_val"], data["y_val"], valid_batch_size, pad_with_last_sample=False
    )
    data["test_loader"] = DataLoader(
        data["x_test"], data["y_test"], test_batch_size, pad_with_last_sample=False
    )
    data["scaler"] = scaler

    return data


def MAE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(true - pred))


def MAPE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(torch.div((true - pred), true)))


def RMSE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.sqrt(torch.mean((pred - true) ** 2))


def WMAPE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    loss = torch.sum(torch.abs(pred - true)) / torch.sum(torch.abs(true))
    return loss

def metric(pred, real):
    mae = MAE_torch(pred, real, 0).item()
    mape = MAPE_torch(pred, real,0).item()
    wmape = WMAPE_torch(pred, real, 0).item()
    rmse = RMSE_torch(pred, real, 0).item()
    return mae, mape, rmse, wmape
