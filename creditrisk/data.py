"""Dataset definitions, duplicate-safe splits, and training-only transformations."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, StandardScaler

TARGET = "default.payment.next.month"
CAT = ["SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]


@dataclass(frozen=True)
class Dataset:
    name: str
    frame: pd.DataFrame
    categorical: tuple[str, ...]
    source: str
    path: str

    @property
    def features(self):
        return self.frame.drop(columns=TARGET)


def load_dataset():
    """Load only the original credit-card data used by this project."""
    path = "UCI_Credit_Card.csv"
    frame = pd.read_csv(path).drop(columns="ID")
    categorical = tuple(CAT)
    validate_frame(frame, categorical)
    return Dataset(
        "credit_card",
        frame,
        categorical,
        "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients",
        path,
    )


def validate_frame(frame, categorical):
    if frame.columns.duplicated().any() or frame.isna().any().any():
        raise ValueError("Duplicate column names or missing data")
    numeric = frame.select_dtypes(include="number")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Nonfinite numeric values")
    if TARGET not in frame or set(frame[TARGET].unique()) != {0, 1}:
        raise ValueError("Expected a binary target coded 0/1")
    if not set(categorical) <= set(frame.columns) - {TARGET}:
        raise ValueError("Invalid categorical schema")


def fingerprint(frame):
    return hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()


def feature_groups(frame):
    return pd.util.hash_pandas_object(
        frame.drop(columns=[TARGET, "ID"], errors="ignore"), index=False
    ).to_numpy()


def partitions(frame, seed):
    """Approximately 60/20/20; identical features cannot cross partitions."""
    groups = feature_groups(frame)
    x = frame.drop(columns=[TARGET, "ID"], errors="ignore")
    outer = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    development, test = next(outer.split(x, frame[TARGET], groups))
    inner = StratifiedGroupKFold(4, shuffle=True, random_state=seed + 1)
    train, validation = next(
        inner.split(x.iloc[development], frame[TARGET].iloc[development], groups[development])
    )
    indices = [development[train], development[validation], test]
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        if set(groups[indices[i]]) & set(groups[indices[j]]):
            raise RuntimeError("Duplicate feature groups crossed partitions")
    return [frame.iloc[idx].reset_index(drop=True) for idx in indices], indices


def classifier_preprocessor(x, categorical=None):
    categorical = [c for c in (CAT if categorical is None else categorical) if c in x]
    numeric = [c for c in x if c not in categorical]
    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ]
    )


class Schema:
    """A fitted numeric quantile transform and exact categorical vocabulary."""

    def fit(self, x, seed=42, categorical=None):
        self.columns = list(x.columns)
        self.cat = [c for c in (CAT if categorical is None else categorical) if c in x]
        self.num = [c for c in x if c not in self.cat]
        if not self.num or not self.cat:
            raise ValueError("Mixed diffusion requires numeric and categorical features")
        self.values = [np.sort(x[c].unique()) for c in self.cat]
        self.cards = [len(values) for values in self.values]
        self.integer_numeric = [c for c in self.num if np.equal(x[c], np.rint(x[c])).all()]
        self.qt = QuantileTransformer(
            n_quantiles=min(1000, len(x)), output_distribution="normal", random_state=seed
        )
        self.qt.fit(x[self.num].to_numpy())
        self.bounds = (x[self.num].min().to_numpy(), x[self.num].max().to_numpy())
        return self

    def encode(self, x):
        numbers = self.qt.transform(x[self.num].to_numpy()).astype(np.float32)
        categories = np.stack(
            [
                pd.Categorical(x[c], categories=values).codes
                for c, values in zip(self.cat, self.values, strict=False)
            ],
            axis=1,
        )
        if (categories < 0).any():
            raise ValueError("Category absent from fitted training vocabulary")
        return numbers, categories.astype(np.int64)

    def decode(self, numbers, categories):
        if not np.isfinite(numbers).all():
            raise ValueError("Nonfinite diffusion samples")
        numbers = np.clip(self.qt.inverse_transform(numbers), *self.bounds)
        result = pd.DataFrame(numbers, columns=self.num)
        for column in self.integer_numeric:
            result[column] = np.rint(result[column])
        for index, (column, values) in enumerate(zip(self.cat, self.values, strict=False)):
            result[column] = values[categories[:, index]]
        return result[self.columns]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
