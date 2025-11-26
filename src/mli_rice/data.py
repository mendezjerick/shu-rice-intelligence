"""Data loading helpers for the Rice Price forecasting project."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = ROOT_DIR / "rice.csv"


def _coerce_region_value(value: str) -> str:
    return value.strip()


def load_rice_data(path: str | Path | None = None, *, columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Load the rice dataset and apply a minimal schema."""
    csv_path = Path(path) if path is not None else DEFAULT_DATA_PATH
    if not csv_path.exists():
        raise FileNotFoundError(f"Cannot locate rice dataset at {csv_path}")
    df = pd.read_csv(csv_path)
    required = {"region", "year", "month", "price"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataset: {sorted(missing)}")
    df["region"] = df["region"].astype(str).map(_coerce_region_value)
    if "province" in df.columns:
        df["province"] = df["province"].astype(str).map(_coerce_region_value)
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))
    df = df.sort_values(["region", "date"]).reset_index(drop=True)
    if columns is not None:
        df = df[list(columns)]
    return df


def national_monthly_average(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the dataset to a national-level monthly average price."""
    if "date" not in df.columns:
        raise ValueError("DataFrame must contain a 'date' column; run load_rice_data first")
    grouped = (
        df.groupby("date", as_index=False)["price"].mean().rename(columns={"price": "national_price"})
    )
    grouped["year"] = grouped["date"].dt.year
    grouped["month"] = grouped["date"].dt.month
    return grouped


def region_monthly_average(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the average monthly price per region."""
    if "date" not in df.columns:
        raise ValueError("DataFrame must contain a 'date' column; run load_rice_data first")
    grouped = (
        df.groupby(["region", "date"], as_index=False)["price"].mean().rename(columns={"price": "avg_price"})
    )
    grouped["year"] = grouped["date"].dt.year
    grouped["month"] = grouped["date"].dt.month
    return grouped
