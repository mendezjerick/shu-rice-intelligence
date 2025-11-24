"""Feature engineering utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass(slots=True)
class FeatureConfig:
    horizon: int = 1
    lags: Sequence[int] = (1, 2, 3, 6)
    rolling_windows: Sequence[int] = (3, 6)
    min_history: int = 6


def _add_temporal_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["month_sin"] = np.sin(2 * np.pi * frame["month"] / 12)
    frame["month_cos"] = np.cos(2 * np.pi * frame["month"] / 12)
    frame["time_index"] = np.arange(len(frame))
    return frame


def build_feature_table(
    df: pd.DataFrame,
    *,
    price_col: str = "avg_price",
    region_col: str = "admin1",
    config: FeatureConfig | None = None,
    drop_last_horizon: bool = True,
    drop_rows_with_na: bool = True,
) -> pd.DataFrame:
    """Turn the monthly panel data into a supervised learning table."""
    if config is None:
        config = FeatureConfig()
    results: list[pd.DataFrame] = []
    for region, group in df.groupby(region_col):
        group = group.sort_values("date").copy()
        value = group[price_col]
        for lag in config.lags:
            group[f"lag_{lag}"] = value.shift(lag)
        for window in config.rolling_windows:
            group[f"roll_mean_{window}"] = value.rolling(window).mean()
            group[f"roll_std_{window}"] = value.rolling(window).std()
        group["target"] = value.shift(-config.horizon)
        group = _add_temporal_columns(group)
        group[region_col] = region
        end_idx = len(group) - (config.horizon if drop_last_horizon and config.horizon else 0)
        group = group.iloc[config.min_history:end_idx]
        results.append(group)
    feature_df = pd.concat(results, ignore_index=True)
    if drop_rows_with_na:
        feature_df = feature_df.dropna().reset_index(drop=True)
    else:
        feature_df = feature_df.reset_index(drop=True)
    return feature_df


def select_feature_columns(
    df: pd.DataFrame,
    *,
    price_col: str = "avg_price",
    require_target: bool = True,
) -> tuple[pd.DataFrame, pd.Series | None]:
    cols_to_drop: Iterable[str] = [price_col, "date"]
    features = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    X = features.drop(columns=["target"]) if "target" in features.columns else features
    y = df["target"].copy() if require_target and "target" in df.columns else None
    return X, y
