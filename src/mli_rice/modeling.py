"""Model training and forecasting utilities."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import FeatureConfig, build_feature_table, select_feature_columns


@dataclass(slots=True)
class TrainingConfig:
    random_state: int = 42
    n_splits: int = 5
    holdout_months: int = 12
    artifacts_dir: Path = Path("artifacts")
    price_col: str = "avg_price"
    region_col: str = "admin1"

    @property
    def artifact_path(self) -> Path:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        return self.artifacts_dir / "best_model.joblib"

    @property
    def metrics_path(self) -> Path:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        return self.artifacts_dir / "metrics.json"


def _infer_column_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical = [col for col in X.columns if X[col].dtype == "object"]
    numeric = [col for col in X.columns if col not in categorical]
    return categorical, numeric


def _build_preprocessor(categorical: Iterable[str], numeric: Iterable[str]) -> ColumnTransformer:
    numeric = list(numeric)
    categorical = list(categorical)
    transformers = []
    if numeric:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def _candidate_estimators(random_state: int) -> Dict[str, object]:
    return {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_depth=5,
            learning_rate=0.05,
            max_iter=400,
            random_state=random_state,
        ),
    }


def _build_pipeline(estimator, categorical: Iterable[str], numeric: Iterable[str]) -> Pipeline:
    return Pipeline([
        ("preprocess", _build_preprocessor(categorical, numeric)),
        ("model", estimator),
    ])


def _time_series_cv(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int,
) -> dict:
    splitter = TimeSeriesSplit(n_splits=n_splits)
    rmses: list[float] = []
    r2_scores: list[float] = []
    for train_idx, val_idx in splitter.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        fold_model = clone(pipeline)
        fold_model.fit(X_train, y_train)
        preds = fold_model.predict(X_val)
        rmses.append(float(np.sqrt(mean_squared_error(y_val, preds))))
        r2_scores.append(r2_score(y_val, preds))
    return {"rmse": float(np.mean(rmses)), "r2": float(np.mean(r2_scores))}


def train_regressors(
    feature_table: pd.DataFrame,
    *,
    config: TrainingConfig | None = None,
    feature_config: FeatureConfig | None = None,
) -> dict:
    if config is None:
        config = TrainingConfig()
    if feature_config is None:
        feature_config = FeatureConfig()

    feature_table = feature_table.sort_values(["date", config.region_col]).reset_index(drop=True)
    cutoff_date = feature_table["date"].max() - pd.DateOffset(months=config.holdout_months - 1)
    holdout_mask = feature_table["date"] >= cutoff_date
    train_df = feature_table.loc[~holdout_mask]
    holdout_df = feature_table.loc[holdout_mask]

    X_train, y_train = select_feature_columns(train_df, price_col=config.price_col)
    X_holdout, y_holdout = select_feature_columns(holdout_df, price_col=config.price_col)

    categorical, numeric = _infer_column_types(X_train)
    estimators = _candidate_estimators(config.random_state)
    metrics: dict[str, dict] = {}
    fitted_models: dict[str, Pipeline] = {}

    for name, estimator in estimators.items():
        pipeline = _build_pipeline(estimator, categorical, numeric)
        cv_scores = _time_series_cv(pipeline, X_train, y_train, config.n_splits)
        pipeline.fit(X_train, y_train)
        holdout_preds = pipeline.predict(X_holdout)
        holdout_rmse = float(np.sqrt(mean_squared_error(y_holdout, holdout_preds)))
        holdout_r2 = r2_score(y_holdout, holdout_preds)
        metrics[name] = {
            "cv_rmse": cv_scores["rmse"],
            "cv_r2": cv_scores["r2"],
            "holdout_rmse": float(holdout_rmse),
            "holdout_r2": float(holdout_r2),
        }
        fitted_models[name] = pipeline

    best_name = min(metrics, key=lambda name: metrics[name]["holdout_rmse"])
    best_pipeline = _build_pipeline(estimators[best_name], categorical, numeric)
    full_X = pd.concat([X_train, X_holdout], ignore_index=True)
    full_y = pd.concat([y_train, y_holdout], ignore_index=True)
    best_pipeline.fit(full_X, full_y)

    joblib.dump(best_pipeline, config.artifact_path)
    with config.metrics_path.open("w", encoding="utf-8") as fp:
        json.dump({"models": metrics, "selected_model": best_name}, fp, indent=2)

    return {
        "metrics": metrics,
        "selected_model": best_name,
        "pipeline": best_pipeline,
        "holdout_predictions": {
            "actual": y_holdout.reset_index(drop=True).to_list(),
            "predicted": fitted_models[best_name].predict(X_holdout).tolist(),
            "dates": holdout_df["date"].dt.strftime("%Y-%m").to_list(),
            "regions": holdout_df[config.region_col].to_list(),
        },
    }


def prepare_inference_frame(
    region_df: pd.DataFrame,
    *,
    feature_config: FeatureConfig | None = None,
) -> pd.DataFrame:
    feature_config = feature_config or FeatureConfig()
    inference_table = build_feature_table(
        region_df,
        config=feature_config,
        drop_last_horizon=False,
        drop_rows_with_na=False,
    )
    latest_rows = inference_table.sort_values(["admin1", "date"]).groupby("admin1").tail(1)
    return latest_rows.reset_index(drop=True)


def forecast_next_month(
    region_df: pd.DataFrame,
    pipeline: Pipeline,
    *,
    config: TrainingConfig | None = None,
    feature_config: FeatureConfig | None = None,
) -> pd.DataFrame:
    config = config or TrainingConfig()
    feature_config = feature_config or FeatureConfig()
    inference_rows = prepare_inference_frame(region_df, feature_config=feature_config)
    X_latest, _ = select_feature_columns(inference_rows, price_col=config.price_col, require_target=False)
    predictions = pipeline.predict(X_latest)
    latest_actual = (
        region_df.sort_values([config.region_col, "date"]).groupby(config.region_col).tail(1)[
            [config.region_col, config.price_col, "date"]
        ].rename(columns={"date": "current_date"})
    )
    merged = inference_rows[[config.region_col, "date"]].rename(columns={"date": "feature_date"})
    merged["forecast_price"] = predictions
    merged = merged.merge(latest_actual, on=config.region_col)
    merged["forecast_date"] = merged["feature_date"] + pd.DateOffset(months=feature_config.horizon)
    merged["price_change"] = merged["forecast_price"] - merged[config.price_col]
    merged["pct_change"] = merged["price_change"] / merged[config.price_col]
    return merged[
        [
            config.region_col,
            "current_date",
            "forecast_date",
            config.price_col,
            "forecast_price",
            "price_change",
            "pct_change",
        ]
    ]


def _append_predictions_to_history(
    history: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    config: TrainingConfig,
) -> pd.DataFrame:
    additions = predictions[[config.region_col, "forecast_date", "forecast_price"]].rename(
        columns={
            config.region_col: "admin1",
            "forecast_date": "date",
            "forecast_price": config.price_col,
        }
    )
    additions["year"] = additions["date"].dt.year
    additions["month"] = additions["date"].dt.month
    updated = (
        pd.concat([history, additions], ignore_index=True)
        .sort_values(["admin1", "date"])
        .reset_index(drop=True)
    )
    return updated


def multi_step_forecast(
    region_df: pd.DataFrame,
    pipeline: Pipeline,
    *,
    months: int,
    config: TrainingConfig | None = None,
    feature_config: FeatureConfig | None = None,
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    if months < 1:
        raise ValueError("months must be >= 1")
    config = config or TrainingConfig()
    feature_config = feature_config or FeatureConfig()
    working_history = region_df.sort_values(["admin1", "date"]).reset_index(drop=True).copy()
    forecasts: list[pd.DataFrame] = []
    history_snapshots: list[pd.DataFrame] = []
    for step in range(months):
        history_snapshots.append(working_history.copy())
        step_predictions = forecast_next_month(
            working_history,
            pipeline,
            config=config,
            feature_config=feature_config,
        ).copy()
        step_predictions["step"] = step + 1
        forecasts.append(step_predictions)
        working_history = _append_predictions_to_history(
            working_history,
            step_predictions,
            config=config,
        )
    combined = pd.concat(forecasts, ignore_index=True)
    return combined, history_snapshots
