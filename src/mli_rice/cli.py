"""Command-line interface for managing the rice forecasting pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from . import data as data_module
from .features import FeatureConfig, build_feature_table
from .modeling import TrainingConfig, multi_step_forecast, train_regressors
from .rules import advisories_to_frame, generate_advisories

console = Console()
app = typer.Typer(help="Utilities for training and evaluating rice price forecasting models.")


def _resolve_data(path: Optional[Path]) -> pd.DataFrame:
    return data_module.load_rice_data(path)


@app.command()
def describe(data_path: Path = typer.Option(None, help="Path to rice.csv")) -> None:
    """Show basic dataset statistics."""
    df = _resolve_data(data_path)
    console.print(f"Loaded {len(df):,} rows spanning {df['year'].min()} to {df['year'].max()}.")
    table = Table(title="Regional coverage")
    table.add_column("Region", justify="left")
    table.add_column("Observations", justify="right")
    table.add_column("First Year", justify="right")
    table.add_column("Last Year", justify="right")
    for region, group in df.groupby("admin1"):
        table.add_row(region, f"{len(group):,}", str(group["year"].min()), str(group["year"].max()))
    console.print(table)


@app.command()
def train(
    data_path: Path = typer.Option(None, help="Path to rice.csv"),
    holdout_months: int = typer.Option(12, help="Number of months to reserve for evaluation"),
    horizon: int = typer.Option(1, help="Forecast horizon in months"),
) -> None:
    """Train the regression models and persist the best pipeline."""
    df = _resolve_data(data_path)
    region_df = data_module.region_monthly_average(df)
    feature_cfg = FeatureConfig(horizon=horizon)
    feature_table = build_feature_table(region_df, config=feature_cfg)
    train_cfg = TrainingConfig(holdout_months=holdout_months)
    result = train_regressors(feature_table, config=train_cfg, feature_config=feature_cfg)
    console.print(f"Best model: [bold]{result['selected_model']}[/bold]")
    for name, stats in result["metrics"].items():
        console.print(
            f"{name}: CV RMSE={stats['cv_rmse']:.3f}, CV R2={stats['cv_r2']:.3f}, "
            f"Holdout RMSE={stats['holdout_rmse']:.3f}, Holdout R2={stats['holdout_r2']:.3f}"
        )
    console.print(f"Artifacts saved to {train_cfg.artifact_path} and {train_cfg.metrics_path}")


@app.command()
def forecast(
    data_path: Path = typer.Option(None, help="Path to rice.csv"),
    output_path: Path = typer.Option(Path("reports/next_month_forecast.csv")),
    advisories_path: Path = typer.Option(Path("reports/rule_based_advisories.csv")),
    forecast_months: int = typer.Option(1, min=1, help="Number of future months to forecast"),
    horizon: int = typer.Option(1, help="Horizon used during training (must match saved model)"),
) -> None:
    """Generate next-month forecasts for each region using the saved pipeline."""
    df = _resolve_data(data_path)
    region_df = data_module.region_monthly_average(df)
    config = TrainingConfig()
    pipeline = joblib.load(config.artifact_path)
    feature_cfg = FeatureConfig(horizon=horizon)
    predictions, history_snapshots = multi_step_forecast(
        region_df,
        pipeline,
        months=forecast_months,
        config=config,
        feature_config=feature_cfg,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    advisory_frames: list[pd.DataFrame] = []
    for step_idx, history in enumerate(history_snapshots, start=1):
        step_predictions = predictions[predictions["step"] == step_idx]
        advisories = generate_advisories(history, step_predictions)
        if not advisories:
            continue
        adv_frame = advisories_to_frame(advisories)
        adv_frame = adv_frame.merge(
            step_predictions[[config.region_col, "forecast_date"]],
            left_on="region",
            right_on=config.region_col,
            how="left",
        ).drop(columns=[config.region_col])
        adv_frame["forecast_step"] = step_idx
        advisory_frames.append(adv_frame)
    if advisory_frames:
        advisories_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(advisory_frames, ignore_index=True).to_csv(advisories_path, index=False)
        console.print(f"Saved forecasts to {output_path} and advisories to {advisories_path}")
    else:
        console.print(f"Saved forecasts to {output_path}. No advisories triggered.")


if __name__ == "__main__":
    app()
