from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

# Prefer NumPy-backed dtypes to avoid pyarrow struct/list accessor errors (guarded for older pandas).
try:  # pandas>=2.1
    pd.set_option("mode.dtype_backend", "numpy_nullable")
except Exception:  # pragma: no cover
    pass

ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from mli_rice import data as data_module  # noqa: E402
from mli_rice.features import FeatureConfig, build_feature_table  # noqa: E402
from mli_rice.modeling import TrainingConfig, multi_step_forecast  # noqa: E402
from mli_rice.rules import advisories_to_frame, generate_advisories  # noqa: E402

_WARNED_GLOBAL: set[str] = set()

@dataclass
class AppPaths:
    data_path: Path
    artifacts_dir: Path
    metrics_path: Path


class DataManager:
    """Wraps the existing rice forecasting logic so it can be called from PyWebView."""

    def __init__(self, base_path: Path) -> None:
        self._warned: set[str] = set()
        self.paths = AppPaths(
            data_path=base_path / "rice.csv",
            artifacts_dir=base_path / "artifacts",
            metrics_path=base_path / "artifacts" / "metrics.json",
        )
        self.raw_df = self._load_rice()
        # Normalize dtypes to numpy to avoid pyarrow struct/list accessors.
        try:
            self.raw_df = self.raw_df.convert_dtypes(dtype_backend="numpy_nullable")
        except Exception:
            pass
        self.region_df = data_module.region_monthly_average(self.raw_df) if not self.raw_df.empty else pd.DataFrame()
        self.national_df = data_module.national_monthly_average(self.raw_df) if not self.raw_df.empty else pd.DataFrame()
        self.feature_cfg = FeatureConfig()
        self.training_cfg = TrainingConfig(artifacts_dir=self.paths.artifacts_dir)
        self.feature_table = self._build_feature_table()
        self.train_df, self.holdout_df = self._split_feature_table()
        self.metrics = self._load_metrics()
        self.pipeline = self._load_pipeline()

    def _load_rice(self) -> pd.DataFrame:
        if not self.paths.data_path.exists():
            self._warn_once("data_missing", f"Missing dataset at {self.paths.data_path}")
            return pd.DataFrame()
        try:
            return data_module.load_rice_data(self.paths.data_path)
        except Exception as exc:  # noqa: BLE001
            self._warn_once("data_load", f"Failed to load rice.csv: {exc}")
            return pd.DataFrame()

    def _build_feature_table(self) -> pd.DataFrame:
        if self.region_df.empty:
            return pd.DataFrame()
        try:
            return build_feature_table(self.region_df, config=self.feature_cfg)
        except Exception as exc:  # noqa: BLE001
            # Swallow feature-table errors to avoid console spam; UI will fall back to naive forecast.
            self._warn_once("feature_table", f"Feature table unavailable: {exc}")
            return pd.DataFrame()

    def _split_feature_table(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self.feature_table.empty:
            return pd.DataFrame(), pd.DataFrame()
        ft = self.feature_table.sort_values(["date", self.training_cfg.region_col]).reset_index(drop=True)
        cutoff = ft["date"].max() - pd.DateOffset(months=self.training_cfg.holdout_months - 1)
        mask = ft["date"] >= cutoff
        return ft.loc[~mask], ft.loc[mask]

    def _load_pipeline(self):
        artifact = self.training_cfg.artifact_path
        if not artifact.exists():
            return None
        return joblib.load(artifact)

    def _load_metrics(self) -> Optional[dict]:
        if self.paths.metrics_path.exists():
            return json.loads(self.paths.metrics_path.read_text(encoding="utf-8"))
        return None

    def forecast(self, months: int, target_year: Optional[int], target_month: Optional[int]):
        if self.pipeline is None or self.region_df.empty:
            return None, []
        try:
            forecasts, histories = multi_step_forecast(
                self.region_df,
                self.pipeline,
                months=months,
                config=self.training_cfg,
                feature_config=self.feature_cfg,
            )
        except Exception as exc:  # noqa: BLE001
            self._warn_once("forecast_error", f"Forecast error: {exc}")
            return None, []
        if target_year and target_month:
            target_date = pd.Timestamp(year=target_year, month=target_month, day=1)
            forecasts = forecasts[forecasts["forecast_date"] == target_date]
        forecasts = forecasts.sort_values(["step", self.training_cfg.region_col]).reset_index(drop=True)
        return forecasts, histories

    def _warn_once(self, key: str, message: str) -> None:
        if key in self._warned or key in _WARNED_GLOBAL:
            return
        self._warned.add(key)
        _WARNED_GLOBAL.add(key)
        # Keep warnings minimal to avoid flooding the console under pywebview.
        print(message)


class RiceAppBackend:
    """Exposes methods that PyWebView can reach from JavaScript."""

    def __init__(self, base_path: Path | None = None) -> None:
        base_dir = Path(getattr(sys, "_MEIPASS", base_path or Path(__file__).resolve().parent))
        # Avoid exposing Path objects on the API; keep a string only if needed for logs.
        self._root = str(base_dir)
        # Keep DataManager private so pywebview does not try to serialize its fields.
        self._data_mgr = DataManager(base_dir)
        self._warned: set[str] = set()

    def get_overview(self) -> dict:
        """Return summary metrics for the dashboard tiles."""
        df = getattr(self._data_mgr, "region_df", pd.DataFrame())
        if not df.empty:
            row_count = int(len(df))
            region_col = getattr(self._data_mgr.training_cfg, "region_col", "region")
            region_count = int(df[region_col].nunique())
            min_date = pd.to_datetime(df["date"]).min()
            max_date = pd.to_datetime(df["date"]).max()
            year_range = f"{min_date.year}–{max_date.year}"
            latest_month = max_date.strftime("%Y-%m")
        else:
            today = pd.Timestamp.today()
            row_count = 0
            region_count = 0
            year_range = f"{today.year - 2}–{today.year}"
            latest_month = pd.Timestamp(today.year, today.month, 1).strftime("%Y-%m")

        return {
            "rows": row_count,
            "regions": region_count,
            "year_range": year_range,
            "latest_month": latest_month,
        }

    def get_national_series(self) -> dict:
        """Serve national price history for the Plotly chart."""
        national_df = getattr(self._data_mgr, "national_df", pd.DataFrame())
        if not national_df.empty:
            series = national_df.sort_values("date").tail(36)
            dates = series["date"].dt.strftime("%Y-%m").tolist()
            values = [round(float(v), 2) for v in series["national_price"]]
            return {"dates": dates, "values": values}

        # Fallback demo data if no dataset/model is available.
        today = pd.Timestamp.today().normalize()
        dates = []
        values = []
        base = 42.0
        for months_back in range(15, -1, -1):
            ts = today - pd.DateOffset(months=months_back)
            dates.append(ts.strftime("%Y-%m"))
            values.append(round(base + months_back * 0.25, 2))
        return {"dates": dates, "values": values}

    def get_regions(self) -> list[str]:
        """Return unique regions available in the dataset."""
        region_df = getattr(self._data_mgr, "region_df", pd.DataFrame())
        region_col = getattr(self._data_mgr.training_cfg, "region_col", "region")
        if region_df.empty or region_col not in region_df.columns:
            return []
        regions = sorted(region_df[region_col].dropna().astype(str).unique().tolist())
        return regions

    def get_price(self, date_str: str, location: str) -> dict:
        """Lookup the closest price on or before the provided date."""
        try:
            target_date = pd.to_datetime(date_str).to_period("M").to_timestamp()
        except Exception:
            return {
                "date": date_str,
                "location": location,
                "price": None,
                "status": "error",
                "message": "Invalid date format. Use YYYY-MM or YYYY-MM-DD.",
            }

        region_df = self._data_mgr.region_df
        region_col = getattr(self._data_mgr.training_cfg, "region_col", "region")
        if region_df.empty or not location:
            return {
                "date": date_str,
                "location": location,
                "price": None,
                "status": "no_data",
                "message": "No data available for that location.",
            }

        min_date = region_df["date"].min()
        max_date = region_df["date"].max()
        if target_date < min_date or target_date > max_date:
            return {
                "date": date_str,
                "location": location,
                "price": None,
                "status": "no_data",
                "message": "Date is outside the available data range.",
            }

        matches = (
            region_df.loc[region_df[region_col] == location]
            .loc[lambda df: df["date"] <= target_date]
            .sort_values("date")
        )
        if matches.empty:
            return {
                "date": date_str,
                "location": location,
                "price": None,
                "status": "no_data",
                "message": "No data found on or before the selected date.",
            }

        row = matches.iloc[-1]
        return {
            "date": row["date"].strftime("%Y-%m-%d"),
            "location": location,
            "price": round(float(row["avg_price"]), 2),
            "status": "ok",
        }

    def get_forecast(self, months: int, target_year: Optional[int], target_month: Optional[int]) -> dict:
        """Return chart, table, and preview data for the forecast screen."""
        months = max(1, min(int(months or 1), 24))
        forecasts, histories = self._data_mgr.forecast(months, target_year, target_month)
        notice = self._target_notice(months, target_year, target_month, forecasts)
        if notice and (forecasts is None or forecasts.empty):
            return self._empty_forecast(notice)
        if forecasts is None or forecasts.empty:
            return self._naive_forecast(months, notice=notice)

        historical_series = self._data_mgr.national_df.sort_values("date").tail(24)
        forecast_series = (
            forecasts.groupby("forecast_date")["forecast_price"]
            .mean()
            .reset_index()
            .sort_values("forecast_date")
        )

        payload = {
            "historical": {
                "dates": historical_series["date"].dt.strftime("%Y-%m").tolist(),
                "values": [round(float(v), 2) for v in historical_series["national_price"]],
            },
            "forecast": {
                "dates": forecast_series["forecast_date"].dt.strftime("%Y-%m").tolist(),
                "values": [round(float(v), 2) for v in forecast_series["forecast_price"]],
            },
            "table": self._format_table_rows(forecasts),
            "train_head": self._frame_preview(self._data_mgr.train_df),
            "holdout_head": self._frame_preview(self._data_mgr.holdout_df),
            "metrics": self._data_mgr.metrics or {"message": "Train your model to populate metrics."},
            "explanation": (
                "Multi-step forecasts are generated from the trained pipeline using region-level "
                "features. Plug your DataManager here if you want to customize the logic."
            ),
            "advisories": self._build_advisories(forecasts, histories),
            "notice": notice,
        }

        # Optionally compute rule-based advisories using histories and forecasts.
        return payload

    def _naive_forecast(self, months: int, notice: Optional[str] = None) -> dict:
        """Fallback forecast using a simple growth curve if the model is unavailable."""
        history = self._data_mgr.national_df.copy()
        if history.empty:
            dates = pd.date_range(end=pd.Timestamp.today(), periods=12, freq="MS")
            history = pd.DataFrame({"date": dates, "national_price": [40 + i for i in range(len(dates))]})

        history = history.sort_values("date")
        history_dates = history["date"].dt.strftime("%Y-%m").tolist()
        history_values = [round(float(v), 2) for v in history["national_price"]]
        last_price = history_values[-1]
        growth = 0.01
        forecast_dates: list[str] = []
        forecast_values: list[float] = []
        for step in range(1, months + 1):
            forecast_date = history["date"].iloc[-1] + pd.DateOffset(months=step)
            forecast_dates.append(forecast_date.strftime("%Y-%m"))
            forecast_values.append(round(last_price * ((1 + growth) ** step), 2))

        region_col = getattr(self._data_mgr.training_cfg, "region_col", "region")
        sample_regions = (
            self._data_mgr.region_df[region_col].unique().tolist() if not self._data_mgr.region_df.empty else ["Sample Region"]
        )
        table_rows = []
        for region in sample_regions:
            for step, (f_date, f_price) in enumerate(zip(forecast_dates, forecast_values, strict=False), start=1):
                table_rows.append(
                    {
                        "region": region,
                        "current_date": history_dates[-1],
                        "forecast_date": f_date,
                        "avg_price": last_price,
                        "forecast_price": f_price,
                        "pct_change": round(((f_price - last_price) / last_price) * 100, 2),
                        "step": step,
                    }
                )

        return {
            "historical": {"dates": history_dates, "values": history_values},
            "forecast": {"dates": forecast_dates, "values": forecast_values},
            "table": table_rows,
            "train_head": [],
            "holdout_head": [],
            "metrics": {"message": "Model not loaded. This is a placeholder forecast."},
            "explanation": (
                "No trained model detected, so this uses a simple growth-based projection. "
                "Replace _naive_forecast with your model output."
            ),
            "advisories": [],
            "notice": notice,
        }

    def _empty_forecast(self, notice: str) -> dict:
        """Return an empty forecast payload with a helpful notice."""
        return {
            "historical": {"dates": [], "values": []},
            "forecast": {"dates": [], "values": []},
            "table": [],
            "train_head": [],
            "holdout_head": [],
            "metrics": {"message": notice},
            "explanation": notice,
            "advisories": [],
            "notice": notice,
        }

    def _frame_preview(self, df: pd.DataFrame) -> list[dict]:
        if df is None or df.empty:
            return []
        preview = df.copy()
        for col in preview.columns:
            if pd.api.types.is_datetime64_any_dtype(preview[col]):
                preview[col] = preview[col].dt.strftime("%Y-%m")
        return preview.head(10).to_dict(orient="records")

    def _format_table_rows(self, forecasts: pd.DataFrame) -> list[dict]:
        rows: list[dict] = []
        for _, row in forecasts.iterrows():
            rows.append(
                {
                    "region": row[getattr(self._data_mgr.training_cfg, "region_col", "region")],
                    "current_date": row["current_date"].strftime("%Y-%m"),
                    "forecast_date": row["forecast_date"].strftime("%Y-%m"),
                    "avg_price": round(float(row["avg_price"]), 2),
                    "forecast_price": round(float(row["forecast_price"]), 2),
                    "pct_change": round(float(row["pct_change"]) * 100, 2),
                    "step": int(row["step"]),
                }
            )
        return rows

    def _build_advisories(self, forecasts: pd.DataFrame, histories: list[pd.DataFrame]) -> list[dict]:
        advisories: list[dict] = []
        region_col = getattr(self._data_mgr.training_cfg, "region_col", "region")
        for step_idx, history in enumerate(histories or [], start=1):
            step_preds = forecasts[forecasts["step"] == step_idx]
            if step_preds.empty:
                continue
            step_advs = generate_advisories(history, step_preds)
            if not step_advs:
                continue
            frame = advisories_to_frame(step_advs)
            frame["forecast_date"] = step_preds["forecast_date"].dt.strftime("%Y-%m").iloc[0]
            frame["step"] = step_idx
            # Ensure region key matches frontend expectation.
            frame = frame.rename(columns={region_col: "region"}) if region_col in frame.columns else frame
            advisories.extend(frame.to_dict(orient="records"))
        return advisories

    def _target_notice(
        self,
        months: int,
        target_year: Optional[int],
        target_month: Optional[int],
        forecasts: pd.DataFrame,
    ) -> Optional[str]:
        if not target_year or not target_month:
            return None
        target_date = pd.Timestamp(year=target_year, month=target_month, day=1)
        if forecasts is not None and not forecasts.empty:
            return None
        latest_feature_date = None
        if not self._data_mgr.region_df.empty:
            latest_feature_date = pd.to_datetime(self._data_mgr.region_df["date"]).max()
        if latest_feature_date is None:
            return "No data available to estimate steps for that target date."
        months_diff = (target_date.year - latest_feature_date.year) * 12 + (target_date.month - latest_feature_date.month)
        if months_diff <= 0:
            return "Target date is within or before the training window; no forecast rows matched."
        return (
            f"No forecast rows matched the target date. "
            f"Try setting months ahead to at least {months_diff} to reach {target_date.strftime('%Y-%m')}."
        )

    def _warn_once(self, key: str, message: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            print(message)
