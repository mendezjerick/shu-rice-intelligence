"""Rule-based advisory system for interpreting price forecasts."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, List

import pandas as pd


@dataclass(slots=True)
class Advisory:
    region: str
    category: str
    message: str
    rationale: str
    sdg_focus: str


def _recent_trend(prices: pd.Series, window: int = 3) -> float:
    if len(prices) < window:
        return 0.0
    return (prices.iloc[-1] - prices.iloc[-window]) / max(prices.iloc[-window], 1e-6)


def generate_advisories(
    region_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    *,
    trend_window: int = 3,
) -> List[Advisory]:
    advisories: list[Advisory] = []
    national_monthly = region_df.groupby("date")["avg_price"].mean()
    national_mean = national_monthly.tail(12).mean()
    national_std = national_monthly.tail(12).std()

    for _, row in forecast_df.iterrows():
        region = row["admin1"]
        regional_history = region_df[region_df["admin1"] == region].sort_values("date")
        pct_change = row["pct_change"]
        trend = _recent_trend(regional_history["avg_price"], window=trend_window)
        latest_price = regional_history["avg_price"].iloc[-1]
        future_price = row["forecast_price"]

        if pct_change >= 0.05 or trend > 0.03:
            advisories.append(
                Advisory(
                    region=region,
                    category="Supply Risk",
                    message="Prepare for lean supply-plan short-term imports or buffer stocks.",
                    rationale=(
                        "Forecasted price is rising faster than 5% month-over-month, "
                        "which mirrors consecutive increases observed in recent records."
                    ),
                    sdg_focus="SDG2",
                )
            )
        if pct_change <= -0.03:
            advisories.append(
                Advisory(
                    region=region,
                    category="Market Opportunity",
                    message="Encourage local market sales to stabilize farmer income.",
                    rationale="Projected price decline exceeds 3%, signaling demand-softening windows for volume sales.",
                    sdg_focus="SDG8",
                )
            )
        if future_price > national_mean + national_std:
            advisories.append(
                Advisory(
                    region=region,
                    category="Consumer Protection",
                    message="Intensify price monitoring and activate targeted subsidies for vulnerable households.",
                    rationale="Regional forecast exceeds the recent national average by more than one standard deviation.",
                    sdg_focus="SDG2",
                )
            )
        if future_price < national_mean - national_std and pct_change < 0:
            advisories.append(
                Advisory(
                    region=region,
                    category="Trade Optimization",
                    message="Consider transporting surplus to deficit regions or timing bulk buyers for better margins.",
                    rationale="Prices are dipping well below the national mean, indicating an opportunity to balance inventories across regions.",
                    sdg_focus="SDG8",
                )
            )
        if trend > 0.02 and pct_change > 0:
            advisories.append(
                Advisory(
                    region=region,
                    category="Local Government Action",
                    message="Alert LGUs/DA for early interventions-cash-for-work on post-harvest storage and logistics.",
                    rationale="Three-month trend and forecasts both point upward, risking household purchasing power.",
                    sdg_focus="SDG2",
                )
            )
    return advisories


def advisories_to_frame(advisories: Iterable[Advisory]) -> pd.DataFrame:
    return pd.DataFrame([asdict(a) for a in advisories])
