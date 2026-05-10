"""Cleaning, transforming, and exporting economic indicator data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from backend.config import SERIES
from backend.scoring import MACRO_STATE_BUCKETS, macro_state_for_score


def clean(series: pd.Series, freq: str = "MS") -> pd.Series:
    """
    Resample to monthly starts, forward-fill short gaps, and remove empty ends.
    """
    if series is None or series.empty:
        return pd.Series(dtype=float)

    monthly = pd.to_numeric(series, errors="coerce").sort_index()
    monthly = monthly.resample(freq).last().ffill(limit=5)
    return monthly.dropna()


def yoy(series: pd.Series) -> pd.Series:
    """Year-over-year percent change for monthly data."""
    if series is None or series.empty:
        return pd.Series(dtype=float)
    return series.pct_change(12) * 100


def build_category_df(category: str, raw: dict[str, pd.Series]) -> pd.DataFrame:
    """Build one category dataframe with raw values and YoY columns."""
    frames: dict[str, pd.Series] = {}

    for label, series_id in SERIES[category].items():
        series = clean(raw.get(series_id, pd.Series(dtype=float)))
        if series.empty:
            continue
        frames[label] = series
        frames[f"{label}_yoy"] = yoy(series)

    if not frames:
        return pd.DataFrame()

    df = pd.DataFrame(frames).sort_index()
    df.index.name = "date"
    return df.dropna(how="all")


def build_all_category_dfs(raw: dict[str, pd.Series]) -> dict[str, pd.DataFrame]:
    """Build processed dataframes for every configured category."""
    return {category: build_category_df(category, raw) for category in SERIES}


def latest_metrics(raw: dict[str, pd.Series]) -> dict[str, dict[str, dict[str, Any]]]:
    """Return the latest value and YoY change for each named indicator."""
    out: dict[str, dict[str, dict[str, Any]]] = {}

    for category, series_map in SERIES.items():
        out[category] = {}
        for label, series_id in series_map.items():
            series = clean(raw.get(series_id, pd.Series(dtype=float)))
            if series.empty:
                out[category][label] = {
                    "series_id": series_id,
                    "value": None,
                    "yoy": None,
                    "date": None,
                }
                continue

            yoy_series = yoy(series).dropna()
            out[category][label] = {
                "series_id": series_id,
                "value": round(float(series.iloc[-1]), 4),
                "yoy": round(float(yoy_series.iloc[-1]), 4) if not yoy_series.empty else None,
                "date": str(series.index[-1].date()),
            }

    return out


def df_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a dated dataframe to JSON-safe records."""
    if df.empty:
        return []

    export_df = df.copy()
    export_df.index = pd.to_datetime(export_df.index).strftime("%Y-%m-%d")
    export_df = export_df.reset_index()
    return json.loads(export_df.to_json(orient="records"))


def processed_indicators_payload(dfs: dict[str, pd.DataFrame]) -> dict[str, list[dict[str, Any]]]:
    """Convert all category dataframes to JSON-safe payloads."""
    return {category: df_to_records(df) for category, df in dfs.items()}


def write_json(payload: Any, path: Path) -> None:
    """Write a JSON payload with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False))


def _round_or_none(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), 4)


def _metric_change(df: pd.DataFrame, column: str, periods: int) -> float | None:
    if column not in df.columns:
        return None

    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if len(values) <= periods:
        return None

    return _round_or_none(values.iloc[-1] - values.iloc[-1 - periods])


def _state_counts(scores: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if "overall_health" not in scores.columns:
        return {}

    states = pd.to_numeric(scores["overall_health"], errors="coerce").dropna().apply(
        lambda score: macro_state_for_score(score)["key"]
    )
    counts = states.value_counts()
    return {
        bucket["key"]: {
            "label": bucket["label"],
            "range": bucket["range"],
            "months": int(counts.get(bucket["key"], 0)),
        }
        for bucket in MACRO_STATE_BUCKETS
    }


def build_dashboard_metrics(
    scores: pd.DataFrame,
    latest: dict[str, Any],
    model_results: dict[str, Any],
    raw_source: str,
) -> dict[str, Any]:
    """Create a compact JSON payload for a future dashboard frontend."""
    if "overall_health" in scores.columns:
        usable_scores = scores.dropna(subset=["overall_health"])
    else:
        usable_scores = scores.dropna(how="all")

    latest_scores = usable_scores.tail(1)
    if latest_scores.empty:
        score_payload: dict[str, Any] = {}
        as_of = None
    else:
        row = latest_scores.iloc[0]
        as_of = str(latest_scores.index[0].date())
        score_payload = {
            column: round(float(value), 4)
            for column, value in row.items()
            if pd.notna(value) and pd.api.types.is_number(value)
        }

    overall_health = score_payload.get("overall_health")
    macro_state = macro_state_for_score(overall_health)

    return {
        "as_of": as_of,
        "raw_data_source": raw_source,
        "overall_health": overall_health,
        "macro_state": macro_state,
        "state_counts": _state_counts(scores),
        "trends": {
            "overall_health_change_1m": _metric_change(usable_scores, "overall_health", 1),
            "overall_health_change_3m": _metric_change(usable_scores, "overall_health", 3),
            "overall_health_change_12m": _metric_change(usable_scores, "overall_health", 12),
            "student_cost_pressure_change_1m": model_results.get("student_cost_pressure_change_1m"),
            "student_cost_pressure_change_3m": model_results.get("student_cost_pressure_change_3m"),
        },
        "category_scores": {
            key: value
            for key, value in score_payload.items()
            if key != "overall_health"
        },
        "latest_metrics": latest,
        "student_cost_pressure": {
            "current": model_results.get("latest_student_cost_pressure"),
            "prediction_3_months_ahead": model_results.get("best_prediction"),
            "prediction_date": model_results.get("prediction_date"),
            "best_model": model_results.get("best_model"),
        },
    }
