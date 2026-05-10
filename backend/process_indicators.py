"""Cleaning, transforming, and exporting economic indicator data."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from backend.config import SERIES
from backend.scoring import MACRO_STATE_BUCKETS, macro_state_for_score


CATEGORY_LABELS = {
    "rent_pressure": "Rent Pressure",
    "job_market_strength": "Job Market",
    "inflation_pressure": "Inflation",
    "borrowing_pressure": "Borrowing",
    "wage_strength": "Wages",
    "consumer_sentiment": "Consumer Sentiment",
}

CATEGORY_READS = {
    "rent_pressure": {
        "drag": "housing costs are still pressuring affordability",
        "support": "housing affordability is not adding much extra stress",
    },
    "job_market_strength": {
        "drag": "the labor market is softening",
        "support": "the labor market is still providing support",
    },
    "inflation_pressure": {
        "drag": "inflation conditions remain uncomfortable",
        "support": "price pressure is more contained",
    },
    "borrowing_pressure": {
        "drag": "borrowing conditions remain tight",
        "support": "credit conditions are less restrictive",
    },
    "wage_strength": {
        "drag": "wage momentum is not doing enough to offset costs",
        "support": "wages are helping offset household costs",
    },
    "consumer_sentiment": {
        "drag": "households are reporting weak confidence",
        "support": "household confidence is helping the read",
    },
}


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


def _trend_sentence(change_1m: float | None, change_3m: float | None) -> str:
    if change_1m is None:
        return "Recent trend data is limited."

    one_month = abs(change_1m)
    if abs(change_1m) < 0.5:
        direction = "was roughly flat over the last month"
    elif change_1m > 0:
        direction = f"improved by {one_month:.1f} points over the last month"
    else:
        direction = f"fell by {one_month:.1f} points over the last month"

    if change_3m is None or abs(change_3m) < 0.5:
        return f"The score {direction}."

    three_month = abs(change_3m)
    if change_3m > 0:
        return f"The score {direction}, but it is still up {three_month:.1f} points over three months."
    return f"The score {direction}, and it is down {three_month:.1f} points over three months."


def _join_names(names: list[str]) -> str:
    if not names:
        return "the available categories"
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{', '.join(names[:-1])}, and {names[-1]}"


def _outlook_sentence(state_key: str | None, trend_1m: float | None) -> str:
    if state_key == "severe_stress":
        return "This points to a high-stress economy where recession-like pressure is already visible."
    if state_key == "strained":
        if trend_1m is not None and trend_1m < -0.5:
            return "This means the economy is under pressure and could slow further if weak confidence and cost conditions persist."
        return "This means the economy is under pressure, but not in the deepest stress zone."
    if state_key == "pre_growth":
        return "This suggests an uneven economy: not crisis-level, but not broad-based growth yet."
    if state_key == "normal":
        return "This suggests a mostly stable economy with mixed but manageable pressure."
    if state_key == "growth":
        return "This suggests healthy economic momentum across most categories."
    if state_key == "strong":
        return "This suggests unusually strong broad-based expansion."
    return "The outlook description is unavailable because the score is missing."


def _score_explanation(
    score_payload: dict[str, Any],
    trends: dict[str, Any],
    macro_state: dict[str, str | None],
) -> dict[str, Any]:
    overall = score_payload.get("overall_health")
    if overall is None:
        return {
            "summary": "Overall health could not be calculated because too many inputs are missing.",
            "drivers": [],
            "supports": [],
            "outlook": "Add more complete monthly data to generate an interpretation.",
            "method": "Overall Health is a weighted 0-100 score built from labor, sentiment, inflation, wages, rent, and borrowing conditions.",
        }

    category_scores = {
        key: float(value)
        for key, value in score_payload.items()
        if key != "overall_health" and pd.notna(value)
    }
    weak = sorted(category_scores.items(), key=lambda item: item[1])[:3]
    strong = sorted(category_scores.items(), key=lambda item: item[1], reverse=True)[:2]

    drivers = [
        {
            "key": key,
            "label": CATEGORY_LABELS.get(key, key),
            "score": round(value, 1),
            "description": CATEGORY_READS.get(key, {}).get("drag", "this category is weighing on the score"),
        }
        for key, value in weak
    ]
    supports = [
        {
            "key": key,
            "label": CATEGORY_LABELS.get(key, key),
            "score": round(value, 1),
            "description": CATEGORY_READS.get(key, {}).get("support", "this category is supporting the score"),
        }
        for key, value in strong
    ]

    main_drivers = _join_names([driver["label"].lower() for driver in drivers[:2]])
    state_label = macro_state.get("label") or "Unknown"
    summary = (
        f"The {overall:.1f} score lands in the {state_label} range because the weakest inputs are "
        f"{main_drivers}. {_trend_sentence(trends.get('overall_health_change_1m'), trends.get('overall_health_change_3m'))}"
    )

    return {
        "summary": summary,
        "drivers": drivers,
        "supports": supports,
        "outlook": _outlook_sentence(macro_state.get("key"), trends.get("overall_health_change_1m")),
        "method": (
            "Formula: base score = 30% job market + 20% consumer sentiment + 18% inflation "
            "+ 12% wages + 10% rent + 10% borrowing. Higher category scores are better. "
            "EconPulse then subtracts stress penalties when labor, sentiment, inflation, "
            "borrowing, rent, or several categories at once are weak, widens the result around "
            "50, and bounds it from 0 to 100."
        ),
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
    trends = {
        "overall_health_change_1m": _metric_change(usable_scores, "overall_health", 1),
        "overall_health_change_3m": _metric_change(usable_scores, "overall_health", 3),
        "overall_health_change_12m": _metric_change(usable_scores, "overall_health", 12),
        "student_cost_pressure_change_1m": model_results.get("student_cost_pressure_change_1m"),
        "student_cost_pressure_change_3m": model_results.get("student_cost_pressure_change_3m"),
    }

    return {
        "as_of": as_of,
        "data_refreshed_at": date.today().isoformat(),
        "raw_data_source": raw_source,
        "overall_health": overall_health,
        "macro_state": macro_state,
        "state_counts": _state_counts(scores),
        "trends": trends,
        "score_explanation": _score_explanation(score_payload, trends, macro_state),
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
            "prediction_label": "3-month forecast",
            "prediction_note": "Forecast target only; this is not observed future FRED data.",
            "best_model": model_results.get("best_model"),
        },
    }
