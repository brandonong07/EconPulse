"""Composite scoring logic for EconPulse."""

from __future__ import annotations

import math

import pandas as pd


MACRO_HEALTH_WEIGHTS = {
    "job_market_strength": 0.30,
    "consumer_sentiment": 0.20,
    "inflation_pressure": 0.18,
    "wage_strength": 0.12,
    "rent_pressure": 0.10,
    "borrowing_pressure": 0.10,
}
MACRO_HEALTH_SPREAD = 1.5
MACRO_STATE_BUCKETS = [
    {
        "key": "severe_stress",
        "label": "Severe Stress",
        "range": "0-20",
        "min": 0,
        "max": 20,
    },
    {
        "key": "strained",
        "label": "Strained",
        "range": "20-40",
        "min": 20,
        "max": 40,
    },
    {
        "key": "pre_growth",
        "label": "Fragile",
        "range": "40-50",
        "min": 40,
        "max": 50,
    },
    {
        "key": "normal",
        "label": "Stable",
        "range": "50-60",
        "min": 50,
        "max": 60,
    },
    {
        "key": "growth",
        "label": "Healthy",
        "range": "60-80",
        "min": 60,
        "max": 80,
    },
    {
        "key": "strong",
        "label": "Strong Expansion",
        "range": "80-100",
        "min": 80,
        "max": 100.0001,
    },
]


def normalize(series: pd.Series, invert: bool = False) -> pd.Series:
    """Normalize a series to a clipped 0-100 score."""
    numeric = pd.to_numeric(series, errors="coerce")
    mean = numeric.mean()
    std = numeric.std()

    if pd.isna(std) or std == 0:
        scaled = pd.Series(50.0, index=numeric.index)
    else:
        z_score = (numeric - mean) / std
        scaled = ((z_score * 10) + 50).clip(0, 100)

    return 100 - scaled if invert else scaled


def _mean_with_min_coverage(components: list[pd.Series], min_coverage: float = 0.5) -> pd.Series | None:
    if not components:
        return None

    frame = pd.concat(components, axis=1).sort_index()
    min_count = max(1, math.ceil(len(components) * min_coverage))
    mean = frame.mean(axis=1, skipna=True)
    return mean.where(frame.count(axis=1) >= min_count)


def _weighted_mean(frame: pd.DataFrame, weights: dict[str, float], min_weight: float = 0.65) -> pd.Series:
    """Compute a weighted mean while allowing some missing categories."""
    usable_weights = {column: weight for column, weight in weights.items() if column in frame.columns}
    if not usable_weights:
        return pd.Series(dtype=float)

    weighted = pd.DataFrame(
        {
            column: pd.to_numeric(frame[column], errors="coerce") * weight
            for column, weight in usable_weights.items()
        }
    )
    available_weight = pd.DataFrame(
        {
            column: frame[column].notna().astype(float) * weight
            for column, weight in usable_weights.items()
        }
    ).sum(axis=1)

    mean = weighted.sum(axis=1, skipna=True) / available_weight
    return mean.where(available_weight >= min_weight)


def _macro_stress_penalty(score_df: pd.DataFrame) -> pd.Series:
    """
    Penalize periods where broad macro stress should not average away.

    This makes labor-market collapses, inflation shocks, weak sentiment, and
    tight credit show up as recessive/strained periods in the overall score.
    """
    penalty = pd.Series(0.0, index=score_df.index)

    if "job_market_strength" in score_df:
        job = score_df["job_market_strength"]
        penalty += (40 - job).clip(lower=0) * 0.65
        penalty += (30 - job).clip(lower=0) * 0.50

    if "consumer_sentiment" in score_df:
        penalty += (40 - score_df["consumer_sentiment"]).clip(lower=0) * 0.22

    if "inflation_pressure" in score_df:
        penalty += (35 - score_df["inflation_pressure"]).clip(lower=0) * 0.28

    if "borrowing_pressure" in score_df:
        penalty += (35 - score_df["borrowing_pressure"]).clip(lower=0) * 0.16

    if "rent_pressure" in score_df:
        penalty += (35 - score_df["rent_pressure"]).clip(lower=0) * 0.16

    stress_columns = [column for column in MACRO_HEALTH_WEIGHTS if column in score_df.columns]
    if stress_columns:
        weak_category_count = score_df[stress_columns].lt(40).sum(axis=1)
        penalty += (weak_category_count - 2).clip(lower=0) * 2

    return penalty


def _compute_overall_macro_health(score_df: pd.DataFrame) -> pd.Series:
    """
    Calculate overall macro health on an absolute-feeling 0-100 scale.

    Category scores are already consumer-friendly, where higher is better. The
    weighted average is then widened around 50 so genuinely weak/strong periods
    are easier to distinguish from ordinary mixed conditions.
    """
    base = _weighted_mean(score_df, MACRO_HEALTH_WEIGHTS)
    penalty = _macro_stress_penalty(score_df)
    raw_macro = base - penalty
    widened = 50 + ((raw_macro - 50) * MACRO_HEALTH_SPREAD)
    return widened.clip(0, 100)


def macro_state_for_score(score: float | int | None) -> dict[str, str | None]:
    """Return a frontend-friendly macro state label for an overall score."""
    if pd.isna(score):
        return {"key": None, "label": None, "range": None}

    value = float(score)
    for bucket in MACRO_STATE_BUCKETS:
        if bucket["min"] <= value < bucket["max"]:
            return {
                "key": bucket["key"],
                "label": bucket["label"],
                "range": bucket["range"],
            }

    return {"key": None, "label": None, "range": None}


def add_macro_state_columns(score_df: pd.DataFrame) -> pd.DataFrame:
    """Attach macro state columns to a score dataframe."""
    if "overall_health" not in score_df.columns:
        return score_df

    out = score_df.copy()
    states = out["overall_health"].apply(macro_state_for_score)
    out["macro_state"] = states.apply(lambda state: state["key"])
    out["macro_state_label"] = states.apply(lambda state: state["label"])
    out["macro_state_range"] = states.apply(lambda state: state["range"])
    return out


def compute_scores(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combine normalized sub-components into composite scores.

    Higher scores are better for consumers. Unlike the notebook's
    dropna(how="any"), this keeps rows with enough available category data.
    """
    scores: dict[str, pd.Series] = {}

    rp = dfs.get("rent_pressure", pd.DataFrame())
    rp_components = [
        normalize(rp[col], invert=True)
        for col in ["rent_cpi_yoy", "shelter_cpi_yoy", "mortgage_rate"]
        if col in rp.columns
    ]
    rent_score = _mean_with_min_coverage(rp_components)
    if rent_score is not None:
        scores["rent_pressure"] = rent_score

    jm = dfs.get("job_market", pd.DataFrame())
    jm_components = [
        normalize(jm[col], invert=invert)
        for col, invert in [
            ("unemployment", True),
            ("job_openings", False),
            ("initial_claims", True),
            ("labor_force_part", False),
        ]
        if col in jm.columns
    ]
    job_score = _mean_with_min_coverage(jm_components)
    if job_score is not None:
        scores["job_market_strength"] = job_score

    ip = dfs.get("inflation_pressure", pd.DataFrame())
    ip_components = [
        normalize(ip[col], invert=True)
        for col in ["cpi_all_yoy", "core_cpi_yoy", "food_cpi_yoy"]
        if col in ip.columns
    ]
    inflation_score = _mean_with_min_coverage(ip_components)
    if inflation_score is not None:
        scores["inflation_pressure"] = inflation_score

    bp = dfs.get("borrowing_pressure", pd.DataFrame())
    bp_components = [
        normalize(bp[col], invert=True)
        for col in ["fed_funds_rate", "credit_card_rate", "mortgage_rate"]
        if col in bp.columns
    ]
    borrowing_score = _mean_with_min_coverage(bp_components)
    if borrowing_score is not None:
        scores["borrowing_pressure"] = borrowing_score

    ws = dfs.get("wage_strength", pd.DataFrame())
    ws_components = [
        normalize(ws[col], invert=False)
        for col in ["avg_hourly_earn_yoy", "real_hourly_earn_yoy"]
        if col in ws.columns
    ]
    wage_score = _mean_with_min_coverage(ws_components)
    if wage_score is not None:
        scores["wage_strength"] = wage_score

    cs = dfs.get("consumer_sentiment", pd.DataFrame())
    if "umich_sentiment" in cs.columns:
        scores["consumer_sentiment"] = normalize(cs["umich_sentiment"])

    score_df = pd.DataFrame(scores).dropna(how="all").sort_index()

    score_df["overall_health"] = _compute_overall_macro_health(score_df)
    return add_macro_state_columns(score_df)
