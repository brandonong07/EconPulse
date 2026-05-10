"""FRED fetching, caching, and mock fallback utilities."""

from __future__ import annotations

import json
import re
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests

from backend.config import (
    FRED_BASE_URL,
    FRED_RETRIES,
    FRED_RETRY_BACKOFF_SECONDS,
    FRED_START_DATE,
    FRED_TIMEOUT_SECONDS,
    LEGACY_PROCESSED_CACHE_PATH,
    LEGACY_PROCESSED_CACHE_PATHS,
    RAW_CACHE_PATH,
    SERIES,
    all_series_ids,
    ensure_data_dirs,
)


class FREDClientError(RuntimeError):
    """Raised when live FRED data cannot be fetched."""


def _sanitize_error(message: object) -> str:
    """Remove secrets from request/library error strings before logging."""
    return re.sub(r"(api_key=)[^&\\s]+", r"\1***", str(message))


def _series_to_records(series: pd.Series) -> list[dict]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return [
        {"date": str(index.date()), "value": float(value)}
        for index, value in clean.items()
    ]


def _records_to_series(records: list[dict], name: str) -> pd.Series:
    if not records:
        return pd.Series(dtype=float, name=name)

    df = pd.DataFrame(records)
    if "date" not in df.columns:
        return pd.Series(dtype=float, name=name)

    value_col = "value" if "value" in df.columns else name
    if value_col not in df.columns:
        return pd.Series(dtype=float, name=name)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    return df[value_col].dropna().rename(name)


def fetch_series(series_id: str, api_key: str, session: requests.Session | None = None) -> pd.Series:
    """Fetch one FRED series with timeout and retries."""
    client = session or requests.Session()
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": FRED_START_DATE,
    }

    last_error = "unknown error"
    for attempt in range(1, FRED_RETRIES + 1):
        try:
            response = client.get(
                FRED_BASE_URL,
                params=params,
                timeout=FRED_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            payload = response.json()

            if "observations" not in payload:
                message = payload.get("error_message", "missing observations")
                raise FREDClientError(f"{series_id}: {message}")

            df = pd.DataFrame(payload["observations"])
            if df.empty:
                raise FREDClientError(f"{series_id}: empty observations")

            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna(subset=["date"]).set_index("date").sort_index()
            return df["value"].rename(series_id)
        except Exception as exc:
            last_error = _sanitize_error(exc)
            if attempt < FRED_RETRIES:
                time.sleep(FRED_RETRY_BACKOFF_SECONDS * attempt)

    raise FREDClientError(f"Failed to fetch {series_id}: {last_error}")


def _save_raw_cache(raw: dict[str, pd.Series], path=RAW_CACHE_PATH) -> None:
    ensure_data_dirs()
    payload = {
        "source": "fred",
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "series": {series_id: _series_to_records(series) for series_id, series in raw.items()},
    }
    path.write_text(json.dumps(payload, indent=2))


def load_raw_cache(path=RAW_CACHE_PATH) -> dict[str, pd.Series]:
    """Load the new raw cache format from data/raw/fred_raw.json."""
    payload = json.loads(path.read_text())
    series_payload = payload.get("series", payload)
    return {
        series_id: _records_to_series(records, series_id)
        for series_id, records in series_payload.items()
    }


def load_legacy_processed_cache(path=LEGACY_PROCESSED_CACHE_PATH) -> dict[str, pd.Series]:
    """
    Load the notebook-era processed cache and reconstruct raw series by label.

    This lets the new pipeline run immediately with the existing fred_data.json
    even before a live FRED request succeeds.
    """
    payload = json.loads(path.read_text())
    raw: dict[str, pd.Series] = {}

    for category, records in payload.items():
        if category not in SERIES or not records:
            continue
        df = pd.DataFrame(records)
        if "date" not in df.columns:
            continue
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date").sort_index()

        for label, series_id in SERIES[category].items():
            if label not in df.columns:
                continue
            series = pd.to_numeric(df[label], errors="coerce").dropna().rename(series_id)
            if series.empty:
                continue
            if series_id not in raw or len(series) > len(raw[series_id]):
                raw[series_id] = series

    if not raw:
        raise FREDClientError(f"No usable series found in {path}")
    return raw


def _mock_profile(series_id: str) -> tuple[float, float, float, float | None, float | None]:
    profiles = {
        "CUSR0000SEHA": (281.0, 0.32, 0.25, None, None),
        "CUSR0000SAH1": (275.0, 0.30, 0.25, None, None),
        "MORTGAGE30US": (3.8, 0.018, 0.18, 2.5, 8.0),
        "FEDFUNDS": (0.15, 0.026, 0.14, 0.0, 5.5),
        "AHETPI": (20.8, 0.035, 0.08, None, None),
        "CSUSHPISA": (168.0, 0.62, 1.6, None, None),
        "UNRATE": (5.6, -0.006, 0.18, 3.2, 9.0),
        "JTSJOL": (5200.0, 12.0, 260.0, 2500.0, 12500.0),
        "ICSA": (280000.0, -80.0, 12000.0, 170000.0, 650000.0),
        "CIVPART": (62.8, 0.003, 0.08, 60.0, 64.5),
        "CES0500000003": (24.8, 0.075, 0.09, None, None),
        "CPIAUCSL": (234.7, 0.55, 0.45, None, None),
        "CPILFESL": (239.8, 0.48, 0.35, None, None),
        "CPIUFDSL": (245.0, 0.58, 0.65, None, None),
        "CPIENGSL": (210.0, 0.35, 4.5, 120.0, 360.0),
        "GASREGCOVW": (2.3, 0.008, 0.22, 1.6, 5.2),
        "TERMCBCCALLNS": (12.0, 0.055, 0.45, 10.0, 24.0),
        "CES0500000011": (10.7, 0.006, 0.04, None, None),
        "UMCSENT": (92.0, -0.055, 4.5, 45.0, 105.0),
    }
    return profiles.get(series_id, (100.0, 0.1, 1.0, None, None))


def generate_mock_monthly_data() -> dict[str, pd.Series]:
    """Generate deterministic, realistic-looking monthly fallback data."""
    rng = np.random.default_rng(42)
    dates = pd.date_range(FRED_START_DATE, pd.Timestamp.today().normalize(), freq="MS")
    raw: dict[str, pd.Series] = {}

    for series_id in all_series_ids():
        start, monthly_trend, noise, floor, ceiling = _mock_profile(series_id)
        seasonal = np.sin(np.arange(len(dates)) / 6) * noise * 0.45
        random_walk = rng.normal(0, noise, len(dates)).cumsum() * 0.12
        values = start + (np.arange(len(dates)) * monthly_trend) + seasonal + random_walk
        if floor is not None or ceiling is not None:
            values = np.clip(values, floor if floor is not None else -np.inf, ceiling if ceiling is not None else np.inf)
        raw[series_id] = pd.Series(values, index=dates, name=series_id)

    return raw


def _fallback_raw_data(error_message: str) -> tuple[dict[str, pd.Series], str]:
    if RAW_CACHE_PATH.exists():
        print(f"Live FRED unavailable ({error_message}). Loading {RAW_CACHE_PATH}.")
        return load_raw_cache(RAW_CACHE_PATH), "cache"

    for legacy_path in LEGACY_PROCESSED_CACHE_PATHS:
        if legacy_path.exists():
            print(f"Live FRED unavailable ({error_message}). Loading {legacy_path}.")
            return load_legacy_processed_cache(legacy_path), "legacy_cache"

    print(f"Live FRED unavailable ({error_message}). Generating mock monthly data.")
    return generate_mock_monthly_data(), "mock"


def fetch_all(api_key: str | None = None, *, return_source: bool = False):
    """
    Fetch every unique FRED series, falling back to cached or mock data.

    Returns a raw series dictionary by default. Set return_source=True to also
    receive "fred", "cache", "legacy_cache", or "mock".
    """
    if not api_key:
        raw, source = _fallback_raw_data("FRED_API_KEY is not set")
        return (raw, source) if return_source else raw

    try:
        raw: dict[str, pd.Series] = {}
        with requests.Session() as session:
            for series_id in all_series_ids():
                raw[series_id] = fetch_series(series_id, api_key, session=session)
        _save_raw_cache(raw)
        return (raw, "fred") if return_source else raw
    except Exception as exc:
        raw, source = _fallback_raw_data(str(exc))
        return (raw, source) if return_source else raw
