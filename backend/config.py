"""Shared configuration for the EconPulse backend pipeline."""

from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models"
FRONTEND_DATA_DIR = BASE_DIR / "frontend" / "public" / "data"

RAW_CACHE_PATH = RAW_DIR / "fred_raw.json"
LEGACY_PROCESSED_CACHE_PATH = BASE_DIR / "fred_data.json"
LEGACY_PROCESSED_CACHE_PATHS = (
    LEGACY_PROCESSED_CACHE_PATH,
    BASE_DIR / "eda-analysis" / "fred_data.json",
)

PROCESSED_INDICATORS_PATH = PROCESSED_DIR / "processed_indicators.json"
SCORES_PATH = PROCESSED_DIR / "scores.json"
LATEST_METRICS_PATH = PROCESSED_DIR / "latest_metrics.json"
MODEL_RESULTS_PATH = PROCESSED_DIR / "model_results.json"
MODEL_ARTIFACTS_PATH = PROCESSED_DIR / "model_artifacts.json"
DASHBOARD_METRICS_PATH = PROCESSED_DIR / "dashboard_metrics.json"

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_START_DATE = "1990-01-01"
FRED_TIMEOUT_SECONDS = 10
FRED_RETRIES = 3
FRED_RETRY_BACKOFF_SECONDS = 1.5

MODEL_HORIZON_MONTHS = 3


# category -> { label: FRED series_id }
SERIES = {
    "rent_pressure": {
        "rent_cpi": "CUSR0000SEHA",
        "shelter_cpi": "CUSR0000SAH1",
        "mortgage_rate": "MORTGAGE30US",
        "fed_funds_rate": "FEDFUNDS",
        "real_wages": "AHETPI",
        "housing_price": "CSUSHPISA",
    },
    "job_market": {
        "unemployment": "UNRATE",
        "job_openings": "JTSJOL",
        "initial_claims": "ICSA",
        "labor_force_part": "CIVPART",
        "real_wage_growth": "CES0500000003",
    },
    "inflation_pressure": {
        "cpi_all": "CPIAUCSL",
        "core_cpi": "CPILFESL",
        "food_cpi": "CPIUFDSL",
        "energy_cpi": "CPIENGSL",
        "gas_prices": "GASREGCOVW",
    },
    "borrowing_pressure": {
        "fed_funds_rate": "FEDFUNDS",
        "credit_card_rate": "TERMCBCCALLNS",
        "mortgage_rate": "MORTGAGE30US",
    },
    "wage_strength": {
        "avg_hourly_earn": "CES0500000003",
        "real_hourly_earn": "CES0500000011",
        "cpi_inflation": "CPIAUCSL",
    },
    "consumer_sentiment": {
        "umich_sentiment": "UMCSENT",
    },
}


def ensure_data_dirs() -> None:
    """Create the data folders used by the pipeline."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def all_series_ids() -> list[str]:
    """Return every unique FRED series id used by the app."""
    return sorted({sid for category in SERIES.values() for sid in category.values()})


def get_fred_api_key() -> str | None:
    """
    Load FRED_API_KEY from the environment or a local .env file.

    The key is never defined in source code. This lightweight parser avoids
    adding python-dotenv as a required dependency for the hackathon backend.
    """
    key = os.getenv("FRED_API_KEY")
    if key:
        return key.strip()

    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        return None

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() == "FRED_API_KEY":
            return value.strip().strip('"').strip("'") or None

    return None
