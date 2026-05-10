"""Run the complete EconPulse backend data pipeline."""

from __future__ import annotations

import shutil

from backend.config import (
    DASHBOARD_METRICS_PATH,
    FRONTEND_DATA_DIR,
    LATEST_METRICS_PATH,
    MODEL_ARTIFACTS_PATH,
    MODEL_RESULTS_PATH,
    PROCESSED_INDICATORS_PATH,
    SCORES_PATH,
    ensure_data_dirs,
    get_fred_api_key,
)
from backend.fred_client import fetch_all
from backend.model_artifacts import evaluate_model_artifacts
from backend.process_indicators import (
    build_all_category_dfs,
    build_dashboard_metrics,
    df_to_records,
    latest_metrics,
    processed_indicators_payload,
    write_json,
)
from backend.scoring import compute_scores
from backend.train_model import train_models


def sync_frontend_data(outputs: dict[str, str]) -> None:
    """Copy processed JSON into the static frontend public data folder."""
    if not FRONTEND_DATA_DIR.parent.exists():
        return

    FRONTEND_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for path in outputs.values():
        shutil.copy2(path, FRONTEND_DATA_DIR / path.split("/")[-1])


def run_pipeline() -> dict[str, str]:
    """Fetch data, process indicators, train model, and export JSON files."""
    ensure_data_dirs()

    raw, raw_source = fetch_all(get_fred_api_key(), return_source=True)
    category_dfs = build_all_category_dfs(raw)
    scores = compute_scores(category_dfs)
    latest = latest_metrics(raw)
    model_results = train_models(scores)
    model_artifacts = evaluate_model_artifacts(scores)
    dashboard_metrics = build_dashboard_metrics(scores, latest, model_results, raw_source)

    write_json(processed_indicators_payload(category_dfs), PROCESSED_INDICATORS_PATH)
    write_json(df_to_records(scores), SCORES_PATH)
    write_json(latest, LATEST_METRICS_PATH)
    write_json(model_results, MODEL_RESULTS_PATH)
    write_json(model_artifacts, MODEL_ARTIFACTS_PATH)
    write_json(dashboard_metrics, DASHBOARD_METRICS_PATH)

    outputs = {
        "processed_indicators": str(PROCESSED_INDICATORS_PATH),
        "scores": str(SCORES_PATH),
        "latest_metrics": str(LATEST_METRICS_PATH),
        "model_results": str(MODEL_RESULTS_PATH),
        "model_artifacts": str(MODEL_ARTIFACTS_PATH),
        "dashboard_metrics": str(DASHBOARD_METRICS_PATH),
    }

    print("\nEconPulse pipeline complete")
    print(f"Raw data source: {raw_source}")
    for name, path in outputs.items():
        print(f"- {name}: {path}")

    sync_frontend_data(outputs)
    if FRONTEND_DATA_DIR.exists():
        print(f"- frontend_data: {FRONTEND_DATA_DIR}")

    return outputs


if __name__ == "__main__":
    run_pipeline()
