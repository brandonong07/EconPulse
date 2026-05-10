"""Run the complete EconPulse backend data pipeline."""

from __future__ import annotations

from backend.config import (
    DASHBOARD_METRICS_PATH,
    LATEST_METRICS_PATH,
    MODEL_RESULTS_PATH,
    PROCESSED_INDICATORS_PATH,
    SCORES_PATH,
    ensure_data_dirs,
    get_fred_api_key,
)
from backend.fred_client import fetch_all
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


def run_pipeline() -> dict[str, str]:
    """Fetch data, process indicators, train model, and export JSON files."""
    ensure_data_dirs()

    raw, raw_source = fetch_all(get_fred_api_key(), return_source=True)
    category_dfs = build_all_category_dfs(raw)
    scores = compute_scores(category_dfs)
    latest = latest_metrics(raw)
    model_results = train_models(scores)
    dashboard_metrics = build_dashboard_metrics(scores, latest, model_results, raw_source)

    write_json(processed_indicators_payload(category_dfs), PROCESSED_INDICATORS_PATH)
    write_json(df_to_records(scores), SCORES_PATH)
    write_json(latest, LATEST_METRICS_PATH)
    write_json(model_results, MODEL_RESULTS_PATH)
    write_json(dashboard_metrics, DASHBOARD_METRICS_PATH)

    outputs = {
        "processed_indicators": str(PROCESSED_INDICATORS_PATH),
        "scores": str(SCORES_PATH),
        "latest_metrics": str(LATEST_METRICS_PATH),
        "model_results": str(MODEL_RESULTS_PATH),
        "dashboard_metrics": str(DASHBOARD_METRICS_PATH),
    }

    print("\nEconPulse pipeline complete")
    print(f"Raw data source: {raw_source}")
    for name, path in outputs.items():
        print(f"- {name}: {path}")

    return outputs


if __name__ == "__main__":
    run_pipeline()
