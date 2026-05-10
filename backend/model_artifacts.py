"""Evaluate saved model files for the frontend model hub."""

from __future__ import annotations

import warnings
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.config import MODEL_DIR


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

WHAT_IF_FEATURES = [
    "rent_pressure",
    "job_market_strength",
    "inflation_pressure",
    "borrowing_pressure",
    "wage_strength",
    "consumer_sentiment",
]

MODEL_ARTIFACTS = [
    {
        "key": "multiple_linear_regression",
        "label": "Multiple Linear Regression",
        "file": "multiple_linear_regression_model.pkl",
        "kind": "sklearn",
    },
    {
        "key": "xgboost_saved",
        "label": "XGBoost",
        "file": "xgboost_model.pkl",
        "kind": "xgboost",
    },
    {
        "key": "lightgbm_saved",
        "label": "LightGBM",
        "file": "lightgbm_model.pkl",
        "kind": "lightgbm",
    },
    {
        "key": "mlp_saved",
        "label": "MLP Neural Network",
        "file": "mlp_model.pkl",
        "kind": "sklearn_mlp",
        "scaler": "mlp_scaler.pkl",
    },
]


def _latest_feature_row(scores: pd.DataFrame) -> pd.Series:
    usable = scores.dropna(subset=WHAT_IF_FEATURES)
    if usable.empty:
        return pd.Series(dtype=float)
    return usable.iloc[-1][WHAT_IF_FEATURES].astype(float)


def _load_pickle(path: Path):
    import joblib

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return joblib.load(path)


def _feature_names(model: Any, scaler: Any | None = None) -> list[str]:
    for obj in [model, scaler]:
        if obj is not None and hasattr(obj, "feature_names_in_"):
            return [str(name) for name in getattr(obj, "feature_names_in_")]
    return WHAT_IF_FEATURES


def _predict(model: Any, values: pd.Series, feature_names: list[str], scaler: Any | None = None) -> float:
    frame = pd.DataFrame([[float(values[name]) for name in feature_names]], columns=feature_names)
    matrix = scaler.transform(frame) if scaler is not None else frame
    prediction = model.predict(matrix)
    return round(float(np.clip(np.ravel(prediction)[0], 0, 100)), 4)


def _sensitivity(model: Any, base_values: pd.Series, feature_names: list[str], scaler: Any | None = None) -> dict[str, Any]:
    sensitivity: dict[str, Any] = {}

    for feature in feature_names:
        if feature not in base_values:
            continue

        base = float(base_values[feature])
        points = []
        for delta in [-20, -10, -5, 0, 5, 10, 20]:
            scenario = base_values.copy()
            scenario[feature] = float(np.clip(base + delta, 0, 100))
            points.append(
                {
                    "delta": delta,
                    "feature_value": round(float(scenario[feature]), 4),
                    "prediction": _predict(model, scenario, feature_names, scaler),
                }
            )

        low = next(point["prediction"] for point in points if point["delta"] == -10)
        high = next(point["prediction"] for point in points if point["delta"] == 10)
        sensitivity[feature] = {
            "base_value": round(base, 4),
            "approx_slope_per_point": round((high - low) / 20, 5),
            "points": points,
        }

    return sensitivity


def evaluate_model_artifacts(scores: pd.DataFrame) -> dict[str, Any]:
    """
    Export saved model predictions and local sensitivity data.

    The frontend cannot execute model files, so this function converts the
    saved models into JSON: current prediction plus slider-friendly sensitivities.
    """
    latest = _latest_feature_row(scores)
    if latest.empty:
        return {"status": "skipped_no_score_features", "features": WHAT_IF_FEATURES, "models": []}

    payload = {
        "status": "evaluated",
        "source": "saved_model_files",
        "features": WHAT_IF_FEATURES,
        "base_input": {feature: round(float(latest[feature]), 4) for feature in WHAT_IF_FEATURES},
        "models": [],
    }

    for artifact in MODEL_ARTIFACTS:
        model_path = MODEL_DIR / artifact["file"]
        scaler_path = MODEL_DIR / artifact.get("scaler", "")
        result = {
            "key": artifact["key"],
            "label": artifact["label"],
            "file": artifact["file"],
            "kind": artifact["kind"],
        }

        if not model_path.exists():
            payload["models"].append({**result, "status": "missing_file"})
            continue

        try:
            model = _load_pickle(model_path)
            scaler = _load_pickle(scaler_path) if artifact.get("scaler") and scaler_path.exists() else None
            feature_names = _feature_names(model, scaler)
            missing = [feature for feature in feature_names if feature not in latest.index]
            if missing:
                payload["models"].append({**result, "status": "missing_features", "missing_features": missing})
                continue

            result.update(
                {
                    "status": "available",
                    "model_class": f"{type(model).__module__}.{type(model).__name__}",
                    "feature_names": feature_names,
                    "feature_count": len(feature_names),
                    "prediction": _predict(model, latest, feature_names, scaler),
                    "sensitivity": _sensitivity(model, latest, feature_names, scaler),
                }
            )
        except Exception as exc:
            result.update({"status": "unavailable", "reason": str(exc)})

        payload["models"].append(result)

    return payload
