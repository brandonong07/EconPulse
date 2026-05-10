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

TARGET_COLUMN = "overall_health"
HORIZON_MONTHS = 3
TEST_SHARE = 0.2

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
    },
]


def train_model_artifacts(scores: pd.DataFrame) -> dict[str, Any]:
    """Train and save the dashboard model artifacts against overall health."""
    import joblib
    from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    training = _training_frame(scores)
    if training.empty:
        return {"status": "skipped_no_training_rows", "target": f"{TARGET_COLUMN}_{HORIZON_MONTHS}_months_ahead"}

    x = training[WHAT_IF_FEATURES]
    y = training["target"].to_numpy(dtype=float)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    models = {
        "multiple_linear_regression_model.pkl": LinearRegression(),
        "xgboost_model.pkl": GradientBoostingRegressor(
            n_estimators=120,
            max_depth=2,
            learning_rate=0.04,
            random_state=42,
        ),
        "lightgbm_model.pkl": ExtraTreesRegressor(
            n_estimators=160,
            max_depth=5,
            min_samples_leaf=4,
            n_jobs=1,
            random_state=42,
        ),
        "mlp_model.pkl": make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(24, 12),
                activation="relu",
                alpha=0.05,
                learning_rate_init=0.01,
                max_iter=2000,
                random_state=42,
            ),
        ),
    }

    saved = []
    for filename, model in models.items():
        model.fit(x, y)
        joblib.dump(model, MODEL_DIR / filename)
        saved.append(filename)

    return {
        "status": "trained",
        "target": f"{TARGET_COLUMN}_{HORIZON_MONTHS}_months_ahead",
        "training_rows": int(len(training)),
        "saved": saved,
    }


def _latest_feature_row(scores: pd.DataFrame) -> pd.Series:
    usable = scores.dropna(subset=WHAT_IF_FEATURES)
    if usable.empty:
        return pd.Series(dtype=float)
    return usable.iloc[-1][WHAT_IF_FEATURES].astype(float)


def _training_frame(scores: pd.DataFrame) -> pd.DataFrame:
    if TARGET_COLUMN not in scores.columns:
        return pd.DataFrame()

    columns = WHAT_IF_FEATURES + [TARGET_COLUMN]
    available = [column for column in columns if column in scores.columns]
    frame = scores.sort_index()[available].copy()
    for column in available:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["target"] = frame[TARGET_COLUMN].shift(-HORIZON_MONTHS)
    return frame.dropna(subset=WHAT_IF_FEATURES + ["target"])


def _regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    mae = float(np.mean(np.abs(actual - predicted)))
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    variance = float(np.sum((actual - np.mean(actual)) ** 2))
    r2 = None if variance == 0 else float(1 - (np.sum((actual - predicted) ** 2) / variance))
    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4) if r2 is not None else None}


def _load_pickle(path: Path):
    import joblib

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return joblib.load(path)


def _feature_names(model: Any, scaler: Any | None = None) -> list[str]:
    for obj in [model, scaler]:
        if obj is not None and hasattr(obj, "feature_names_in_"):
            return [str(name) for name in getattr(obj, "feature_names_in_")]
        if obj is not None and hasattr(obj, "named_steps"):
            for step in getattr(obj, "named_steps").values():
                if hasattr(step, "feature_names_in_"):
                    return [str(name) for name in getattr(step, "feature_names_in_")]
    return WHAT_IF_FEATURES


def _predict(model: Any, values: pd.Series, feature_names: list[str], scaler: Any | None = None) -> float:
    frame = pd.DataFrame([[float(values[name]) for name in feature_names]], columns=feature_names)
    matrix = scaler.transform(frame) if scaler is not None else frame
    prediction = model.predict(matrix)
    return round(float(np.clip(np.ravel(prediction)[0], 0, 100)), 4)


def _predict_frame(model: Any, frame: pd.DataFrame, feature_names: list[str], scaler: Any | None = None) -> np.ndarray:
    feature_frame = frame[feature_names].astype(float)
    matrix = scaler.transform(feature_frame) if scaler is not None else feature_frame
    prediction = model.predict(matrix)
    return np.clip(np.ravel(prediction).astype(float), 0, 100)


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

    training = _training_frame(scores)
    split_index = max(1, int(len(training) * (1 - TEST_SHARE))) if not training.empty else 0
    test_frame = training.iloc[split_index:] if split_index < len(training) else pd.DataFrame()
    valid_target = pd.to_numeric(scores.get(TARGET_COLUMN, pd.Series(dtype=float)), errors="coerce").dropna()
    prediction_date = (
        str((valid_target.index[-1] + pd.DateOffset(months=HORIZON_MONTHS)).date()) if not valid_target.empty else None
    )

    payload = {
        "status": "evaluated",
        "source": "saved_model_files",
        "target": f"{TARGET_COLUMN}_{HORIZON_MONTHS}_months_ahead",
        "horizon_months": HORIZON_MONTHS,
        "prediction_date": prediction_date,
        "latest_overall_health": round(float(valid_target.iloc[-1]), 4) if not valid_target.empty else None,
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
            if not test_frame.empty:
                predictions = _predict_frame(model, test_frame, feature_names, scaler)
                result["metrics"] = _regression_metrics(test_frame["target"].to_numpy(dtype=float), predictions)
        except Exception as exc:
            result.update({"status": "unavailable", "reason": str(exc)})

        payload["models"].append(result)

    return payload
