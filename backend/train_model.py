"""Forecast overall economic health from processed EconPulse scores."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from backend.config import MODEL_HORIZON_MONTHS


FEATURE_LAGS = (1, 3, 6, 12)
ROLLING_WINDOWS = (3, 6, 12)
MIN_TRAINING_ROWS = 36
TEST_SHARE = 0.2


def _student_cost_pressure(scores: pd.DataFrame) -> pd.Series:
    """
    Composite target where higher means students face more pressure.

    Input scores are consumer-friendly, so pressure components are inverted.
    """
    weighted_components = {
        "rent_pressure": 0.35,
        "inflation_pressure": 0.30,
        "borrowing_pressure": 0.20,
        "wage_strength": 0.15,
    }

    pieces = []
    weights = []
    for column, weight in weighted_components.items():
        if column not in scores.columns:
            continue
        pieces.append(100 - pd.to_numeric(scores[column], errors="coerce"))
        weights.append(weight)

    if not pieces:
        return pd.Series(dtype=float, name="student_cost_pressure")

    frame = pd.concat(pieces, axis=1)
    weight_array = np.array(weights, dtype=float)
    weighted_sum = frame.mul(weight_array, axis=1).sum(axis=1, skipna=True)
    available_weight = frame.notna().mul(weight_array, axis=1).sum(axis=1)
    pressure = (weighted_sum / available_weight).where(available_weight >= 0.5)
    return pressure.clip(0, 100).rename("student_cost_pressure")


def build_training_frame(scores: pd.DataFrame, horizon_months: int = MODEL_HORIZON_MONTHS) -> tuple[pd.DataFrame, list[str]]:
    """Create a chronological supervised learning table."""
    if scores.empty:
        return pd.DataFrame(), []

    frame = scores.copy().sort_index()
    frame["student_cost_pressure"] = _student_cost_pressure(frame)

    base_features = [
        column
        for column in [
            "rent_pressure",
            "job_market_strength",
            "inflation_pressure",
            "borrowing_pressure",
            "wage_strength",
            "consumer_sentiment",
            "overall_health",
            "student_cost_pressure",
        ]
        if column in frame.columns
    ]

    feature_frame = frame[base_features].copy()

    for column in base_features:
        source = pd.to_numeric(frame[column], errors="coerce")
        for lag in FEATURE_LAGS:
            feature_frame[f"{column}_lag_{lag}"] = source.shift(lag)
        for window in ROLLING_WINDOWS:
            feature_frame[f"{column}_roll_{window}"] = source.rolling(window).mean()

    feature_frame = feature_frame.ffill(limit=2)
    target = frame["overall_health"].shift(-horizon_months).rename("target")
    training = feature_frame.assign(target=target).dropna()
    return training, [column for column in training.columns if column != "target"]


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    variance = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = None if variance == 0 else float(1 - (np.sum((y_true - y_pred) ** 2) / variance))
    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4) if r2 is not None else None}


def _model_result(
    name: str,
    backend: str,
    y_test: np.ndarray,
    test_predictions: np.ndarray,
    forecast: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        "model": name,
        "backend": backend,
        "metrics": _regression_metrics(y_test, test_predictions),
        "prediction": round(float(np.clip(forecast, 0, 100)), 4),
    }
    if extra:
        result.update(extra)
    return result


def _fit_linear_regression(x_train: np.ndarray, y_train: np.ndarray):
    try:
        from sklearn.linear_model import LinearRegression

        model = LinearRegression()
        model.fit(x_train, y_train)
        return model, "sklearn"
    except Exception:
        design = np.column_stack([np.ones(len(x_train)), x_train])
        coefficients = np.linalg.lstsq(design, y_train, rcond=None)[0]

        class NumpyLinearRegression:
            def __init__(self, coefficients):
                self.coefficients = coefficients

            def predict(self, x):
                return np.column_stack([np.ones(len(x)), x]) @ self.coefficients

        return NumpyLinearRegression(coefficients), "numpy"


def _fit_regularized_models(x_train: np.ndarray, y_train: np.ndarray) -> list[tuple[str, str, Any]]:
    """Fit regularized sklearn models that are safer on small macro datasets."""
    fitted_models: list[tuple[str, str, Any]] = []

    try:
        from sklearn.linear_model import RidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        ridge = make_pipeline(
            StandardScaler(),
            RidgeCV(alphas=np.logspace(-3, 3, 25)),
        )
        ridge.fit(x_train, y_train)
        fitted_models.append(("ridge_regression", "sklearn", ridge))
    except Exception:
        return fitted_models

    return fitted_models


def _feature_subset(feature_columns: list[str], strategy: str) -> list[str]:
    if strategy == "health_autoregressive":
        return [column for column in feature_columns if column.startswith("overall_health")]
    if strategy == "macro_snapshot":
        return [column for column in feature_columns if "_lag_" not in column and "_roll_" not in column]
    return feature_columns


def _train_test_arrays(training: pd.DataFrame, feature_columns: list[str], split_index: int):
    x = training[feature_columns].to_numpy(dtype=float)
    y = training["target"].to_numpy(dtype=float)
    return x[:split_index], x[split_index:], y[:split_index], y[split_index:]


def _walk_forward_metrics(training: pd.DataFrame, feature_columns: list[str], min_train_size: int = 60) -> dict[str, Any]:
    """Evaluate a Ridge model with expanding-window time-series validation."""
    if len(training) <= min_train_size + 6:
        return {"status": "skipped_not_enough_rows"}

    try:
        from sklearn.linear_model import RidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        return {"status": "skipped", "reason": str(exc)}

    predictions = []
    actuals = []
    for test_index in range(min_train_size, len(training)):
        train = training.iloc[:test_index]
        test = training.iloc[test_index : test_index + 1]
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25)))
        model.fit(train[feature_columns].to_numpy(dtype=float), train["target"].to_numpy(dtype=float))
        predictions.append(float(model.predict(test[feature_columns].to_numpy(dtype=float))[0]))
        actuals.append(float(test["target"].iloc[0]))

    return {
        "status": "evaluated",
        "model": "ridge_regression",
        "feature_count": len(feature_columns),
        "initial_train_rows": min_train_size,
        "test_windows": len(actuals),
        "metrics": _regression_metrics(np.array(actuals), np.array(predictions)),
    }


def _forecast_date(series: pd.Series, horizon_months: int) -> str | None:
    valid_values = series.dropna()
    if valid_values.empty:
        return None
    return str((valid_values.index[-1] + pd.DateOffset(months=horizon_months)).date())


def train_models(scores: pd.DataFrame, horizon_months: int = MODEL_HORIZON_MONTHS) -> dict[str, Any]:
    """Train regularized economic-health forecasters."""
    if scores.empty:
        return {
            "target": f"overall_health_{horizon_months}_months_ahead",
            "horizon_months": horizon_months,
            "status": "skipped_empty_scores",
            "models": [],
        }

    working_scores = scores.copy().sort_index()
    working_scores["student_cost_pressure"] = _student_cost_pressure(working_scores)
    training, feature_columns = build_training_frame(working_scores, horizon_months)
    latest_feature_frame, _ = build_training_frame(working_scores, 0)

    if len(training) < MIN_TRAINING_ROWS or not feature_columns:
        return {
            "target": f"overall_health_{horizon_months}_months_ahead",
            "horizon_months": horizon_months,
            "status": "skipped_not_enough_training_data",
            "training_rows": int(len(training)),
            "feature_columns": feature_columns,
            "latest_overall_health": _latest_metric(working_scores, "overall_health"),
            "overall_health_change_1m": _metric_change(working_scores, "overall_health", 1),
            "overall_health_change_3m": _metric_change(working_scores, "overall_health", 3),
            "latest_student_cost_pressure": _latest_pressure(working_scores),
            "student_cost_pressure_change_1m": _pressure_change(working_scores, 1),
            "student_cost_pressure_change_3m": _pressure_change(working_scores, 3),
            "models": [],
        }

    split_index = max(1, int(len(training) * (1 - TEST_SHARE)))
    if split_index >= len(training):
        split_index = len(training) - 1

    latest_features = latest_feature_frame[feature_columns].tail(1)
    if latest_features.empty:
        latest_features = training[feature_columns].tail(1)

    model_results: list[dict[str, Any]] = []
    health_features = _feature_subset(feature_columns, "health_autoregressive")
    macro_features = _feature_subset(feature_columns, "macro_snapshot")

    x_train, x_test, y_train, y_test = _train_test_arrays(training, health_features, split_index)
    latest_x = latest_features[health_features].to_numpy(dtype=float)
    linear_model, linear_backend = _fit_linear_regression(x_train, y_train)
    linear_predictions = linear_model.predict(x_test)
    linear_forecast = float(linear_model.predict(latest_x)[0])
    model_results.append(
        _model_result(
            "linear_regression",
            linear_backend,
            y_test,
            linear_predictions,
            linear_forecast,
            {"feature_set": "health_autoregressive", "feature_count": len(health_features)},
        )
    )

    x_train, x_test, y_train, y_test = _train_test_arrays(training, macro_features, split_index)
    latest_x = latest_features[macro_features].to_numpy(dtype=float)
    for model_name, backend, model in _fit_regularized_models(x_train, y_train):
        predictions = model.predict(x_test)
        forecast = float(model.predict(latest_x)[0])
        model_results.append(
            _model_result(
                model_name,
                backend,
                y_test,
                predictions,
                forecast,
                {"feature_set": "macro_snapshot", "feature_count": len(macro_features)},
            )
        )

    xgboost_available = False
    try:
        from xgboost import XGBRegressor

        xgboost_available = True
        xgb_model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=80,
            max_depth=2,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=10.0,
            random_state=42,
        )
        x_train, x_test, y_train, y_test = _train_test_arrays(training, feature_columns, split_index)
        latest_x = latest_features[feature_columns].to_numpy(dtype=float)
        xgb_model.fit(x_train, y_train)
        xgb_predictions = xgb_model.predict(x_test)
        xgb_forecast = float(xgb_model.predict(latest_x)[0])
        model_results.append(
            _model_result(
                "xgboost_regressor",
                "xgboost",
                y_test,
                xgb_predictions,
                xgb_forecast,
                {"feature_set": "all_lagged_macro_features", "feature_count": len(feature_columns)},
            )
        )
    except Exception as exc:
        model_results.append(
            {
                "model": "xgboost_regressor",
                "status": "skipped",
                "reason": str(exc),
            }
        )

    scored_models = [model for model in model_results if "metrics" in model]
    best = min(scored_models, key=lambda item: item["metrics"]["rmse"]) if scored_models else None

    return {
        "target": f"overall_health_{horizon_months}_months_ahead",
        "horizon_months": horizon_months,
        "status": "trained",
        "validation_strategy": "chronological_holdout_with_walk_forward_ridge_check",
        "training_rows": int(len(training)),
        "test_rows": int(len(y_test)),
        "feature_columns": feature_columns,
        "latest_overall_health": _latest_metric(working_scores, "overall_health"),
        "overall_health_change_1m": _metric_change(working_scores, "overall_health", 1),
        "overall_health_change_3m": _metric_change(working_scores, "overall_health", 3),
        "latest_student_cost_pressure": _latest_pressure(working_scores),
        "student_cost_pressure_change_1m": _pressure_change(working_scores, 1),
        "student_cost_pressure_change_3m": _pressure_change(working_scores, 3),
        "prediction_date": _forecast_date(working_scores["overall_health"], horizon_months),
        "best_model": best["model"] if best else None,
        "best_prediction": best["prediction"] if best else None,
        "walk_forward_validation": _walk_forward_metrics(training, health_features),
        "xgboost_available": xgboost_available,
        "models": model_results,
    }


def _latest_metric(scores: pd.DataFrame, column: str) -> float | None:
    metric = scores.get(column)
    if metric is None:
        return None
    metric = pd.to_numeric(metric, errors="coerce").dropna()
    if metric.empty:
        return None
    return round(float(metric.iloc[-1]), 4)


def _metric_change(scores: pd.DataFrame, column: str, periods: int) -> float | None:
    metric = scores.get(column)
    if metric is None:
        return None
    metric = pd.to_numeric(metric, errors="coerce").dropna()
    if len(metric) <= periods:
        return None
    return round(float(metric.iloc[-1] - metric.iloc[-1 - periods]), 4)


def _latest_pressure(scores: pd.DataFrame) -> float | None:
    pressure = scores.get("student_cost_pressure")
    if pressure is None:
        pressure = _student_cost_pressure(scores)
    pressure = pressure.dropna()
    if pressure.empty:
        return None
    return round(float(pressure.iloc[-1]), 4)


def _pressure_change(scores: pd.DataFrame, periods: int) -> float | None:
    pressure = scores.get("student_cost_pressure")
    if pressure is None:
        pressure = _student_cost_pressure(scores)
    pressure = pressure.dropna()
    if len(pressure) <= periods:
        return None
    return round(float(pressure.iloc[-1] - pressure.iloc[-1 - periods]), 4)
