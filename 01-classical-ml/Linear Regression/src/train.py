"""Train and compare regression models for Mexico City property prices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:  # Support both ``python src/train.py`` and ``python -m src.train``.
    from .preprocess import load_and_clean_data
except ImportError:  # pragma: no cover - exercised by direct script invocation
    from preprocess import load_and_clean_data


FEATURES = ["surface_covered_in_m2", "property_type", "state"]
TARGET = "price_aprox_usd"
MODEL_FILE = "real-estate-model.joblib"
METRICS_FILE = "training-metrics.json"


def build_feature_matrix(data: pd.DataFrame,
                         feature_columns: list[str] | None = None) -> pd.DataFrame:
    """Apply the notebook's categorical baseline and one-hot feature encoding."""
    missing = sorted(set(FEATURES) - set(data.columns))
    if missing:
        raise ValueError(f"Input data is missing required feature columns: {missing}")

    X = data[FEATURES].copy()
    for column in ("property_type", "state"):
        # Match the notebook: the most common level is the reference category.
        if feature_columns is None:
            baseline = X[column].value_counts().idxmax()
            levels = [baseline] + sorted(level for level in X[column].dropna().unique()
                                         if level != baseline)
            X[column] = pd.Categorical(X[column], categories=levels)

    X = pd.get_dummies(
        X, columns=["property_type", "state"], drop_first=True, dtype=float
    )
    if feature_columns is not None:
        X = X.reindex(columns=feature_columns, fill_value=0.0)
    return X


def _load_training_data(data_path: str | Path | None) -> pd.DataFrame:
    if data_path is None:
        return load_and_clean_data()
    return pd.read_csv(data_path)


def _select_modeling_rows(data: pd.DataFrame) -> pd.DataFrame:
    """Apply the notebook's 10th/90th percentile price and area filter."""
    lower_limit, upper_limit = 0.10, 0.90
    price = data[TARGET]
    area = data["surface_covered_in_m2"]
    return data[
        (price > price.quantile(lower_limit))
        & (price < price.quantile(upper_limit))
        & (area > area.quantile(lower_limit))
        & (area < area.quantile(upper_limit))
    ].copy()


def train_model(data_path: str | Path | None = None,
                model_path: str | Path | None = None,
                metrics_path: str | Path | None = None) -> dict:
    """Train the notebook's three models, report metrics, and save the winner.

    The target is fitted on ``log1p(price)`` and metrics are calculated in USD
    after converting predictions back with ``expm1``.
    """
    data = _load_training_data(data_path)
    missing = sorted(set(FEATURES + [TARGET]) - set(data.columns))
    if missing:
        raise ValueError(f"Training data is missing required columns: {missing}")
    data = data.dropna(subset=FEATURES + [TARGET]).copy()
    data = _select_modeling_rows(data)

    X = build_feature_matrix(data)
    feature_columns = list(X.columns)
    y = np.log1p(data[TARGET].astype(float))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    y_train_usd = np.expm1(y_train)
    y_test_usd = np.expm1(y_test)

    models = {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge (CV-tuned)": make_pipeline(
            StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 100))
        ),
        "Lasso (CV-tuned)": make_pipeline(
            StandardScaler(), LassoCV(alphas=np.logspace(-4, 2, 100), max_iter=10000)
        ),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        train_predictions = np.expm1(model.predict(X_train))
        test_predictions = np.expm1(model.predict(X_test))
        results[name] = {
            "train_rmse_usd": float(mean_squared_error(
                y_train_usd, train_predictions
            ) ** 0.5),
            "test_rmse_usd": float(mean_squared_error(
                y_test_usd, test_predictions
            ) ** 0.5),
            "test_r2": float(r2_score(y_test_usd, test_predictions)),
        }

    # As in the notebook's final selection cell, select the model with minimum
    # held-out test RMSE and persist the feature schema with it for inference.
    best_name = min(results, key=lambda name: results[name]["test_rmse_usd"])
    best_model = models[best_name]
    best_alpha = None
    if best_name.startswith("Ridge"):
        best_alpha = float(best_model.named_steps["ridgecv"].alpha_)
    elif best_name.startswith("Lasso"):
        best_alpha = float(best_model.named_steps["lassocv"].alpha_)

    project_root = Path(__file__).resolve().parent.parent
    model_destination = Path(model_path) if model_path else project_root / "src" / MODEL_FILE
    metrics_destination = Path(metrics_path) if metrics_path else project_root / "src" / METRICS_FILE
    model_destination.parent.mkdir(parents=True, exist_ok=True)
    metrics_destination.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": best_model,
        "model_name": best_name,
        "feature_columns": feature_columns,
        "features": FEATURES,
        "target": TARGET,
        "target_transform": "log1p",
    }
    joblib.dump(bundle, model_destination)
    report = {"best_model": best_name, "best_alpha": best_alpha, "models": results}
    metrics_destination.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Model comparison (USD metrics)")
    for name, scores in results.items():
        print(f"{name}: train RMSE=${scores['train_rmse_usd']:,.2f}; "
              f"test RMSE=${scores['test_rmse_usd']:,.2f}; "
              f"test R²={scores['test_r2']:.4f}")
    print(f"Selected: {best_name}")
    print(f"Saved model: {model_destination}")
    print(f"Saved metrics: {metrics_destination}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", help="Clean training CSV; defaults to preprocessing source files")
    parser.add_argument("--model-out", help="Model bundle path")
    parser.add_argument("--metrics-out", help="Metrics JSON path")
    args = parser.parse_args()
    train_model(args.data, args.model_out, args.metrics_out)


if __name__ == "__main__":
    main()
