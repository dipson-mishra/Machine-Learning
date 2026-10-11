"""Fit and evaluate the KNN classifier using the prepared CSV splits."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
TARGET = "class"


def load_split(name: str) -> tuple[pd.DataFrame, pd.Series]:
    path = DATA_DIR / f"{name}.csv"
    frame = pd.read_csv(path)
    if TARGET not in frame:
        raise ValueError(f"{path} must contain a '{TARGET}' column.")
    return frame.drop(columns=TARGET), frame[TARGET]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neighbors", type=int, default=10, help="Number of neighbors (default: 10).")
    parser.add_argument(
        "--random-state", type=int, default=42, help="Seed for reproducible oversampling (default: 42)."
    )
    args = parser.parse_args()
    if args.neighbors < 1:
        parser.error("--neighbors must be at least 1")

    x_train, y_train = load_split("train")
    x_valid, y_valid = load_split("valid")
    x_test, y_test = load_split("test")
    if not (list(x_train.columns) == list(x_valid.columns) == list(x_test.columns)):
        raise ValueError("Train, validation, and test feature columns must match in order.")

    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("oversample", RandomOverSampler(random_state=args.random_state)),
            ("knn", KNeighborsClassifier(n_neighbors=args.neighbors)),
        ]
    )
    model.fit(x_train, y_train)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "knn.joblib")
    reports = {}
    for name, features, labels in (
        ("validation", x_valid, y_valid),
        ("test", x_test, y_test),
    ):
        predictions = model.predict(features)
        report = classification_report(
            labels,
            predictions,
            target_names=["hadron (0)", "gamma (1)"],
            output_dict=True,
            zero_division=0,
        )
        reports[name] = report
        print(f"\n{name.title()} set")
        print(classification_report(
            labels,
            predictions,
            target_names=["hadron (0)", "gamma (1)"],
            zero_division=0,
        ))

    (MODEL_DIR / "metrics.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"Saved model and metrics in {MODEL_DIR}")


if __name__ == "__main__":
    main()
