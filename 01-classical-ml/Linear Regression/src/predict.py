"""Load a trained regression bundle and predict property prices from a CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:  # Support both ``python src/predict.py`` and ``python -m src.predict``.
    from .train import MODEL_FILE, build_feature_matrix
except ImportError:  # pragma: no cover - exercised by direct script invocation
    from train import MODEL_FILE, build_feature_matrix


def predict(data: pd.DataFrame, model_path: str | Path | None = None) -> pd.DataFrame:
    """Return input rows with a ``predicted_price_aprox_usd`` column added.

    ``data`` must contain ``surface_covered_in_m2``, ``property_type``, and
    ``state``. Category levels not seen during training receive zero values in
    the saved one-hot feature schema.
    """
    project_root = Path(__file__).resolve().parent.parent
    bundle_path = Path(model_path) if model_path else project_root / "src" / MODEL_FILE
    if not bundle_path.is_file():
        raise FileNotFoundError(
            f"Model bundle not found at {bundle_path}; run src/train.py first"
        )

    bundle = joblib.load(bundle_path)
    features = build_feature_matrix(data, feature_columns=bundle["feature_columns"])
    predicted_log_price = bundle["model"].predict(features)
    result = data.copy()
    result["predicted_price_aprox_usd"] = np.expm1(predicted_log_price)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", help="CSV with surface, property_type, and state columns")
    parser.add_argument("--model", help="Saved model bundle path")
    parser.add_argument("--output", help="Output CSV path; defaults to <input>-predictions.csv")
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    data = pd.read_csv(input_path)
    predictions = predict(data, args.model)
    output_path = Path(args.output) if args.output else input_path.with_name(
        f"{input_path.stem}-predictions.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    print(f"Saved {len(predictions):,} predictions to {output_path}")


if __name__ == "__main__":
    main()
