"""Flask API for the Mexico City real-estate price model."""

from __future__ import annotations

import math

from flask import Flask, jsonify, request

try:  # Support both ``python src/app.py`` and ``python -m src.app``.
    from .predict import predict
except ImportError:  # pragma: no cover - exercised by direct script invocation
    from predict import predict


app = Flask(__name__)
REQUIRED_FIELDS = ("surface_covered_in_m2", "property_type", "state")


@app.get("/health")
def health():
    """Simple liveness endpoint for local use and deployment platforms."""
    return jsonify(status="ok")


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Handle both browser checks (GET) and model predictions (POST)."""
    if request.method == 'GET':
        return "This is a machine learning API endpoint. Send a POST request with JSON data to get a prediction!"

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="Request body must be a JSON object."), 400

    properties = payload.get("properties", [payload])
    if not isinstance(properties, list) or not properties:
        return jsonify(error="'properties' must be a non-empty array."), 400

    for index, item in enumerate(properties):
        if not isinstance(item, dict):
            return jsonify(error=f"Property at index {index} must be an object."), 400
        missing = [field for field in REQUIRED_FIELDS if field not in item]
        if missing:
            return jsonify(error=f"Property at index {index} is missing: {', '.join(missing)}."), 400
        try:
            area = float(item["surface_covered_in_m2"])
        except (TypeError, ValueError):
            return jsonify(error=f"Property at index {index} has an invalid surface_covered_in_m2."), 400
        if not math.isfinite(area) or area <= 0:
            return jsonify(error=f"Property at index {index} surface_covered_in_m2 must be a positive number."), 400
        if not isinstance(item["property_type"], str) or not item["property_type"].strip():
            return jsonify(error=f"Property at index {index} property_type must be a non-empty string."), 400
        if not isinstance(item["state"], str) or not item["state"].strip():
            return jsonify(error=f"Property at index {index} state must be a non-empty string."), 400
        item["surface_covered_in_m2"] = area

    try:
        import pandas as pd

        predictions = predict(pd.DataFrame(properties))
    except FileNotFoundError as error:
        app.logger.exception("Model bundle is missing")
        return jsonify(error="Model bundle is unavailable. Train the model first.", detail=str(error)), 503
    except (ValueError, KeyError) as error:
        return jsonify(error=str(error)), 400

    results = [
        {"predicted_price_aprox_usd": float(value)}
        for value in predictions["predicted_price_aprox_usd"]
    ]
    if "properties" in payload:
        return jsonify(predictions=results)
    return jsonify(results[0])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
