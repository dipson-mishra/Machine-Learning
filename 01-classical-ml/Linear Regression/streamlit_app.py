import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from src.predict import predict

MODEL_BUNDLE = Path(__file__).parent / "src" / "real-estate-model.joblib"
bundle = joblib.load(MODEL_BUNDLE)
category_levels = bundle["category_levels"]

st.title("🇲🇽 Mexico City Real Estate Price Predictor")
st.write("Enter the property details below to estimate its price in USD.")
st.caption(
    f"Model: {bundle['model_name']} using covered area, property type, and state/region."
)

# Create input widgets for your features
surface_covered_in_m2 = st.number_input(
    "Surface Covered (m²)", min_value=1.0, max_value=1000.0, value=100.0, step=1.0
)

property_type = st.selectbox(
    "Property Type", options=category_levels["property_type"]
)

state = st.selectbox(
    "State / Region", options=category_levels["state"]
)

# Button to trigger prediction
if st.button("Predict Price"):
    # Build a DataFrame matching what your model expects
    input_data = pd.DataFrame([{
        "surface_covered_in_m2": surface_covered_in_m2,
        "property_type": property_type,
        "state": state
    }])

    try:
        # Run prediction using your existing model pipeline
        prediction_df = predict(input_data)
        predicted_price = float(prediction_df["predicted_price_aprox_usd"].iloc[0])

        # Display the result nicely
        st.success(f"Estimated Property Price: **${predicted_price:,.2f} USD**")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
