import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:5000")

st.title("🇲🇽 Mexico City Real Estate Price Predictor")
st.write("Enter the property details below to estimate its price in USD.")

@st.cache_data
def get_options():
    r = requests.get(f"{API_URL}/options", timeout=10)
    r.raise_for_status()
    return r.json()

try:
    category_levels = get_options()
except requests.exceptions.RequestException as e:
    st.error(f"Could not reach the API: {e}")
    st.stop()

surface_covered_in_m2 = st.number_input(
    "Surface Covered (m²)", min_value=1.0, max_value=1000.0, value=100.0, step=1.0
)
property_type = st.selectbox("Property Type", options=category_levels["property_type"])
state = st.selectbox("State / Region", options=category_levels["state"])

if st.button("Predict Price"):
    payload = {
        "surface_covered_in_m2": surface_covered_in_m2,
        "property_type": property_type,
        "state": state,
    }
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        r.raise_for_status()
        result = r.json()

        predicted_price = float(result["predicted_price_aprox_usd"])
        st.success(
            f"Estimated Property Price: **${predicted_price:,.2f} USD**"
        )
    except requests.exceptions.RequestException as e:
        st.error(f"API error: {e}")
    except (KeyError, ValueError) as e:
        st.error(f"Unexpected response format: {e}")