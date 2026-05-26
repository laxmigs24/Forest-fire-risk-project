import streamlit as st
import joblib
import pandas as pd
# Page configuration
st.set_page_config(
    page_title="Wildfire Risk Estimation",
    page_icon="🔥",
    layout="centered"
)
# Load the trained model
model = joblib.load("models/best_mlp_model.pkl")

scaler = joblib.load("models/feature_scaler.pkl")

feature_columns = joblib.load("models/feature_columns.pkl")

label_encoder = joblib.load("models/mlp_label_encoder.pkl")
# App title
st.title("🔥 Wildfire Risk Estimation System")

st.write("Machine Learning based wildfire risk prediction application.")
st.header("Enter Wildfire Conditions")
# Create numeric inputs
ffmc = st.number_input(
    "Dryness of Surface Fuel (%)",
    min_value=0.0,
    max_value=100.0,
    value=85.0
)

dmc = st.number_input(
    "Dryness of Forest Floor (%)",
    min_value=0.0,
    max_value=100.0,
    value=26.0
)

dc = st.number_input(
    "Drought Severity (%)",
    min_value=0.0,
    max_value=100.0,
    value=94.0
)

isi = st.number_input(
    "Fire Spread Potential (km/h)",
    min_value=0.0,
    max_value=100.0,
    value=5.0
)

temp = st.number_input(
    "Temperature (°C)",
    value=18.0
)

rh = st.number_input(
    "Relative Humidity (%)",
    min_value=0,
    max_value=100,
    value=40
)

wind = st.number_input(
    "Wind Speed (km/h)",
    min_value=0.0,
    value=4.0
)

rain = st.number_input(
    "Rain (mm/m²)",
    min_value=0.0,
    value=0.0
)

# Create month drop down
month = st.selectbox(
    "Month",
    [
        "jan","feb","mar","apr","may","jun",
        "jul","aug","sep","oct","nov","dec"
    ]
)
# Create day drop down
day = st.selectbox(
    "Day",
    [
        "mon","tue","wed",
        "thu","fri","sat","sun"
    ]
)
# Add prediction button
predict_button = st.button("Predict Wildfire Risk")
if predict_button:

    # Create input dataframe
    input_data = pd.DataFrame({
        "FFMC": [ffmc],
        "DMC": [dmc],
        "DC": [dc],
        "ISI": [isi],
        "temp": [temp],
        "RH": [rh],
        "wind": [wind],
        "rain": [rain],
        "month": [month],
        "day": [day]
    })

    # One-hot encoding
    input_data = pd.get_dummies(
        input_data,
        columns=["month", "day"]
    )

    # Align columns
    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Scale
    scaled_input = scaler.transform(input_data)

    # Predict
    prediction = model.predict(scaled_input)

    # Decode label
    predicted_label = label_encoder.inverse_transform(prediction)

    risk = predicted_label[0]

    # Risk display
    if risk == "Low":
        st.success(f"🟢 Predicted Wildfire Risk: {risk}")

    elif risk == "Moderate":
        st.warning(f"🟡 Predicted Wildfire Risk: {risk}")

    elif risk == "High":
        st.warning(f"🟠 Predicted Wildfire Risk: {risk}")

    elif risk == "Extreme":
        st.error(f"🔴 Predicted Wildfire Risk: {risk}")

    st.caption(
        "⚠️ Prediction based on historical wildfire weather conditions."
    )









