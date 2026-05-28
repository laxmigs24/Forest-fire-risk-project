import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="🔥 Wildfire Risk Intelligence System",
    page_icon="🔥",
    layout="wide"
)

# =====================================================
# LOAD MODEL + THRESHOLD
# =====================================================

saved_model = joblib.load(
    "models/best_model.pkl"
)

model = saved_model["model"]

threshold = saved_model["threshold"]

# =====================================================
# PAGE TITLE
# =====================================================

st.title("🔥 Wildfire Risk Intelligence System")

st.markdown("""
This machine learning system predicts wildfire severity risk
using meteorological, environmental, and fire spread indicators.

The deployed model is a *Threshold-Tuned XGBoost Classifier*
optimised to prioritise detection of dangerous wildfire conditions.
""")

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("ℹ️ System Overview")

st.sidebar.info("""
### Prediction Classes

•⁠  ⁠🔴 High Severity
•⁠  ⁠🟢 Low Severity

The model prioritises identifying
dangerous wildfire conditions.
""")

st.sidebar.header("⚙️ Model Configuration")

st.sidebar.write(f"Model Type: Threshold-Tuned XGBoost")

st.sidebar.write(f"Classification Threshold: {threshold}")

st.sidebar.header("🔥 High-Risk Indicators")

st.sidebar.markdown("""
•⁠  ⁠High temperature
•⁠  ⁠Low humidity
•⁠  ⁠Strong winds
•⁠  ⁠High drought code
•⁠  ⁠Rapid fire spread
""")

# =====================================================
# GEOGRAPHIC CONDITIONS
# =====================================================

st.header("📍 Geographic Conditions")

col1, col2 = st.columns(2)

with col1:

    x_coordinate = st.number_input(
        "X Coordinate",
        min_value=1.0,
        max_value=9.0,
        value=5.0
    )

with col2:

    y_coordinate = st.number_input(
        "Y Coordinate",
        min_value=1.0,
        max_value=9.0,
        value=5.0
    )

# =====================================================
# ENVIRONMENTAL CONDITIONS
# =====================================================

st.header("🌡️ Environmental Conditions")

col3, col4, col5 = st.columns(3)

with col3:

    temp = st.slider(
        "Temperature (°C)",
        min_value=-10.0,
        max_value=50.0,
        value=25.0
    )

with col4:

    relative_humidity = st.slider(
        "Relative Humidity (%)",
        min_value=0,
        max_value=100,
        value=40
    )

with col5:

    wind = st.slider(
        "Wind Speed (km/h)",
        min_value=0.0,
        max_value=50.0,
        value=5.0
    )

# =====================================================
# FUEL CONDITIONS
# =====================================================

st.header("🌲 Fuel Conditions")

col6, col7, col8 = st.columns(3)

with col6:

    fine_fuel_moisture_code = st.slider(
        "Fine Fuel Moisture Code",
        min_value=0.0,
        max_value=100.0,
        value=85.0
    )

with col7:

    duff_moisture_code = st.slider(
        "Duff Moisture Code",
        min_value=0.0,
        max_value=300.0,
        value=50.0
    )

with col8:

    drought_code = st.slider(
        "Drought Code",
        min_value=0.0,
        max_value=900.0,
        value=300.0
    )

# =====================================================
# FIRE SPREAD CONDITIONS
# =====================================================

st.header("🔥 Fire Spread Conditions")

initial_spread_index = st.slider(
    "Initial Spread Index",
    min_value=0.0,
    max_value=50.0,
    value=5.0
)

# =====================================================
# TIME CONDITIONS
# =====================================================

st.header("📅 Time Conditions")

col9, col10 = st.columns(2)

with col9:

    month = st.selectbox(
        "Month",
        [
            "jan", "feb", "mar", "apr",
            "may", "jun", "jul", "aug",
            "sep", "oct", "nov", "dec"
        ]
    )

with col10:

    day = st.selectbox(
        "Day of Week",
        [
            "mon", "tue", "wed",
            "thu", "fri", "sat", "sun"
        ]
    )

# =====================================================
# MONTH ENCODING
# =====================================================

month_mapping = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12
}

# =====================================================
# DAY ENCODING
# =====================================================

day_mapping = {
    "mon": 1,
    "tue": 2,
    "wed": 3,
    "thu": 4,
    "fri": 5,
    "sat": 6,
    "sun": 7
}

month_num = month_mapping[month]

day_num = day_mapping[day]

# =====================================================
# CYCLICAL ENCODING
# =====================================================

month_sin = np.sin(
    2 * np.pi * month_num / 12
)

month_cos = np.cos(
    2 * np.pi * month_num / 12
)

day_sin = np.sin(
    2 * np.pi * day_num / 7
)

day_cos = np.cos(
    2 * np.pi * day_num / 7
)

# =====================================================
# LOG TRANSFORM WIND
# =====================================================

log_wind = np.log1p(wind)

# =====================================================
# INPUT WARNINGS
# =====================================================

st.divider()

st.subheader("⚠️ Environmental Alerts")

if temp > 40:
    st.warning("Extreme temperature detected.")

if relative_humidity < 20:
    st.warning("Very low humidity detected.")

if drought_code > 600:
    st.warning("Severe drought conditions detected.")

if wind > 20:
    st.warning("Strong wind conditions detected.")

if initial_spread_index > 15:
    st.warning("Rapid fire spread conditions detected.")

# =====================================================
# PREDICTION BUTTON
# =====================================================

predict_button = st.button(
    "🚀 Predict Wildfire Severity"
)

# =====================================================
# PREDICTION
# =====================================================

if predict_button:

    # =================================================
    # INPUT DATAFRAME
    # =================================================

    input_data = pd.DataFrame({

        "x_coordinate": [x_coordinate],

        "y_coordinate": [y_coordinate],

        "fine_fuel_moisture_code": [
            fine_fuel_moisture_code
        ],

        "duff_moisture_code": [
            duff_moisture_code
        ],

        "drought_code": [
            drought_code
        ],

        "initial_spread_index": [
            initial_spread_index
        ],

        "temp": [temp],

        "relative_humidity": [
            relative_humidity
        ],

        "log_wind": [log_wind],

        "month_sin": [month_sin],

        "month_cos": [month_cos],

        "day_sin": [day_sin],

        "day_cos": [day_cos]

    })

    # =================================================
    # PROBABILITY PREDICTION
    # =================================================

    probability = model.predict_proba(
        input_data
    )[:, 1][0]

    risk_percentage = probability * 100

    # =================================================
    # APPLY THRESHOLD
    # =================================================

    severity = (
        "High Severity"
        if probability >= threshold
        else "Low Severity"
    )

    # =================================================
    # RISK LEVEL
    # =================================================

    if risk_percentage >= 80:

        risk_level = "EXTREME"

    elif risk_percentage >= 60:

        risk_level = "HIGH"

    elif risk_percentage >= 40:

        risk_level = "MODERATE"

    else:

        risk_level = "LOW"

    # =================================================
    # RESULTS
    # =================================================

    st.divider()

    st.header("🔥 Wildfire Risk Assessment")

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    with metric_col1:

        st.metric(
            label="🔥 Risk Probability",
            value=f"{risk_percentage:.1f}%"
        )

    with metric_col2:

        st.metric(
            label="🚨 Risk Level",
            value=risk_level
        )

    with metric_col3:

        st.metric(
            label="⚙️ Threshold",
            value=f"{threshold}"
        )

    # =================================================
    # DYNAMIC PROGRESS BAR
    # =================================================

    st.progress(
        min(int(risk_percentage), 100)
    )

    # =================================================
    # SEVERITY RESULT
    # =================================================

    if severity == "High Severity":

        st.error(
            "🔴 Predicted Wildfire Severity: HIGH"
        )

        st.markdown("""
        ### ⚠️ Dangerous Fire Conditions Detected

        Environmental conditions indicate
        elevated wildfire severity risk.
        """)

        st.markdown("""
        ### Recommended Actions

        - Monitor wildfire alerts continuously
        - Avoid ignition activities
        - Prepare emergency response plans
        - Increase local surveillance
        - Restrict open-fire activities
        """)

    else:

        st.success(
            "🟢 Predicted Wildfire Severity: LOW"
        )

        st.markdown("""
        ### ✅ Lower Wildfire Severity Conditions

        Current environmental conditions
        indicate relatively lower wildfire severity.
        """)

        st.markdown("""
        ### Recommended Actions

        - Continue environmental monitoring
        - Maintain preventive fire measures
        - Monitor changing weather conditions
        """)

    # =================================================
    # KEY RISK INDICATORS
    # =================================================

    st.divider()

    st.subheader("📈 Key Risk Indicators")

    risk_factors = []

    if temp > 30:
        risk_factors.append("High temperature")

    if relative_humidity < 30:
        risk_factors.append("Low humidity")

    if drought_code > 400:
        risk_factors.append("Severe drought conditions")

    if wind > 15:
        risk_factors.append("Strong wind speeds")

    if initial_spread_index > 10:
        risk_factors.append("Rapid fire spread conditions")

    if len(risk_factors) > 0:

        for factor in risk_factors:

            st.warning(f"⚠️ {factor}")

    else:

        st.success(
            "No major wildfire risk indicators detected."
        )

    # =================================================
    # INPUT SUMMARY
    # =================================================

    st.divider()

    st.subheader("📊 Processed Input Features")

    st.dataframe(
        input_data,
        use_container_width=True
    )

# =====================================================
# MODEL INFORMATION
# =====================================================

st.divider()

st.subheader("🤖 Model Information")

st.markdown(f"""
•⁠  ⁠*Model Type:* Threshold-Tuned XGBoost  
•⁠  ⁠*Classification Threshold:* {threshold}  
•⁠  ⁠*Objective:* Prioritise dangerous wildfire detection  
•⁠  ⁠*Optimisation Goal:* Maximise High Severity Recall  
•⁠  ⁠*Deployment Type:* Wildfire Risk Decision Support System  
""")

# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption("""
⚠️ Predictions are generated using historical wildfire
environmental data and machine learning modelling.

This system is intended for decision-support purposes
and should not replace official wildfire emergency systems.
""")