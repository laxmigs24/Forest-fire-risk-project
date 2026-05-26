import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="🔥 Wildfire Severity Prediction",
    page_icon="🔥",
    layout="wide"
)

# ---------------------------------------------------
# LOAD TRAINED MODEL
# ---------------------------------------------------

model = joblib.load(
    "models/best_model.pkl"
)

# ---------------------------------------------------
# PAGE TITLE
# ---------------------------------------------------

st.title("🔥 Wildfire Severity Prediction System")

st.markdown("""
This machine learning application predicts wildfire severity
using environmental and meteorological conditions.
""")

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.header("ℹ️ About")

st.sidebar.info("""
This application predicts:

- 🔴 High Severity Fires
- 🟢 Low Severity Fires

using wildfire environmental indicators.
""")

st.sidebar.header("🔥 High Risk Indicators")

st.sidebar.markdown("""
- High temperature
- Low humidity
- Strong winds
- High drought code
""")

# ---------------------------------------------------
# GEOGRAPHIC CONDITIONS
# ---------------------------------------------------

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

# ---------------------------------------------------
# ENVIRONMENTAL CONDITIONS
# ---------------------------------------------------

st.header("🌡️ Environmental Conditions")

col3, col4, col5 = st.columns(3)

with col3:
    temp = st.number_input(
        "Temperature (°C)",
        min_value=-10.0,
        max_value=50.0,
        value=25.0
    )

with col4:
    relative_humidity = st.number_input(
        "Relative Humidity (%)",
        min_value=0,
        max_value=100,
        value=40
    )

with col5:
    wind = st.number_input(
        "Wind Speed (km/h)",
        min_value=0.0,
        max_value=50.0,
        value=4.0
    )

# ---------------------------------------------------
# FUEL CONDITIONS
# ---------------------------------------------------

st.header("🌲 Fuel Conditions")

col6, col7, col8 = st.columns(3)

with col6:
    fine_fuel_moisture_code = st.number_input(
        "Fine Fuel Moisture Code",
        min_value=0.0,
        max_value=100.0,
        value=85.0
    )

with col7:
    duff_moisture_code = st.number_input(
        "Duff Moisture Code",
        min_value=0.0,
        max_value=300.0,
        value=50.0
    )

with col8:
    drought_code = st.number_input(
        "Drought Code",
        min_value=0.0,
        max_value=900.0,
        value=300.0
    )

# ---------------------------------------------------
# FIRE SPREAD CONDITIONS
# ---------------------------------------------------

st.header("🔥 Fire Spread Conditions")

initial_spread_index = st.number_input(
    "Initial Spread Index",
    min_value=0.0,
    max_value=50.0,
    value=5.0
)

# ---------------------------------------------------
# TIME CONDITIONS
# ---------------------------------------------------

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

# ---------------------------------------------------
# MONTH ENCODING
# ---------------------------------------------------

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

# ---------------------------------------------------
# DAY ENCODING
# ---------------------------------------------------

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

# ---------------------------------------------------
# CYCLICAL ENCODING
# ---------------------------------------------------

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

# ---------------------------------------------------
# LOG TRANSFORM WIND
# ---------------------------------------------------

log_wind = np.log1p(wind)

# ---------------------------------------------------
# PREDICT BUTTON
# ---------------------------------------------------

predict_button = st.button(
    "🚀 Predict Wildfire Severity"
)

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------

if predict_button:

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

    # ---------------------------------------------------
    # MODEL PREDICTION
    # ---------------------------------------------------

    prediction = model.predict(
        input_data
    )

    severity = prediction[0]

    # ---------------------------------------------------
    # RESULTS
    # ---------------------------------------------------

    st.divider()

    st.header("🔥 Prediction Result")

    if str(severity).lower() in [
        "high severity",
        "high",
        "1"
    ]:

        st.error(
            "🔴 Predicted Fire Severity: HIGH"
        )

        st.progress(90)

        st.markdown("""
        ### ⚠️ Dangerous Fire Conditions

        Environmental conditions indicate
        elevated wildfire severity risk.
        """)

    else:

        st.success(
            "🟢 Predicted Fire Severity: LOW"
        )

        st.progress(35)

        st.markdown("""
        ### ✅ Lower Fire Severity Conditions

        Current conditions indicate
        relatively lower wildfire severity.
        """)

    # ---------------------------------------------------
    # INPUT SUMMARY
    # ---------------------------------------------------

    st.divider()

    st.subheader("📊 Processed Input Features")

    st.dataframe(
        input_data,
        use_container_width=True
    )

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.divider()

st.caption("""
⚠️ Predictions are based on historical wildfire
environmental data and machine learning modeling.
""")
