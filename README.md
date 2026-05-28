![alt text](image-1.png)
# 🔥 Wildfire Risk Intelligence System
> Machine Learning–Driven Decision Support Platform for Early Wildfire Severity Detection

---

# 📌 Project Overview

Wildfires are among the most destructive environmental disasters, causing severe ecological, economic, and human impact worldwide. Rapid detection of dangerous wildfire conditions is critical for improving emergency preparedness and supporting operational decision-making.

This project presents a **Wildfire Risk Intelligence System** powered by Machine Learning. The system predicts wildfire severity risk using environmental, meteorological, and fire-spread indicators through a threshold-optimised XGBoost classification pipeline.

The project was designed not only as a machine learning exercise, but as a deployable environmental intelligence platform capable of supporting wildfire risk assessment workflows.

---

# 🎯 Project Objectives

The main objectives of this project were:

- Build an end-to-end wildfire severity prediction pipeline
- Perform environmental data cleaning and preprocessing
- Engineer meaningful wildfire risk features
- Develop and optimise machine learning models
- Prioritise high-severity wildfire detection
- Implement threshold tuning for operational reliability
- Deploy an interactive decision-support system using Streamlit
- Simulate a real-world wildfire intelligence workflow

---

# 🧠 Key Learning Outcomes

This project provided practical experience in:

## Data Engineering
- Data cleaning and preprocessing
- Environmental data transformation
- SQL database integration
- Feature engineering

## Machine Learning
- Classification modelling
- Model evaluation
- Threshold tuning
- Hyperparameter optimisation
- XGBoost implementation

## Deployment Engineering
- Streamlit application development
- Real-time ML inference
- User interaction design
- Operational dashboard workflows

---

# 🏗️ System Architecture

```text
Environmental Data
        │
        ▼
Data Cleaning & Preprocessing
        │
        ▼
Feature Engineering Pipeline
        │
        ▼
Threshold-Tuned XGBoost Model
        │
        ▼
SQLite Database + Model Storage
        │
        ▼
Streamlit Decision Support Interface
        │
        ▼
Wildfire Severity Risk Assessment

```

## 📂 Project Structure

```bash
Forest-fire-risk-analysis
│
├── app/
│   └── streamlit_app.py
│
├── database/
│   └── forestfireanalysis.db
│
├── models/
│   └── best_model.pkl
│
├── notebooks/
│   ├── 1_data_cleaning.ipynb
│   ├── 2_eda.ipynb
│   └── 3_modeling.ipynb
│
├── requirements.txt
└── README.md
```

## 📂 Dataset Source

This project uses the **Forest Fires Dataset** from the UCI Machine Learning Repository:

🔗 https://archive.ics.uci.edu/dataset/162/forest+fires

The dataset contains meteorological and environmental variables associated with wildfire occurrences in the Montesinho Natural Park region of Portugal. It includes features such as temperature, humidity, wind speed, drought indicators, and fire spread metrics, making it well-suited for wildfire severity prediction and environmental risk analysis.

The dataset was used for:
- environmental feature engineering
- wildfire risk pattern analysis
- machine learning model training
- operational wildfire severity prediction

## 📊 Dataset Features

The system uses multiple environmental and wildfire-related indicators.

| Feature | Description |
|---|---|
| Temperature | Environmental temperature |
| Relative Humidity | Atmospheric moisture level |
| Wind Speed | Wind intensity affecting fire spread |
| Fine Fuel Moisture Code | Surface fuel dryness |
| Duff Moisture Code | Medium-depth fuel moisture |
| Drought Code | Long-term drought severity |
| Initial Spread Index | Fire spread potential |
| Month / Day | Temporal wildfire patterns |
| Geographic Coordinates | Spatial environmental indicators |

## ⚙️ Feature Engineering

Several transformations were applied to improve model performance.

### Cyclical Encoding

Temporal features such as month and day were encoded using sine and cosine transformations to preserve cyclical behaviour.

```python
month_sin = np.sin(2 * np.pi * month_num / 12)
month_cos = np.cos(2 * np.pi * month_num / 12)
```

### Logarithmic Wind Transformation

Wind speed was log-transformed to reduce skewness and improve model stability.

```python
log_wind = np.log1p(wind)
```


## 🔍 Exploratory Data Analysis

EDA revealed several important wildfire risk patterns:

- High temperature strongly correlates with wildfire severity
- Low humidity significantly increases fire spread risk
- Drought severity contributes heavily to dangerous wildfire conditions
- Wind speed accelerates wildfire propagation
- Fire spread metrics improve predictive capability

## 🤖 Machine Learning Pipeline

Multiple classification models were explored during experimentation.

- Baseline model - DummyClassifier
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost Classifier
- MLP

## 🏆 Final Model
### Threshold-Tuned XGBoost Classifier

The final deployed model is a threshold-optimised XGBoost classifier designed to prioritise identification of dangerous wildfire conditions.

Why XGBoost?
Strong performance on structured environmental data
Handles nonlinear feature interactions effectively
Excellent generalisation capability
Robust feature importance analysis

## 🚨 Major Challenge

A major project challenge was discovering that:
- High accuracy alone was insufficient for operational wildfire prediction.
- False negatives were particularly dangerous because missing a severe wildfire event could lead to serious operational consequences.

## 🔥 Threshold Tuning Strategy

To improve detection reliability, threshold tuning was implemented.

Instead of relying on the default classification threshold: 
0.5

The classification threshold was optimised to improve:

- High-severity wildfire recall
- Operational sensitivity
- Dangerous wildfire detection

This significantly improved the model’s ability to identify critical wildfire scenarios.


## 🖥️ Streamlit Deployment

The final model was deployed using Streamlit as an interactive wildfire decision-support application.

### Features

#### Environmental Input Controls

Users can configure:

- temperature
- humidity
- wind speed
- drought conditions
- fire spread indicators

#### Real-Time Prediction

The system generates:

- wildfire severity prediction
- risk probability
- operational risk level

#### Dynamic Alerts

The interface automatically displays:

- extreme temperature warnings
- drought alerts
- fire spread warnings
- high-risk environmental indicators

---



## 🗄️ SQLite Database Integration

SQLite was used for:

- wildfire data management
- local storage
- structured querying
- analysis workflows

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core development |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib / Seaborn | Data visualisation |
| Scikit-learn | ML utilities |
| XGBoost | Final classification model |
| SQLite | Database management |
| Streamlit | Deployment |
| Joblib | Model persistence |

## 🔮 Future Improvements

Potential future enhancements include:

- Real-time weather API integration
- Satellite imagery integration
- Geospatial wildfire mapping
- Live monitoring dashboard
- Cloud deployment
- Explainable AI integration
- Automated retraining pipelines

---

## 🚀 Technical Stack Summary

| Area | Technologies |
|---|---|
| Data Processing | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost |
| Database | SQLite |
| Deployment | Streamlit |
| Model Storage | Joblib |
---

## 🙌 Final Note

This project demonstrates how machine learning can evolve beyond experimentation into a deployable operational intelligence system capable of supporting real-world environmental risk assessment.