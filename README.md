# Flight Delay Prediction - MSIS 522 Homework 1

## Overview
End-to-end data science workflow for predicting US domestic flight delays using 2024 BTS flight data (1M+ records). Includes EDA, five classification models (Logistic Regression, Decision Tree, Random Forest, XGBoost, MLP), SHAP explainability, and an interactive Streamlit app.

## Project Structure
```
flight-delay-project/
├── analysis.py              # Full analysis pipeline (EDA + modeling + SHAP)
├── streamlit_app.py         # Streamlit web application
├── flight_data_2024.csv     # Dataset (2024 US domestic flights)
├── requirements.txt         # Python dependencies
├── models/                  # Saved models and metadata
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── random_forest.joblib
│   ├── xgboost.joblib
│   ├── mlp_model.keras
│   ├── scaler.joblib
│   ├── feature_names.joblib
│   ├── top_states.joblib
│   ├── best_params.json
│   ├── model_comparison.csv
│   ├── shap_info.json
│   └── stats.json
└── figures/                 # Generated visualizations
```

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the analysis (generates models and figures)
```bash
python analysis.py
```

### 3. Launch the Streamlit app locally
```bash
streamlit run streamlit_app.py
```

### 4. Deployed App
The app is deployed at: [Streamlit Community Cloud link]

## Dataset
- **Source:** Bureau of Transportation Statistics (BTS)
- **Period:** January – February 2024
- **Records:** 1,048,575 flights
- **Target:** Binary classification — delayed (weather or late aircraft delay) vs. not delayed

## Author
Fardeen Meeran — University of Washington, MSIS 522
