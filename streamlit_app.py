"""
MSIS 522 - Homework 1: Flight Delay Prediction
Streamlit Application
Author: Fardeen Meeran
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Flight Delay Prediction - MSIS 522",
    page_icon="✈️",
    layout="wide"
)

# ============================================================
# LOAD ARTIFACTS
# ============================================================

@st.cache_data
def load_data():
    df = pd.read_csv('flight_data_2024.csv')
    df['delayed'] = ((df['weather_delay'] > 0) | (df['late_aircraft_delay'] > 0)).astype(int)
    df['dep_hour'] = (df['dep_time'] // 100).clip(0, 23)
    df_clean = df.dropna(subset=['dep_time', 'taxi_out', 'taxi_in']).copy()
    return df, df_clean

@st.cache_resource
def load_models():
    models = {}
    model_files = {
        'Logistic Regression': 'models/logistic_regression.joblib',
        'Decision Tree': 'models/decision_tree.joblib',
        'Random Forest': 'models/random_forest.joblib',
        'XGBoost': 'models/xgboost.joblib'
    }
    for name, path in model_files.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)

    # Load MLP (TensorFlow may not be available on all platforms)
    try:
        if os.path.exists('models/mlp_model.keras'):
            import tensorflow as tf
            models['Neural Network (MLP)'] = tf.keras.models.load_model('models/mlp_model.keras')
    except ImportError:
        pass  # TF not available in this environment

    return models

@st.cache_resource
def load_scaler_and_meta():
    scaler = joblib.load('models/scaler.joblib')
    feature_names = joblib.load('models/feature_names.joblib')
    top_states = joblib.load('models/top_states.joblib')
    with open('models/best_params.json') as f:
        best_params = json.load(f)
    with open('models/stats.json') as f:
        stats = json.load(f)
    with open('models/shap_info.json') as f:
        shap_info = json.load(f)
    metrics_df = pd.read_csv('models/model_comparison.csv')
    return scaler, feature_names, top_states, best_params, stats, shap_info, metrics_df

# Load everything
df_raw, df_clean = load_data()
models = load_models()
scaler, feature_names, top_states, best_params, stats, shap_info, metrics_df = load_scaler_and_meta()

# ============================================================
# APP LAYOUT
# ============================================================

st.title("✈️ Flight Delay Prediction - 2024 US Domestic Flights")
st.markdown("**MSIS 522 - Analytics and Machine Learning | University of Washington**")

tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Executive Summary",
    "📊 Descriptive Analytics",
    "🤖 Model Performance",
    "🔍 Explainability & Prediction"
])

# ============================================================
# TAB 1: EXECUTIVE SUMMARY
# ============================================================
with tab1:
    st.header("Executive Summary")

    st.subheader("Dataset & Prediction Task")
    st.markdown(f"""
    This project analyzes **{stats['total_rows']:,} US domestic flight records from January–February 2024**,
    sourced from the Bureau of Transportation Statistics (BTS). Each record represents a single domestic flight
    and includes information about the flight date, origin airport and state, departure time, taxi times,
    flight distance, and delay breakdowns (weather delays and late aircraft delays). The dataset covers
    **{stats['num_origins']} unique origin airports** across **{stats['num_states']} US states and territories**,
    providing a comprehensive snapshot of early-2024 air travel patterns.

    The prediction target is a **binary classification variable** indicating whether a flight experienced a
    significant delay — specifically, whether it was affected by a weather delay or a late-arriving aircraft
    delay (the two most common causes of delay beyond airline control). A flight is labeled as "delayed" if
    either `weather_delay > 0` or `late_aircraft_delay > 0`. Out of all flights in the dataset, approximately
    **{stats['delay_rate']*100:.1f}%** experienced such a delay, making this a moderately imbalanced classification problem.
    """)

    st.subheader("Why This Matters")
    st.markdown("""
    Flight delays cost the US economy an estimated **$33 billion annually** in lost productivity, missed
    connections, and operational disruptions. For airlines, delays translate to increased fuel costs, crew
    overtime, and customer dissatisfaction. For passengers, even a 15-minute delay can cascade into missed
    connections and hours of wasted time. Understanding which factors — time of day, day of week, departure
    airport, flight distance — contribute most to delays allows airlines to proactively adjust scheduling,
    enables airports to optimize resource allocation during high-risk periods, and helps travelers make more
    informed booking decisions. This problem sits at the intersection of operational efficiency and customer
    experience, making it a high-impact use case for predictive analytics.
    """)

    st.subheader("Approach & Key Findings")
    st.markdown(f"""
    The analysis follows a complete data science workflow: exploratory data analysis (EDA), predictive modeling
    with five different algorithms, model explainability using SHAP, and interactive deployment via this
    Streamlit application.

    Five models were trained and compared: **Logistic Regression** (baseline), **Decision Tree**, **Random Forest**,
    **XGBoost**, and a **Neural Network (MLP)**. All models used class weighting to handle the ~10:1 class
    imbalance, and tree-based models were tuned via 5-fold cross-validated grid search. The best-performing
    model was **{metrics_df.loc[metrics_df['f1'].idxmax(), 'model']}** with an F1 score of
    **{metrics_df['f1'].max():.3f}** and AUC-ROC of **{metrics_df.loc[metrics_df['f1'].idxmax(), 'auc_roc']:.3f}**.

    SHAP analysis revealed that the most influential features for predicting delays include departure hour,
    distance, and certain origin states — confirming the intuition that late-evening flights and flights from
    specific hub airports are at higher risk. These insights are actionable: airlines could allocate extra
    buffer time for high-risk routes, and travelers could prioritize morning departures to minimize delay risk.
    """)

    # Quick stats cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Flights", f"{stats['total_rows']:,}")
    col2.metric("Features Used", f"{len(feature_names)}")
    col3.metric("Delay Rate", f"{stats['delay_rate']*100:.1f}%")
    best_f1 = metrics_df['f1'].max()
    col4.metric("Best F1 Score", f"{best_f1:.3f}")


# ============================================================
# TAB 2: DESCRIPTIVE ANALYTICS
# ============================================================
with tab2:
    st.header("Descriptive Analytics")

    # 1.1 Dataset Introduction
    st.subheader("1.1 Dataset Overview")
    st.markdown(f"""
    | Property | Value |
    |----------|-------|
    | **Source** | Bureau of Transportation Statistics (BTS) |
    | **Time Period** | January – February 2024 |
    | **Total Records** | {stats['total_rows']:,} |
    | **Features** | {stats['total_columns']} columns |
    | **Numerical Features** | {', '.join(stats['feature_types']['numerical'])} |
    | **Categorical Features** | {', '.join(stats['feature_types']['categorical'])} + origin, origin_city_name |
    | **Target Variable** | `delayed` (binary: 1 = flight delayed, 0 = not delayed) |
    | **Unique Airports** | {stats['num_origins']} |
    | **US States/Territories** | {stats['num_states']} |
    """)

    # 1.2 Target Distribution
    st.subheader("1.2 Target Distribution")
    if os.path.exists('figures/target_distribution.png'):
        st.image('figures/target_distribution.png', use_container_width=True)
    st.markdown(f"""
    The target variable is **heavily imbalanced**: only **{stats['delay_rate']*100:.1f}%** of flights are
    classified as delayed. This roughly 10:1 ratio between non-delayed and delayed flights means that a naive
    classifier predicting "not delayed" for every flight would achieve ~{(1-stats['delay_rate'])*100:.0f}% accuracy —
    which is misleading. To address this imbalance, all models use **class weighting** (via `class_weight='balanced'`
    or `scale_pos_weight`) to ensure the minority class (delayed flights) receives proportionally higher
    importance during training. Evaluation focuses on **F1 score** and **AUC-ROC** rather than accuracy alone.
    """)

    # 1.3 Feature Visualizations
    st.subheader("1.3 Feature Distributions & Relationships")

    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists('figures/delay_by_month.png'):
            st.image('figures/delay_by_month.png', use_container_width=True)
            st.markdown("""
            **Delay Rate by Month:** The delay rate varies between January and February 2024.
            Winter weather patterns and holiday travel surges in January tend to create more operational
            disruptions, while February typically shows slightly different patterns as winter storms shift.
            """)
    with col2:
        if os.path.exists('figures/delay_by_dow.png'):
            st.image('figures/delay_by_dow.png', use_container_width=True)
            st.markdown("""
            **Delay Rate by Day of Week:** Mid-week days and weekends show different delay patterns.
            Days with heavier business travel (Tuesday–Thursday) may see cascading delays due to
            tighter scheduling, while weekends may benefit from reduced traffic or suffer from
            reduced staffing depending on the airport.
            """)

    col3, col4 = st.columns(2)
    with col3:
        if os.path.exists('figures/delay_by_hour.png'):
            st.image('figures/delay_by_hour.png', use_container_width=True)
            st.markdown("""
            **Delay Rate by Departure Hour:** This is one of the most revealing patterns. Early morning
            flights (5–8 AM) have the lowest delay rates because aircraft are starting fresh. As the day
            progresses, delays compound — a late-arriving aircraft from a morning route causes cascading
            delays throughout the day. Evening flights (after 6 PM) consistently show the highest delay rates.
            """)
    with col4:
        if os.path.exists('figures/distance_by_delay.png'):
            st.image('figures/distance_by_delay.png', use_container_width=True)
            st.markdown("""
            **Distance Distribution by Delay Status:** The distribution of flight distances is similar
            for both delayed and non-delayed flights, with most flights being short- to medium-haul
            (under 1,500 miles). However, there are subtle differences in the tails — very short flights
            and very long flights may have slightly different delay profiles due to turnaround times
            and routing complexity.
            """)

    if os.path.exists('figures/delay_by_state.png'):
        st.image('figures/delay_by_state.png', use_container_width=True)
        st.markdown("""
        **Top States by Delay Rate:** Among states with at least 5,000 flights, certain states stand out
        with higher delay rates. States with major hub airports, severe winter weather, or high air traffic
        congestion tend to appear at the top. This geographic variation suggests that origin location is a
        meaningful predictor of flight delays.
        """)

    # 1.4 Correlation Heatmap
    st.subheader("1.4 Correlation Heatmap")
    if os.path.exists('figures/correlation_heatmap.png'):
        st.image('figures/correlation_heatmap.png', use_container_width=True)
    st.markdown("""
    The correlation heatmap reveals several notable relationships. The `weather_delay` and `late_aircraft_delay`
    features show moderate positive correlation with the `delayed` target, which is expected since they directly
    define it. Among the potential predictor features, `taxi_out` and `taxi_in` times show some correlation with
    each other (both reflect airport congestion). `dep_hour` shows a slight positive correlation with delay —
    consistent with the pattern that later flights are more likely to be delayed. Distance shows relatively weak
    correlation with delay, suggesting that delay risk is driven more by timing and location than by route length.
    """)


# ============================================================
# TAB 3: MODEL PERFORMANCE
# ============================================================
with tab3:
    st.header("Model Performance Comparison")

    # Model comparison table
    st.subheader("2.7 Model Comparison Summary")
    st.dataframe(
        metrics_df.style.highlight_max(subset=['accuracy', 'precision', 'recall', 'f1', 'auc_roc'],
                                        color='lightgreen'),
        use_container_width=True
    )

    # F1 comparison bar chart
    if os.path.exists('figures/model_comparison_f1.png'):
        st.image('figures/model_comparison_f1.png', use_container_width=True)

    # Best hyperparameters
    st.subheader("Best Hyperparameters")
    for model_name, params in best_params.items():
        with st.expander(f"**{model_name.replace('_', ' ').title()}**"):
            st.json(params)

    # ROC Curves
    st.subheader("ROC Curves")
    if os.path.exists('figures/roc_curves_all.png'):
        st.image('figures/roc_curves_all.png', use_container_width=True)
    st.markdown("""
    The ROC curves show each model's trade-off between true positive rate (correctly identifying delayed flights)
    and false positive rate (incorrectly flagging non-delayed flights). A higher AUC indicates better overall
    discriminative ability. Models closer to the top-left corner perform better at distinguishing delayed from
    non-delayed flights across all threshold settings.
    """)

    # Decision Tree visualization
    st.subheader("Decision Tree Visualization")
    if os.path.exists('figures/decision_tree.png'):
        st.image('figures/decision_tree.png', use_container_width=True)
        st.markdown("""
        The decision tree (shown at top 3 levels for readability) reveals the splits the model considers
        most informative. The root node typically splits on the feature with the highest information gain —
        often departure hour or a state indicator — confirming the importance of timing and geography.
        """)

    # MLP Training History
    st.subheader("Neural Network Training History")
    if os.path.exists('figures/mlp_training_history.png'):
        st.image('figures/mlp_training_history.png', use_container_width=True)
        st.markdown("""
        The loss and accuracy curves show the MLP's learning progress over 50 epochs. The gap between training
        and validation curves indicates the degree of overfitting. The model uses dropout layers (rate=0.3)
        to regularize and prevent excessive overfitting to the training data.
        """)

    # Model comparison commentary
    st.subheader("Analysis")
    best_model_name = metrics_df.loc[metrics_df['f1'].idxmax(), 'model']
    st.markdown(f"""
    **Best Model: {best_model_name}** achieved the highest F1 score, making it the best at balancing precision
    and recall for this imbalanced dataset. The ensemble and boosting methods (Random Forest, XGBoost) generally
    outperform the simpler models (Logistic Regression, single Decision Tree), which is expected given their
    ability to capture complex, non-linear interactions between features.

    **Trade-offs:** Logistic Regression, while the weakest performer, offers full interpretability — each coefficient
    directly shows a feature's contribution. The Decision Tree is also interpretable and visualizable but tends to
    overfit. Random Forest and XGBoost sacrifice some interpretability for better predictive power, though SHAP
    analysis (see next tab) helps recover interpretability. The Neural Network (MLP) provides competitive
    performance but is the least interpretable and most computationally expensive to train.
    """)


# ============================================================
# TAB 4: EXPLAINABILITY & INTERACTIVE PREDICTION
# ============================================================
with tab4:
    st.header("Explainability & Interactive Prediction")

    # SHAP Plots
    st.subheader("3.1 SHAP Analysis")
    st.markdown(f"**Model used:** {shap_info['best_tree_model']} (best-performing tree-based model)")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**SHAP Summary Plot (Beeswarm)**")
        if os.path.exists('figures/shap_summary.png'):
            st.image('figures/shap_summary.png', use_container_width=True)
    with col2:
        st.markdown("**SHAP Feature Importance (Bar)**")
        if os.path.exists('figures/shap_bar.png'):
            st.image('figures/shap_bar.png', use_container_width=True)

    st.markdown("""
    **Interpretation:** The SHAP beeswarm plot shows each feature's impact on individual predictions. Each dot
    represents one flight — its position along the x-axis shows how much that feature pushed the prediction toward
    "delayed" (right) or "not delayed" (left), while the color indicates the feature's actual value (red = high,
    blue = low). The bar plot ranks features by their average absolute SHAP value across all predictions.

    **Key insights:**
    - **Departure hour** is consistently the most important predictor — later departure times strongly push predictions
    toward "delayed," confirming the cascading delay effect throughout the day.
    - **Distance** and **day of week** also play important roles, though with more nuanced effects.
    - **Origin state indicators** for specific states appear in the top features, reflecting geographic hotspots for delays.
    - These findings are actionable: travelers should book morning flights to minimize delay risk, and airlines
    should allocate extra scheduling buffer for evening departures from high-delay-rate airports.
    """)

    if os.path.exists('figures/shap_waterfall.png'):
        st.subheader("SHAP Waterfall Plot (Example Delayed Flight)")
        st.image('figures/shap_waterfall.png', use_container_width=True)
        st.markdown("""
        The waterfall plot breaks down a single prediction — showing exactly how each feature contributed to
        the model's output for one specific delayed flight. Starting from the base value (average prediction),
        each feature either pushes the prediction higher (toward "delayed") or lower (toward "not delayed").
        """)

    # ============================================================
    # INTERACTIVE PREDICTION
    # ============================================================
    st.subheader("Interactive Prediction Tool")
    st.markdown("Set the flight parameters below and see the predicted delay probability in real time.")

    # Model selector
    available_models = [m for m in ['Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost', 'Neural Network (MLP)']
                        if m in models]
    selected_model_name = st.selectbox("Select Model", available_models, index=available_models.index('XGBoost') if 'XGBoost' in available_models else 0)

    col1, col2, col3 = st.columns(3)
    with col1:
        month = st.selectbox("Month", [1, 2], format_func=lambda x: "January" if x == 1 else "February")
        day_of_month = st.slider("Day of Month", 1, 31, 15)
        day_of_week = st.selectbox("Day of Week", list(range(1, 8)),
                                   format_func=lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][x-1])
    with col2:
        dep_hour = st.slider("Departure Hour (0-23)", 0, 23, 12)
        distance = st.slider("Distance (miles)", 50, 3000, 500, step=50)
    with col3:
        state = st.selectbox("Origin State", sorted(top_states + ['Other']))

    # Build feature vector
    input_data = {feat: 0.0 for feat in feature_names}
    input_data['month'] = month
    input_data['day_of_month'] = day_of_month
    input_data['day_of_week'] = day_of_week
    input_data['dep_hour'] = dep_hour
    input_data['distance'] = distance

    # Set state one-hot
    state_col = f'state_{state}'
    if state_col in input_data:
        input_data[state_col] = 1.0

    input_df = pd.DataFrame([input_data])[feature_names]

    # Predict
    selected_model = models[selected_model_name]

    if selected_model_name in ['Logistic Regression', 'Neural Network (MLP)']:
        input_scaled = scaler.transform(input_df)
        if selected_model_name == 'Neural Network (MLP)':
            proba = float(selected_model.predict(input_scaled, verbose=0)[0][0])
        else:
            proba = float(selected_model.predict_proba(input_scaled)[0][1])
    else:
        proba = float(selected_model.predict_proba(input_df)[0][1])

    prediction = "Delayed" if proba >= 0.5 else "Not Delayed"

    # Display prediction
    st.markdown("---")
    pcol1, pcol2, pcol3 = st.columns(3)
    pcol1.metric("Prediction", prediction)
    pcol2.metric("Delay Probability", f"{proba*100:.1f}%")
    pcol3.metric("Confidence", f"{max(proba, 1-proba)*100:.1f}%")

    # Progress bar for probability
    st.progress(min(proba, 1.0))

    # SHAP waterfall for custom input
    if selected_model_name in ['Random Forest', 'XGBoost', 'Decision Tree']:
        try:
            import shap
            explainer = shap.TreeExplainer(selected_model)
            shap_vals = explainer.shap_values(input_df)

            if isinstance(shap_vals, list):
                sv = shap_vals[1][0]
                base = explainer.expected_value[1]
            else:
                sv = shap_vals[0]
                base = explainer.expected_value
                if isinstance(base, np.ndarray):
                    base = base[0]

            explanation = shap.Explanation(
                values=sv,
                base_values=float(base),
                data=input_df.iloc[0].values,
                feature_names=feature_names
            )

            st.subheader("SHAP Waterfall for Your Input")
            fig, ax = plt.subplots(figsize=(12, 8))
            shap.plots.waterfall(explanation, show=False, max_display=15)
            st.pyplot(fig)
            plt.close()
            st.markdown("""
            This waterfall plot shows how each feature in your custom input contributes to the model's prediction.
            Red bars push the prediction toward "delayed," while blue bars push toward "not delayed."
            """)
        except Exception as e:
            st.warning(f"Could not generate SHAP waterfall: {e}")
    else:
        st.info("SHAP waterfall plots are available for tree-based models (Decision Tree, Random Forest, XGBoost). Select one of these models to see feature-level explanations.")


# Footer
st.markdown("---")
st.markdown("*MSIS 522 - Analytics and Machine Learning | Fardeen Meeran | University of Washington*")
