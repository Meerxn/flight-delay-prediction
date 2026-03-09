"""
MSIS 522 - Homework 1: Complete Data Science Workflow
Flight Delay Prediction (2024 US Domestic Flights)
Author: Fardeen Meeran
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             classification_report, confusion_matrix)
import joblib
import os
import json

# Paths
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(PROJECT_DIR, 'figures')
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ============================================================
# DATA LOADING & PREPARATION
# ============================================================
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

df = pd.read_csv(os.path.join(PROJECT_DIR, 'flight_data_2024.csv'))
print(f"Raw shape: {df.shape}")

# Create binary target: delayed = weather_delay > 0 OR late_aircraft_delay > 0
df['delayed'] = ((df['weather_delay'] > 0) | (df['late_aircraft_delay'] > 0)).astype(int)

# Feature engineering
# Extract hour from dep_time (HHMM format)
df['dep_hour'] = (df['dep_time'] // 100).clip(0, 23)

# Drop rows where dep_time is missing (cancelled flights mostly)
df_clean = df.dropna(subset=['dep_time', 'taxi_out', 'taxi_in']).copy()
print(f"After dropping missing: {df_clean.shape}")

# For modeling, we sample 75K rows for a good balance of speed and representativeness
df_sample = df_clean.sample(n=75000, random_state=RANDOM_STATE).reset_index(drop=True)
print(f"Sample for modeling: {df_sample.shape}")

# Save basic stats
stats = {
    'total_rows': int(df.shape[0]),
    'total_columns': int(df.shape[1]),
    'clean_rows': int(df_clean.shape[0]),
    'sample_rows': 75000,
    'delay_rate': float(df_clean['delayed'].mean()),
    'num_origins': int(df_clean['origin'].nunique()),
    'num_states': int(df_clean['origin_state_nm'].nunique()),
    'feature_types': {
        'numerical': ['month', 'day_of_month', 'day_of_week', 'dep_hour', 'taxi_out', 'taxi_in', 'distance'],
        'categorical': ['origin_state_nm']
    }
}

# ============================================================
# PART 1: DESCRIPTIVE ANALYTICS
# ============================================================
print("\n" + "=" * 60)
print("PART 1: DESCRIPTIVE ANALYTICS")
print("=" * 60)


# 1.2 Target Distribution
fig, ax = plt.subplots(figsize=(8, 5))
counts = df_clean['delayed'].value_counts()
bars = ax.bar(['Not Delayed (0)', 'Delayed (1)'], counts.values, color=['#2ecc71', '#e74c3c'])
for bar, count in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000,
            f'{count:,}\n({count/len(df_clean)*100:.1f}%)', ha='center', fontsize=12)
ax.set_title('Flight Delay Distribution (2024)', fontsize=14, fontweight='bold')
ax.set_ylabel('Number of Flights')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'target_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: target_distribution.png")

# 1.3 Feature Distributions and Relationships

# Viz 1: Delay rate by month
fig, ax = plt.subplots(figsize=(8, 5))
monthly = df_clean.groupby('month')['delayed'].mean() * 100
monthly.plot(kind='bar', color='#3498db', ax=ax)
ax.set_title('Flight Delay Rate by Month (Jan-Feb 2024)', fontsize=14, fontweight='bold')
ax.set_ylabel('Delay Rate (%)')
ax.set_xlabel('Month')
ax.set_xticklabels(['January', 'February'], rotation=0)
for i, v in enumerate(monthly.values):
    ax.text(i, v + 0.2, f'{v:.1f}%', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'delay_by_month.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: delay_by_month.png")

# Viz 2: Delay rate by day of week
fig, ax = plt.subplots(figsize=(10, 5))
dow_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dow_delay = df_clean.groupby('day_of_week')['delayed'].mean() * 100
bars = ax.bar(range(1, 8), dow_delay.values, color='#9b59b6')
ax.set_xticks(range(1, 8))
ax.set_xticklabels(dow_labels, rotation=45)
ax.set_title('Flight Delay Rate by Day of Week', fontsize=14, fontweight='bold')
ax.set_ylabel('Delay Rate (%)')
for i, v in enumerate(dow_delay.values):
    ax.text(i + 1, v + 0.2, f'{v:.1f}%', ha='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'delay_by_dow.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: delay_by_dow.png")

# Viz 3: Delay rate by departure hour
fig, ax = plt.subplots(figsize=(12, 5))
hour_delay = df_clean.groupby('dep_hour')['delayed'].mean() * 100
ax.plot(hour_delay.index, hour_delay.values, marker='o', color='#e67e22', linewidth=2, markersize=6)
ax.fill_between(hour_delay.index, hour_delay.values, alpha=0.2, color='#e67e22')
ax.set_title('Flight Delay Rate by Departure Hour', fontsize=14, fontweight='bold')
ax.set_xlabel('Departure Hour (24h)')
ax.set_ylabel('Delay Rate (%)')
ax.set_xticks(range(0, 24))
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'delay_by_hour.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: delay_by_hour.png")

# Viz 4: Distance distribution by delay status
fig, ax = plt.subplots(figsize=(10, 5))
df_clean[df_clean['delayed'] == 0]['distance'].sample(50000, random_state=42).hist(
    bins=50, alpha=0.6, label='Not Delayed', color='#2ecc71', ax=ax, density=True)
df_clean[df_clean['delayed'] == 1]['distance'].sample(min(50000, df_clean['delayed'].sum()), random_state=42).hist(
    bins=50, alpha=0.6, label='Delayed', color='#e74c3c', ax=ax, density=True)
ax.set_title('Flight Distance Distribution by Delay Status', fontsize=14, fontweight='bold')
ax.set_xlabel('Distance (miles)')
ax.set_ylabel('Density')
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'distance_by_delay.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: distance_by_delay.png")

# Viz 5: Top 15 states by delay rate
fig, ax = plt.subplots(figsize=(12, 6))
state_counts = df_clean.groupby('origin_state_nm').size()
major_states = state_counts[state_counts >= 5000].index
state_delay = df_clean[df_clean['origin_state_nm'].isin(major_states)].groupby('origin_state_nm')['delayed'].mean() * 100
state_delay = state_delay.sort_values(ascending=False).head(15)
state_delay.plot(kind='barh', color='#e74c3c', ax=ax)
ax.set_title('Top 15 States by Flight Delay Rate (min 5,000 flights)', fontsize=14, fontweight='bold')
ax.set_xlabel('Delay Rate (%)')
ax.set_ylabel('')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'delay_by_state.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: delay_by_state.png")

# 1.4 Correlation Heatmap
numerical_cols = ['month', 'day_of_month', 'day_of_week', 'dep_hour', 'taxi_out', 'taxi_in', 'distance', 'cancelled', 'weather_delay', 'late_aircraft_delay', 'delayed']
corr_matrix = df_clean[numerical_cols].corr()

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, square=True, ax=ax)
ax.set_title('Correlation Heatmap of Flight Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: correlation_heatmap.png")

# ============================================================
# PART 2: PREDICTIVE ANALYTICS
# ============================================================
print("\n" + "=" * 60)
print("PART 2: PREDICTIVE ANALYTICS")
print("=" * 60)

# 2.1 Data Preparation
# Features: month, day_of_month, day_of_week, dep_hour, distance, origin_state (encoded)
# We avoid taxi_out/taxi_in as they happen during the flight, not before
# We keep dep_hour as a proxy for scheduled departure time

# Encode origin_state_nm using top 20 states + "Other"
top_states = df_sample['origin_state_nm'].value_counts().head(20).index.tolist()
df_sample['state_encoded'] = df_sample['origin_state_nm'].apply(
    lambda x: x if x in top_states else 'Other')

# One-hot encode state
state_dummies = pd.get_dummies(df_sample['state_encoded'], prefix='state', drop_first=True)

feature_cols = ['month', 'day_of_month', 'day_of_week', 'dep_hour', 'distance']
X = pd.concat([df_sample[feature_cols], state_dummies], axis=1)
y = df_sample['delayed']

print(f"Features shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts()}")
print(f"Delay rate: {y.mean()*100:.1f}%")

# Train/test split (70/30)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save feature names and scaler
feature_names = list(X.columns)
joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.joblib'))
joblib.dump(feature_names, os.path.join(MODELS_DIR, 'feature_names.joblib'))
joblib.dump(top_states, os.path.join(MODELS_DIR, 'top_states.joblib'))

# Helper function to evaluate model
def evaluate_model(name, model, X_test_data, y_test_data, use_scaled=False):
    data = X_test_data if not use_scaled else X_test_scaled
    y_pred = model.predict(data)
    y_proba = model.predict_proba(data)[:, 1] if hasattr(model, 'predict_proba') else y_pred

    metrics = {
        'model': name,
        'accuracy': float(accuracy_score(y_test_data, y_pred)),
        'precision': float(precision_score(y_test_data, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test_data, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test_data, y_pred, zero_division=0)),
        'auc_roc': float(roc_auc_score(y_test_data, y_proba))
    }
    print(f"\n{name}:")
    for k, v in metrics.items():
        if k != 'model':
            print(f"  {k}: {v:.4f}")
    return metrics, y_proba

all_metrics = []

# 2.2 Logistic Regression Baseline
print("\n--- 2.2 Logistic Regression ---")
lr = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, class_weight='balanced')
lr.fit(X_train_scaled, y_train)
lr_metrics, lr_proba = evaluate_model('Logistic Regression', lr, X_test_scaled, y_test)
all_metrics.append(lr_metrics)
joblib.dump(lr, os.path.join(MODELS_DIR, 'logistic_regression.joblib'))

# 2.3 Decision Tree with GridSearchCV
print("\n--- 2.3 Decision Tree ---")
dt_param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_leaf': [5, 10, 20, 50]
}
dt_cv = GridSearchCV(
    DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight='balanced'),
    dt_param_grid, cv=5, scoring='f1', n_jobs=1, verbose=1
)
dt_cv.fit(X_train, y_train)
print(f"Best params: {dt_cv.best_params_}")
print(f"Best CV F1: {dt_cv.best_score_:.4f}")

dt_best = dt_cv.best_estimator_
dt_metrics, dt_proba = evaluate_model('Decision Tree', dt_best, X_test, y_test)
all_metrics.append(dt_metrics)
joblib.dump(dt_best, os.path.join(MODELS_DIR, 'decision_tree.joblib'))

# Visualize the best tree
fig, ax = plt.subplots(figsize=(24, 12))
plot_tree(dt_best, filled=True, feature_names=feature_names,
          class_names=['Not Delayed', 'Delayed'], max_depth=3, ax=ax,
          fontsize=10, rounded=True)
ax.set_title('Decision Tree (Top 3 Levels)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'decision_tree.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: decision_tree.png")

# Save best params
dt_best_params = dt_cv.best_params_

# 2.4 Random Forest with GridSearchCV
print("\n--- 2.4 Random Forest ---")
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 8]
}
rf_cv = GridSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced', n_jobs=1),
    rf_param_grid, cv=5, scoring='f1', n_jobs=1, verbose=1
)
rf_cv.fit(X_train, y_train)
print(f"Best params: {rf_cv.best_params_}")
print(f"Best CV F1: {rf_cv.best_score_:.4f}")

rf_best = rf_cv.best_estimator_
rf_metrics, rf_proba = evaluate_model('Random Forest', rf_best, X_test, y_test)
all_metrics.append(rf_metrics)
joblib.dump(rf_best, os.path.join(MODELS_DIR, 'random_forest.joblib'))

rf_best_params = rf_cv.best_params_

# 2.5 XGBoost with GridSearchCV
print("\n--- 2.5 XGBoost ---")
# Calculate scale_pos_weight for imbalanced data
scale_pos = (y_train == 0).sum() / (y_train == 1).sum()

xgb_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1]
}
xgb_cv = GridSearchCV(
    XGBClassifier(random_state=RANDOM_STATE, scale_pos_weight=scale_pos,
                  eval_metric='logloss', n_jobs=1),
    xgb_param_grid, cv=5, scoring='f1', n_jobs=1, verbose=1
)
xgb_cv.fit(X_train, y_train)
print(f"Best params: {xgb_cv.best_params_}")
print(f"Best CV F1: {xgb_cv.best_score_:.4f}")

xgb_best = xgb_cv.best_estimator_
xgb_metrics, xgb_proba = evaluate_model('XGBoost', xgb_best, X_test, y_test)
all_metrics.append(xgb_metrics)
joblib.dump(xgb_best, os.path.join(MODELS_DIR, 'xgboost.joblib'))

xgb_best_params = xgb_cv.best_params_

# ROC Curves for all models so far
fig, ax = plt.subplots(figsize=(10, 8))
for name, proba in [('Logistic Regression', lr_proba), ('Decision Tree', dt_proba),
                     ('Random Forest', rf_proba), ('XGBoost', xgb_proba)]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - All Models', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'roc_curves.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: roc_curves.png")

# 2.6 Neural Network (MLP) with Keras/TensorFlow
print("\n--- 2.6 Neural Network (MLP) ---")
import tensorflow as tf
tf.random.set_seed(RANDOM_STATE)

model_nn = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model_nn.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Class weights for imbalanced data
class_weights = {0: 1.0, 1: scale_pos}

history = model_nn.fit(
    X_train_scaled, y_train,
    epochs=50, batch_size=256,
    validation_split=0.2,
    class_weight=class_weights,
    verbose=1
)

# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history.history['loss'], label='Train Loss')
axes[0].plot(history.history['val_loss'], label='Val Loss')
axes[0].set_title('Training & Validation Loss', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(history.history['accuracy'], label='Train Accuracy')
axes[1].plot(history.history['val_accuracy'], label='Val Accuracy')
axes[1].set_title('Training & Validation Accuracy', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'mlp_training_history.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: mlp_training_history.png")

# Evaluate MLP
nn_proba = model_nn.predict(X_test_scaled, verbose=0).flatten()
nn_pred = (nn_proba >= 0.5).astype(int)
nn_metrics = {
    'model': 'Neural Network (MLP)',
    'accuracy': float(accuracy_score(y_test, nn_pred)),
    'precision': float(precision_score(y_test, nn_pred, zero_division=0)),
    'recall': float(recall_score(y_test, nn_pred, zero_division=0)),
    'f1': float(f1_score(y_test, nn_pred, zero_division=0)),
    'auc_roc': float(roc_auc_score(y_test, nn_proba))
}
all_metrics.append(nn_metrics)
print(f"\nNeural Network (MLP):")
for k, v in nn_metrics.items():
    if k != 'model':
        print(f"  {k}: {v:.4f}")

# Save MLP model
model_nn.save(os.path.join(MODELS_DIR, 'mlp_model.keras'))

# Add MLP to ROC curves
fig, ax = plt.subplots(figsize=(10, 8))
for name, proba in [('Logistic Regression', lr_proba), ('Decision Tree', dt_proba),
                     ('Random Forest', rf_proba), ('XGBoost', xgb_proba),
                     ('Neural Network', nn_proba)]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - All Models', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'roc_curves_all.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: roc_curves_all.png")

# 2.7 Model Comparison Summary
print("\n--- 2.7 Model Comparison ---")
metrics_df = pd.DataFrame(all_metrics)
print(metrics_df.to_string(index=False))
metrics_df.to_csv(os.path.join(MODELS_DIR, 'model_comparison.csv'), index=False)

# Bar chart comparing F1 scores
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#3498db', '#2ecc71', '#9b59b6', '#e74c3c', '#f39c12']
bars = ax.bar(metrics_df['model'], metrics_df['f1'], color=colors)
for bar, val in zip(bars, metrics_df['f1']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')
ax.set_title('F1 Score Comparison Across Models', fontsize=14, fontweight='bold')
ax.set_ylabel('F1 Score')
ax.set_ylim(0, max(metrics_df['f1']) * 1.15)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'model_comparison_f1.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: model_comparison_f1.png")

# Save hyperparameters
best_params = {
    'logistic_regression': {'max_iter': 1000, 'class_weight': 'balanced'},
    'decision_tree': dt_best_params,
    'random_forest': rf_best_params,
    'xgboost': xgb_best_params,
    'mlp': {'hidden_layers': [128, 128, 64], 'dropout': 0.3, 'lr': 0.001, 'epochs': 50, 'batch_size': 256}
}
with open(os.path.join(MODELS_DIR, 'best_params.json'), 'w') as f:
    json.dump(best_params, f, indent=2)

# ============================================================
# PART 3: SHAP EXPLAINABILITY
# ============================================================
print("\n" + "=" * 60)
print("PART 3: SHAP EXPLAINABILITY")
print("=" * 60)

import shap

# Use the best tree model (XGBoost or RF based on F1)
tree_metrics = {m['model']: m['f1'] for m in all_metrics if m['model'] in ['Random Forest', 'XGBoost']}
best_tree_name = max(tree_metrics, key=tree_metrics.get)
print(f"Best tree model for SHAP: {best_tree_name}")

if best_tree_name == 'XGBoost':
    shap_model = xgb_best
    explainer = shap.TreeExplainer(shap_model)
else:
    shap_model = rf_best
    explainer = shap.TreeExplainer(shap_model)

# Use a subset for SHAP (faster)
X_shap = X_test.sample(n=min(1000, len(X_test)), random_state=RANDOM_STATE)
shap_values = explainer.shap_values(X_shap)

# For binary classification, shap_values might be a list [class0, class1]
if isinstance(shap_values, list):
    shap_vals = shap_values[1]  # class 1 (delayed)
else:
    shap_vals = shap_values

# Summary plot (beeswarm)
fig, ax = plt.subplots(figsize=(12, 8))
shap.summary_plot(shap_vals, X_shap, feature_names=feature_names, show=False, max_display=15)
plt.title('SHAP Summary Plot (Beeswarm)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'shap_summary.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: shap_summary.png")

# Bar plot of mean absolute SHAP values
fig, ax = plt.subplots(figsize=(12, 8))
shap.summary_plot(shap_vals, X_shap, feature_names=feature_names, plot_type='bar', show=False, max_display=15)
plt.title('SHAP Feature Importance (Mean |SHAP|)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'shap_bar.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: shap_bar.png")

# Waterfall plot for one specific prediction (a delayed flight)
delayed_indices = X_shap.index[y_test.loc[X_shap.index] == 1]
if len(delayed_indices) > 0:
    example_idx = delayed_indices[0]
    example_pos = list(X_shap.index).index(example_idx)

    # Create SHAP Explanation object for waterfall
    if isinstance(shap_values, list):
        base_val = explainer.expected_value[1]
    else:
        base_val = explainer.expected_value
        if isinstance(base_val, np.ndarray):
            base_val = base_val[0]

    explanation = shap.Explanation(
        values=shap_vals[example_pos],
        base_values=base_val,
        data=X_shap.iloc[example_pos].values,
        feature_names=feature_names
    )

    fig, ax = plt.subplots(figsize=(12, 8))
    shap.plots.waterfall(explanation, show=False, max_display=15)
    plt.title('SHAP Waterfall Plot (Example Delayed Flight)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'shap_waterfall.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: shap_waterfall.png")

# Save SHAP explainer expected value
shap_info = {
    'best_tree_model': best_tree_name,
    'expected_value': float(base_val),
    'top_features': list(pd.Series(np.abs(shap_vals).mean(axis=0), index=feature_names).sort_values(ascending=False).head(10).index)
}
with open(os.path.join(MODELS_DIR, 'shap_info.json'), 'w') as f:
    json.dump(shap_info, f, indent=2)

# Save stats
with open(os.path.join(MODELS_DIR, 'stats.json'), 'w') as f:
    json.dump(stats, f, indent=2)

print("\n" + "=" * 60)
print("ALL DONE! Models and figures saved.")
print("=" * 60)
print(f"Models directory: {MODELS_DIR}")
print(f"Figures directory: {FIGURES_DIR}")
