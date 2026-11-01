
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from pathlib import Path

print("=" * 60)
print("EXPLAINABLE AI - SHAP ANALYSIS")
print("=" * 60)

# Load trained model
model = joblib.load("models/best_model.joblib")

# Load feature names
feature_columns = joblib.load(
    "models/feature_columns.joblib"
)

# Load processed dataset
df = pd.read_csv("dataset/processed_dataset.csv")

# Prepare binary labels
y = (df["Label"] != "BENIGN").astype(int)

X = df.drop(columns=["Label"])

# Use the exact training features
X = X[feature_columns]

# Clean values
X = X.replace([np.inf, -np.inf], np.nan)

X = X.fillna(X.median())

# Select a manageable sample
sample = X.sample(
    n=min(300, len(X)),
    random_state=42
)

print("\nDataset Sample Shape:", sample.shape)

# Create SHAP explainer
print("\nCreating SHAP explainer...")

explainer = shap.TreeExplainer(model)

# Calculate SHAP values
shap_values = explainer.shap_values(sample)

print("\nSHAP values calculated successfully!")

# Create output folder
output_dir = Path("models")
output_dir.mkdir(exist_ok=True)

# Feature importance bar plot
plt.figure(figsize=(12, 7))

shap.summary_plot(
    shap_values,
    sample,
    plot_type="bar",
    show=False
)

plt.title("SHAP Feature Importance")

plt.tight_layout()

plt.savefig(
    output_dir / "shap_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# SHAP summary plot
plt.figure(figsize=(12, 8))

shap.summary_plot(
    shap_values,
    sample,
    show=False
)

plt.title("SHAP Summary Plot")

plt.tight_layout()

plt.savefig(
    output_dir / "shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nSaved SHAP visualizations:")
print("models/shap_feature_importance.png")
print("models/shap_summary.png")

print("\nSHAP ANALYSIS COMPLETED!")
