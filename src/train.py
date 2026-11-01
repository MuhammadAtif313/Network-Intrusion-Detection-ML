
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

DATA_PATH = Path("dataset/processed_dataset.csv")
MODEL_DIR = Path("models")

MODEL_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


# ==========================================
# 2. LOAD DATASET
# ==========================================

print("=" * 60)
print("NETWORK INTRUSION DETECTION")
print("MACHINE LEARNING MODEL TRAINING")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print("\nDataset Shape:", df.shape)


# ==========================================
# 3. PREPARE FEATURES AND LABELS
# ==========================================

# Remove rows with invalid labels
df["Label"] = df["Label"].astype(str).str.strip()

df = df[
    df["Label"].notna()
    & (df["Label"] != "")
]

# Binary classification:
# BENIGN = 0
# ATTACK = 1

y = (df["Label"] != "BENIGN").astype(int)

X = df.drop(columns=["Label"])

# Keep numerical features only
X = X.select_dtypes(include=np.number)

# Replace infinite values
X = X.replace([np.inf, -np.inf], np.nan)

# Remove columns containing only missing values
X = X.dropna(axis=1, how="all")

# Fill remaining missing values
X = X.fillna(X.median())

# Remove constant features
X = X.loc[:, X.nunique() > 1]

feature_columns = X.columns.tolist()

print("\nNumber of Features:", len(feature_columns))

print("\nClass Distribution:")
print(y.value_counts().rename({
    0: "BENIGN",
    1: "ATTACK"
}))


# ==========================================
# 4. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))


# ==========================================
# 5. DEFINE MACHINE LEARNING MODELS
# ==========================================

negative_count = int((y_train == 0).sum())
positive_count = int((y_train == 1).sum())

models = {

    "Decision Tree": DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=negative_count / positive_count,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
}


# ==========================================
# 6. TRAIN AND EVALUATE MODELS
# ==========================================

results = []
trained_models = {}

for name, model in models.items():

    print("\n" + "=" * 60)
    print("TRAINING:", name)
    print("=" * 60)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test, predictions, zero_division=0
    )

    recall = recall_score(
        y_test, predictions, zero_division=0
    )

    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    trained_models[name] = model

    print("\nAccuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=["BENIGN", "ATTACK"],
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))


# ==========================================
# 7. COMPARE MODEL PERFORMANCE
# ==========================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False))


# ==========================================
# 8. SAVE BEST MODEL
# ==========================================

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[best_model_name]

joblib.dump(
    best_model,
    MODEL_DIR / "best_model.joblib"
)

joblib.dump(
    feature_columns,
    MODEL_DIR / "feature_columns.joblib"
)

results_df.to_csv(
    MODEL_DIR / "model_results.csv",
    index=False
)

print("\nBest Model:", best_model_name)

print("\nSaved Files:")
print("models/best_model.joblib")
print("models/feature_columns.joblib")
print("models/model_results.csv")

print("\nMODEL TRAINING COMPLETED!")
