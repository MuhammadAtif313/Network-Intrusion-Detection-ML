
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Network Intrusion Detection",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Network Intrusion Detection System")

st.markdown(
    """
    ### Machine Learning-Based Cybersecurity Analysis

    Upload network traffic data to identify normal
    and potentially malicious network activity.
    """
)

st.divider()


# ==========================================
# 2. LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_model():

    model = joblib.load(
        "models/best_model.joblib"
    )

    feature_columns = joblib.load(
        "models/feature_columns.joblib"
    )

    reference_data = pd.read_csv(
        "dataset/processed_dataset.csv"
    )

    medians = (
        reference_data[feature_columns]
        .median()
    )

    return model, feature_columns, medians


try:

    model, feature_columns, medians = load_model()

    st.success("Machine Learning Model Loaded Successfully!")

except Exception as e:

    st.error(f"Model loading failed: {e}")
    st.stop()


# ==========================================
# 3. SIDEBAR
# ==========================================

st.sidebar.title("🛡️ Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Home",
        "Traffic Prediction",
        "Model Explainability"
    ]
)


# ==========================================
# 4. HOME PAGE
# ==========================================

if page == "Home":

    st.subheader("System Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "ML Models",
        "3",
        "Compared"
    )

    col2.metric(
        "Best Model",
        "XGBoost"
    )

    col3.metric(
        "Features",
        len(feature_columns)
    )

    st.markdown(
        """
        ### Project Features

        - Network traffic classification
        - Benign and attack detection
        - CSV dataset upload
        - Machine learning predictions
        - SHAP-based model explanations

        ### How to use

        1. Open Traffic Prediction.
        2. Upload a network traffic CSV file.
        3. Click Analyze Network Traffic.
        4. View prediction results.
        """
    )


# ==========================================
# 5. TRAFFIC PREDICTION PAGE
# ==========================================

elif page == "Traffic Prediction":

    st.subheader("Network Traffic Analysis")

    uploaded_file = st.file_uploader(
        "Upload Network Traffic CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            df = pd.read_csv(uploaded_file)

            st.write("### Uploaded Dataset Preview")

            st.dataframe(
                df.head(),
                use_container_width=True
            )

            st.write(
                "Total Records:",
                len(df)
            )

            # Remove identifier columns
            identifier_columns = [
                "Flow ID",
                "Source IP",
                "Destination IP",
                "Timestamp",
                "Label"
            ]

            df_features = df.drop(
                columns=identifier_columns,
                errors="ignore"
            )

            # Clean column names
            df_features.columns = (
                df_features.columns.str.strip()
            )

            # Check required model features
            missing_columns = [
                col for col in feature_columns
                if col not in df_features.columns
            ]

            if missing_columns:

                st.error(
                    "The uploaded CSV does not contain "
                    "all required model features."
                )

                st.write(
                    "Missing columns:",
                    missing_columns
                )

            else:

                if st.button(
                    "🔍 Analyze Network Traffic",
                    type="primary"
                ):

                    # Select exact training features
                    X = df_features[feature_columns].copy()

                    # Convert values to numeric
                    for col in X.columns:

                        X[col] = pd.to_numeric(
                            X[col],
                            errors="coerce"
                        )

                    # Replace infinite values
                    X = X.replace(
                        [np.inf, -np.inf],
                        np.nan
                    )

                    # Fill missing values using training reference medians
                    X = X.fillna(medians)

                    # Make predictions
                    predictions = model.predict(X)

                    probabilities = model.predict_proba(X)[:, 1]

                    # Prepare results
                    results = df.copy()

                    results["Prediction"] = np.where(
                        predictions == 1,
                        "ATTACK",
                        "BENIGN"
                    )

                    results["Attack Probability (%)"] = (
                        probabilities * 100
                    ).round(2)

                    st.success(
                        "Network Traffic Analysis Completed!"
                    )

                    # Summary
                    total = len(results)

                    attacks = int(
                        (predictions == 1).sum()
                    )

                    benign = total - attacks

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "Total Records",
                        total
                    )

                    col2.metric(
                        "Benign Traffic",
                        benign
                    )

                    col3.metric(
                        "Potential Attacks",
                        attacks
                    )

                    # Results table
                    st.subheader("Prediction Results")

                    st.dataframe(
                        results,
                        use_container_width=True
                    )

                    # Download results
                    csv = results.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        "⬇️ Download Prediction Results",
                        data=csv,
                        file_name="intrusion_predictions.csv",
                        mime="text/csv"
                    )

                    # Traffic distribution
                    st.subheader(
                        "Traffic Classification Distribution"
                    )

                    chart_data = pd.Series({
                        "BENIGN": benign,
                        "ATTACK": attacks
                    })

                    st.bar_chart(chart_data)

        except Exception as e:

            st.error(
                f"An error occurred: {e}"
            )


# ==========================================
# 6. MODEL EXPLAINABILITY PAGE
# ==========================================

elif page == "Model Explainability":

    st.subheader("Explainable AI Using SHAP")

    st.write(
        """
        SHAP helps us understand which network traffic
        features influence the machine learning model.
        """
    )

    importance_path = Path(
        "models/shap_feature_importance.png"
    )

    summary_path = Path(
        "models/shap_summary.png"
    )

    if importance_path.exists():

        st.write("### SHAP Feature Importance")

        st.image(
            str(importance_path),
            use_container_width=True
        )

    if summary_path.exists():

        st.write("### SHAP Summary Plot")

        st.image(
            str(summary_path),
            use_container_width=True
        )

    if not importance_path.exists() and not summary_path.exists():

        st.warning(
            "SHAP visualizations are not available."
        )


# ==========================================
# 7. FOOTER
# ==========================================

st.divider()

st.caption(
    "Network Intrusion Detection System | "
    "Machine Learning and Explainable AI"
)
