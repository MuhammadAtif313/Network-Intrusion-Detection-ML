
# Network Intrusion Detection Using Machine Learning and Explainable AI

## Project Overview

This project implements a Machine Learning-based Network Intrusion Detection System (NIDS) using the CICIDS2017 dataset.

The system analyzes network traffic and classifies it into two categories:

- BENIGN: Normal network traffic.
- ATTACK: Potentially malicious network traffic.

Three machine learning algorithms are implemented and compared: Decision Tree, Random Forest, and XGBoost.

Explainable Artificial Intelligence (XAI) using SHAP is integrated to understand the features influencing model predictions.

A Streamlit web application provides an interactive interface for uploading network traffic CSV files and analyzing predictions.

## Key Features

- Network traffic preprocessing and cleaning.
- Binary network intrusion classification.
- Decision Tree, Random Forest, and XGBoost models.
- Model performance comparison.
- Accuracy, Precision, Recall, and F1-score evaluation.
- SHAP-based Explainable AI.
- Interactive Streamlit dashboard.
- CSV upload and prediction.
- Downloadable prediction results.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib
- Streamlit
- Joblib

## Dataset

This project uses the CICIDS2017 dataset.

The dataset contains benign network traffic and different types of cyberattacks.

Dataset source:
https://www.unb.ca/cic/datasets/ids-2017.html

Note: The original dataset files are not included in this repository because of their large size. Download the dataset and place the required CSV files inside the `dataset/` directory.

## Machine Learning Models

The following models are implemented:

1. Decision Tree
2. Random Forest
3. XGBoost

The models are compared using standard classification metrics.

XGBoost was selected as the best-performing model in the initial experiment.

## Explainable AI

SHAP (SHapley Additive exPlanations) is used to interpret the trained model.

The project generates:

- SHAP Feature Importance Plot
- SHAP Summary Plot

These visualizations help identify which network traffic features influence model predictions.

## Project Structure

```text
Network-Intrusion-Detection-ML/
│
├── dataset/
│
├── models/
│   ├── best_model.joblib
│   ├── feature_columns.joblib
│   ├── model_results.csv
│   ├── shap_feature_importance.png
│   └── shap_summary.png
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── evaluation.py
│
├── app.py
├── main.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the project directory

```bash
cd Network-Intrusion-Detection-ML
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the environment on Windows

```powershell
.venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Prepare the dataset

Download CICIDS2017 and place its required CSV files inside the dataset folder.

### 7. Run preprocessing

```bash
python src/preprocessing.py
```

### 8. Train the models

```bash
python src/train.py
```

### 9. Generate SHAP explanations

```bash
python src/evaluation.py
```

### 10. Launch the Streamlit application

```bash
streamlit run app.py
```

## Application Features

The Streamlit application provides three main sections:

- Home: Project overview and model information.
- Traffic Prediction: Upload network traffic CSV files and obtain predictions.
- Model Explainability: Explore SHAP feature importance and summary plots.

## Future Improvements

- Multiclass attack classification.
- Advanced feature selection.
- Improved detection of rare attack categories.
- Time-based model evaluation.
- Real-time network traffic monitoring.
- Enhanced Explainable AI capabilities.

## Author

Muhammad Atif

BS Software Engineering

Research Interests:
- Machine Learning
- Artificial Intelligence
- Cybersecurity
- Explainable AI
- Network Intrusion Detection

---

If you find this project useful, consider giving the repository a star!
