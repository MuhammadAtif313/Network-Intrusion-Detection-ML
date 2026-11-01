# Network Intrusion Detection Using Machine Learning and Explainable AI

A Machine Learning-Based Network Intrusion Detection System (NIDS) using the CICIDS2017 dataset, multiple classification algorithms, and Explainable Artificial Intelligence (XAI) through SHAP.

## 1. Project Overview

This project implements a Machine Learning-based Network Intrusion Detection System to identify normal and potentially malicious network traffic.

The system analyzes network traffic features and classifies network activities into different traffic categories, including benign traffic and various cyberattacks.

Three machine learning algorithms are implemented and compared:

- Decision Tree
- Random Forest
- XGBoost

Explainable Artificial Intelligence (XAI) is integrated using SHAP (SHapley Additive exPlanations) to interpret model predictions and identify the network traffic features that contribute most to intrusion detection.

A Streamlit-based web application provides an interactive interface for uploading network traffic CSV files, generating predictions, and exploring model explanations.

## 2. Key Features

- Network traffic preprocessing and cleaning.
- Multiclass network intrusion classification.
- Decision Tree, Random Forest, and XGBoost algorithms.
- Machine learning model performance comparison.
- Accuracy, Precision, Recall, and F1-score evaluation.
- SHAP-based model explainability.
- Feature importance visualization.
- SHAP summary plot generation.
- Interactive Streamlit dashboard.
- CSV file upload and traffic prediction.
- Downloadable prediction results.

## 3. Machine Learning Models

The project implements and compares three machine learning algorithms:

| Model | Description |
|---|---|
| Decision Tree | Tree-based classification algorithm |
| Random Forest | Ensemble learning using multiple decision trees |
| XGBoost | Gradient boosting-based classification algorithm |

The models are evaluated using standard classification metrics:

- Accuracy
- Precision
- Recall
- F1-score

XGBoost achieved the best overall performance in the initial model comparison.

Detailed evaluation results are saved in `models/model_results.csv`.

## 4. Explainable Artificial Intelligence (XAI)

SHAP is integrated to improve the interpretability and transparency of machine learning predictions.

The project generates two main visualizations:

**SHAP Feature Importance**

Identifies the most influential network traffic features used by the trained model.

**SHAP Summary Plot**

Illustrates how different feature values influence model predictions.

These visualizations help explain the behavior of the trained intrusion detection model.

## 5. Dataset

This project uses the CICIDS2017 dataset.

**Dataset Source:** https://www.unb.ca/cic/datasets/ids-2017.html

The dataset contains benign network traffic and multiple types of cyberattacks.

**Note:** Original dataset CSV files are not included in this repository because of their large size. Download the dataset and place the required CSV files inside the `dataset/` directory before running preprocessing.

## 6. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- SHAP
- Streamlit
- Joblib

## 7. Project Structure

```text
Network-Intrusion-Detection-ML/
│
├── dataset/
│   └── CICIDS2017 CSV files
│
├── models/
│   ├── best_model.joblib
│   ├── feature_columns.joblib
│   ├── model_results.csv
│   ├── shap_feature_importance.png
│   └── shap_summary.png
│
├── notebooks/
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

## 8. Installation and Setup

Follow these steps to run the project locally.

### Step 1: Clone the Repository

```bash
git clone https://github.com/MuhammadAtif313/Network-Intrusion-Detection-ML.git
```

### Step 2: Navigate to the Project Directory

```bash
cd Network-Intrusion-Detection-ML
```

### Step 3: Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 4: Activate the Virtual Environment

For Windows:

```bash
.venv\Scripts\activate
```

### Step 5: Install Required Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Prepare the Dataset

Download the CICIDS2017 dataset and place the required CSV files inside the `dataset/` directory.

### Step 7: Run Data Preprocessing

```bash
python src/preprocessing.py
```

### Step 8: Train the Machine Learning Models

```bash
python src/train.py
```

### Step 9: Generate SHAP Explanations

```bash
python src/evaluation.py
```

### Step 10: Launch the Streamlit Application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal to access the application.

## 9. Web Application

The Streamlit application provides three main sections.

### Home

- Project introduction.
- System overview.
- Machine learning model information.
- Project features.

### Traffic Prediction

- Upload network traffic CSV files.
- Preview uploaded network traffic.
- Analyze network traffic.
- Generate intrusion predictions.
- View prediction results.

### Model Explainability

- SHAP feature importance visualization.
- SHAP summary visualization.
- Understand influential network traffic features.

## 10. Model Performance and Results

The project evaluates three machine learning models using standard classification metrics.

The comparison results are saved in:

```text
models/model_results.csv
```

The best-performing model is saved as:

```text
models/best_model.joblib
```

The trained feature columns are saved as:

```text
models/feature_columns.joblib
```

## 11. Future Improvements

- Improved detection of rare attack categories.
- Advanced feature selection techniques.
- Real-time network traffic monitoring.
- Time-based model evaluation.
- Enhanced multiclass classification.
- Advanced Explainable AI techniques.
- Integration with real-world network monitoring systems.

## 12. Author

**Muhammad Atif**

BS Software Engineering

**Research Interests:**

- Machine Learning
- Artificial Intelligence
- Cybersecurity
- Explainable Artificial Intelligence
- Network Intrusion Detection

**GitHub Profile:**  
https://github.com/MuhammadAtif313

**Project Repository:**  
https://github.com/MuhammadAtif313/Network-Intrusion-Detection-ML

---

If you find this project useful, consider giving the repository a star!