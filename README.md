# 🏦 Loan Approval Prediction

## Overview

This project predicts whether a loan application will be approved using Machine Learning. The model was trained on applicant information such as income, loan amount, credit history, and other financial details.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

## Project Workflow

- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Feature Scaling
- Model Training
- Hyperparameter Tuning
- Model Evaluation
- Model Saving
- Streamlit Deployment

## Model

- Algorithm: Logistic Regression
- Hyperparameter Tuning: RandomizedSearchCV
- Evaluation Metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC
  - Precision-Recall Curve

## Project Files

```
Loan_Approval_Project/
│
├── Loan_Approval.ipynb
├── app.py
├── requirements.txt
├── README.md
└── artifacts/
    └── loan_approval_model.pkl
```

## Run the Project

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## Author

**Karthik A Bangera**