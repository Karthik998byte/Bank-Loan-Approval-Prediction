# ============================================================
# BANK LOAN APPROVAL DECISION SUPPORT SYSTEM
# Production-Ready Streamlit App
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Bank Loan Approval System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
    .stApp { background-color: #0f172a; color: white; }
    .title {
        font-size: 30px;
        font-weight: 700;
        color: #60a5fa;
    }
    .card {
        background: #111827;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #1f2937;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    artifact = joblib.load("artifacts/loan_approval_model.pkl")
    return artifact["model"], artifact["scaler"], artifact["feature_names"]

model, scaler, feature_names = load_model()

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🏦 Bank AI System")
st.sidebar.markdown("---")
st.sidebar.info("Internal Banking Use Only")

st.sidebar.markdown("""
### Model Info
- Algorithm: Logistic Regression  
- Type: Classification  
- Output: Approved / Rejected  

### Workflow
1. Input Data  
2. Feature Engineering  
3. Scaling  
4. Prediction  
5. Risk Scoring  
""")

# =========================
# HEADER
# =========================
st.markdown('<div class="title">Loan Approval Decision System</div>', unsafe_allow_html=True)
st.markdown("AI-powered credit risk evaluation system")

st.markdown("---")

# =========================
# INPUT SECTION
# =========================

col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Applicant Information")

    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Marital Status", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Employment Type", ["Salaried", "Self Employed"])
    credit_history = st.selectbox("Credit History", ["Good", "Poor"])

with col2:
    st.subheader("💰 Financial Information")

    applicant_income = st.number_input("Applicant Monthly Income (₹)", min_value=0, step=1000)
    coapplicant_income = st.number_input("Co-applicant Monthly Income (₹)", min_value=0, step=1000)

    loan_amount = st.number_input("Requested Loan Amount (₹)", min_value=0, step=5000)

    loan_term = st.slider("Loan Tenure (Years)", 1, 30, 10)

    emi = st.number_input("Estimated Monthly EMI (₹)", min_value=0, step=500)

# =========================
# PROPERTY INFO
# =========================
st.subheader("🏠 Property Information")
property_area = st.selectbox("Property Location", ["Rural", "Semiurban", "Urban"])

# =========================
# DERIVED FEATURES
# =========================
total_income = applicant_income + coapplicant_income
loan_income_ratio = loan_amount / total_income if total_income != 0 else 0

st.markdown("### 📊 Derived Metrics")

c1, c2, c3 = st.columns(3)
c1.metric("Total Income", f"₹{total_income:,.0f}")
c2.metric("Loan Amount", f"₹{loan_amount:,.0f}")
c3.metric("Loan Ratio", f"{loan_income_ratio:.2f}")

st.markdown("---")

# =========================
# PREPROCESSING
# =========================
def preprocess():
    married_flag = 1 if married == "Yes" else 0
    self_emp_flag = 1 if self_employed == "Self Employed" else 0
    credit_flag = 1 if credit_history == "Good" else 0
    dependents_val = 3 if dependents == "3+" else int(dependents)

    prop_semi = 1 if property_area == "Semiurban" else 0
    prop_urban = 1 if property_area == "Urban" else 0

    data = {
        "Gender": 1 if gender == "Male" else 0,
        "Married": married_flag,
        "Dependents": dependents_val,
        "Education": 1 if education == "Graduate" else 0,
        "Self_Employed": self_emp_flag,
        "LoanAmount": loan_amount,
        "Credit_History": credit_flag,
        "TotalIncome": total_income,
        "LoanTermYears": loan_term,
        "LoanIncomeRatio": loan_income_ratio,
        "EMI": emi,
        "Property_Area_Semiurban": prop_semi,
        "Property_Area_Urban": prop_urban
    }

    df = pd.DataFrame([data])

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    return df[feature_names]

# =========================
# RISK FUNCTION
# =========================
def risk(prob):
    if prob < 0.2:
        return "Very Low Risk"
    elif prob < 0.4:
        return "Low Risk"
    elif prob < 0.7:
        return "Moderate Risk"
    else:
        return "High Risk"

# =========================
# PREDICTION
# =========================
def predict():
    df = preprocess()
    scaled = scaler.transform(df)
    prob = model.predict_proba(scaled)[0][1]
    pred = model.predict(scaled)[0]
    return pred, prob

# =========================
# VALIDATION RULES
# =========================
if loan_amount > 10 * total_income and total_income > 0:
    st.warning("⚠ High Loan-to-Income ratio detected")

if emi > total_income * 0.5 and total_income > 0:
    st.error("❌ EMI exceeds safe threshold (50% income)")

# =========================
# BUTTON
# =========================
if st.button("🔍 Evaluate Loan Application", use_container_width=True):

    if model is None:
        st.error("Model not loaded")
    else:
        pred, prob = predict()

        status = "APPROVED" if pred == 1 else "REJECTED"
        risk_level = risk(prob)

        st.markdown("## 📊 Result")

        a, b, c = st.columns(3)
        a.metric("Decision", status)
        b.metric("Approval Probability", f"{prob*100:.2f}%")
        c.metric("Risk Level", risk_level)

        st.progress(float(prob))

        st.markdown("### 📋 Summary")
        st.table(pd.DataFrame({
            "Metric": ["Income", "Loan", "Ratio", "Credit History"],
            "Value": [total_income, loan_amount, loan_income_ratio, credit_history]
        }))

        st.markdown("### 🧠 Recommendation")

        if status == "APPROVED":
            st.success("""
✔ Proceed with:
- Document verification  
- Identity check  
- Income verification  
- Final underwriting  
""")
        else:
            st.error("""
❌ Suggested Actions:
- Improve credit history  
- Reduce loan amount  
- Add co-applicant  
- Increase income stability  
""")

        st.info("⚠ This is a decision support system only. Final approval is manual.")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("Bank Internal AI System | Credit Risk Engine")