# =====================================================
# BANK LOAN APPROVAL DECISION SUPPORT SYSTEM
# STREAMLIT APPLICATION
# PART 1: SETUP AND MODEL LOADING
# =====================================================


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os



# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Bank Loan Approval System",
    page_icon="🏦",
    layout="wide"
)



# =========================
# LOAD MODEL
# =========================


@st.cache_resource
def load_model():

    model_path = "artifacts/loan_approval_model.pkl"


    if not os.path.exists(model_path):

        st.error(
            "Model file not found"
        )

        st.stop()



    artifact = joblib.load(
        model_path
    )


    model = artifact["model"]

    feature_names = artifact["feature_names"]


    return model, feature_names



model, feature_names = load_model()

# =====================================================
# PART 2: USER INTERFACE
# =====================================================


st.title(
    "🏦 BANK LOAN APPROVAL DECISION SUPPORT SYSTEM"
)


st.markdown(
"""
This application predicts loan approval probability
using Machine Learning.
"""
)


st.divider()





st.subheader(
    "👤 Applicant Details"
)



col1,col2 = st.columns(2)



with col1:


    gender = st.selectbox(
        "Gender",
        [
            "Male",
            "Female"
        ]
    )


    married = st.selectbox(
        "Married",
        [
            "Yes",
            "No"
        ]
    )


    dependents = st.selectbox(
        "Dependents",
        [
            "0",
            "1",
            "2",
            "3+"
        ]
    )


    education = st.selectbox(
        "Education",
        [
            "Graduate",
            "Not Graduate"
        ]
    )


    self_employed = st.selectbox(
        "Self Employed",
        [
            "Yes",
            "No"
        ]
    )



with col2:


    applicant_income = st.number_input(
        "Applicant Monthly Income (₹)",
        min_value=0,
        value=50000,
        step=5000
    )



    coapplicant_income = st.number_input(
        "Co Applicant Monthly Income (₹)",
        min_value=0,
        value=0,
        step=5000
    )



    loan_amount_lakh = st.number_input(
        "Loan Amount (Lakhs ₹)",
        min_value=0.0,
        value=10.0,
        step=1.0
    )



    loan_term_years = st.slider(
        "Loan Term (Years)",
        1,
        30,
        20
    )



    credit_history = st.selectbox(
        "Credit History",
        [
            "Good",
            "Bad"
        ]
    )



property_area = st.selectbox(

    "Property Area",

    [
        "Urban",
        "Semiurban",
        "Rural"
    ]

)



# =====================================================
# PART 3: DATA PREPARATION
# =====================================================


def prepare_input():



    # -------------------------
    # Unit Conversion
    # -------------------------


    applicant_income_model = (
        applicant_income / 1000
    )


    coapplicant_income_model = (
        coapplicant_income / 1000
    )



    loan_amount_model = (
        loan_amount_lakh * 100
    )



    # -------------------------
    # Feature Engineering
    # -------------------------


    total_income = (

        applicant_income_model +
        coapplicant_income_model

    )


    loan_income_ratio = (

        loan_amount_model / total_income

        if total_income > 0

        else 0

    )



    emi = (

        loan_amount_model /

        (loan_term_years * 12)

    )



    # -------------------------
    # Log Transformation
    # -------------------------


    LoanAmount = np.log1p(
        loan_amount_model
    )


    TotalIncome = np.log1p(
        total_income
    )


    LoanIncomeRatio = np.log1p(
        loan_income_ratio
    )


    EMI = np.log1p(
        emi
    )



    # -------------------------
    # Encoding
    # -------------------------


    dependents_value = (

        3 if dependents=="3+"

        else int(dependents)

    )



    credit_value = (

        1 if credit_history=="Good"

        else 0

    )



    data = {


        "Gender":
        1 if gender=="Male" else 0,


        "Married":
        1 if married=="Yes" else 0,


        "Dependents":
        dependents_value,


        "Education":
        1 if education=="Graduate" else 0,


        "Self_Employed":
        1 if self_employed=="Yes" else 0,


        "LoanAmount":
        LoanAmount,


        "Credit_History":
        credit_value,


        "TotalIncome":
        TotalIncome,


        "LoanTermYears":
        loan_term_years,


        "LoanIncomeRatio":
        LoanIncomeRatio,


        "EMI":
        EMI,


        "Property_Area_Semiurban":
        1 if property_area=="Semiurban" else 0,


        "Property_Area_Urban":
        1 if property_area=="Urban" else 0

    }



    df = pd.DataFrame(
        [data]
    )


    df = df[
        feature_names
    ]


    return df




# =====================================================
# PART 4: PREDICTION
# =====================================================


st.subheader(
    "🔍 Loan Evaluation"
)



if st.button(
    "Predict Loan Status",
    use_container_width=True
):


    input_df = prepare_input()



    prediction = model.predict(
        input_df
    )[0]



    probability = model.predict_proba(
        input_df
    )[0][1]



    if prediction==1:

        status="APPROVED"

    else:

        status="REJECTED"



    if probability >=0.8:

        risk="Low Risk"

    elif probability>=0.5:

        risk="Medium Risk"

    else:

        risk="High Risk"



    st.divider()



    c1,c2,c3 = st.columns(3)



    c1.metric(
        "Decision",
        status
    )


    c2.metric(
        "Approval Probability",
        f"{probability*100:.2f}%"
    )


    c3.metric(
        "Risk Level",
        risk
    )



    st.progress(
        float(probability)
    )



    st.subheader(
        "📋 Application Summary"
    )



    summary=pd.DataFrame(

        {

        "Parameter":
        [
            "Applicant Income",
            "Co Applicant Income",
            "Loan Amount",
            "Loan Term",
            "Credit History",
            "Property Area"
        ],


        "Value":
        [

        f"₹{applicant_income:,.0f}",

        f"₹{coapplicant_income:,.0f}",

        f"{loan_amount_lakh} Lakhs",

        f"{loan_term_years} Years",

        credit_history,

        property_area

        ]

        }

    )



    st.table(summary)



    if prediction==1:

        st.success(
        """
        ✅ Loan application shows approval possibility.

        Final approval requires document verification.
        """
        )

    else:

        st.error(
        """
        ❌ Loan application shows rejection risk.

        Consider reducing loan amount or improving credit profile.
        """
        )