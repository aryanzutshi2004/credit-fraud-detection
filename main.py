import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

# Optional: set page config
st.set_page_config(page_title="Credit Fraud Detector", layout="wide")

# App title
st.title("Credit Fraud Detector")
st.subheader("Enter applicant details below to check credit default risk")

# Layout
row1 = st.columns(3)
row2 = st.columns(3)
row3 = st.columns(3)
row4 = st.columns(3)

# User inputs

with row1[0]:
    age = st.number_input("Age", 18, 100)
with row1[1]:
    income = st.number_input("Income", min_value=0, value=1200000)
with row1[2]:
    loan_amount = st.number_input("Loan Amount", min_value=0, value=2400000)

with row2[0]:
    if income > 0:
        loan_to_inc = np.round(loan_amount / income, 2)
    else:
        loan_to_inc = 0.0
    st.text("Loan to Income Ratio:")
    st.text(loan_to_inc)

with row2[1]:
    loan_tenure = st.number_input("Loan Tenure (Months)", min_value=0)
with row2[2]:
    avg_dpd = np.round(st.number_input("AVG DPD", min_value=0.0), 2)

with row3[0]:
    delinquency_ratio = st.number_input("Delinquency Ratio", min_value=0.0, max_value=1.0)
with row3[1]:
    credit_util_ratio = np.round(st.number_input("Credit Utilization Ratio", min_value=0.0, max_value=1.0), 2)
with row3[2]:
    open_loan_acc = st.number_input("Open Loan Accounts", min_value=0)

with row4[0]:
    residence_type = st.selectbox("Residence Type", ["Owned", "Mortgage", "Rented"])
with row4[1]:
    loan_purpose = st.selectbox("Loan Purpose", ['Personal', 'Education', 'Home', 'Auto'])
with row4[2]:
    loan_type = st.selectbox("Loan Type", ["Secured", "Unsecured"])

# Creating user data dict
user_dict = {
    "age": age,
    "loan_tenure_months": loan_tenure,
    "number_of_open_accounts": open_loan_acc,
    "credit_utilization_ratio": credit_util_ratio,
    "loan_to_inc_ratio": loan_to_inc,
    "delinquent_months_ratio": delinquency_ratio,
    "avg_dpd": avg_dpd,
    "residence_type": residence_type,
    "loan_purpose": loan_purpose,
    "loan_type": loan_type
}

df = pd.DataFrame([user_dict])

# Loading model, scaler, columns to scale, and model features
artifacts = load("./artifacts/final_dict.joblib")
cols_to_scale = artifacts["columns_to_scale"]
scaler = artifacts["scaler"]
model = artifacts["model"]
features = artifacts["features"]

# extra cols for scaling
extra_cols = list(set(cols_to_scale) - set(df.columns))
df[extra_cols] = 0

# Scaling only required columns
df[cols_to_scale] = scaler.transform(df[cols_to_scale])

# Aligning with training columns
df2 = pd.get_dummies(df, dtype=int).reindex(columns=features, fill_value=0)

# # Debug logs
# st.subheader("Debugging / Logging")
# with st.expander("Show User Input Data"):
#     st.write(df)
# with st.expander("Show Processed Model Input Data"):
#     st.write(df2)

# Prediction
if st.button("Check Default Risk"):
    prediction = model.predict(df2)[0]
    result = "may be a defaulter" if prediction == 1 else "is not likely to default"

    # Display result
    if prediction == 0:
        st.success(f"The person {result}.")
    else:
        st.error(f"The person {result}.")

    # Show probability if available
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(df2)[0][1]
        st.info(f"Estimated probability of default: {prob:.2%}")
