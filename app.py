# app.py
"""
Streamlit app for Loan Approval DSS.
Loads loan_pipeline.pkl and provides UI for user input, prediction and simple explanation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------- CONFIG ----------
PIPELINE_PATH = "loan_pipeline.pkl"   # should match training output
THRESHOLD = 0.5   # probability threshold for approve

# ---------- UTILS ----------
def load_pipeline(path=PIPELINE_PATH):
    return joblib.load(path)

def prettify_label(s):
    return s.replace("_", " ").title()

def get_feature_names_from_preprocessor(preprocessor, numeric_features, categorical_features):
    num_names = numeric_features
    cat_ohe = []
    if categorical_features:
        try:
            ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
            cat_ohe = list(ohe.get_feature_names_out(categorical_features))
        except Exception:
            cat_ohe = []
    return num_names + cat_ohe

def explain_contributions(pipeline, X_df, numeric_features, categorical_features):
    """
    Returns a DataFrame of feature contributions for the single input row in X_df.
    Contribution = coef * transformed_feature_value
    This uses the pipeline's preprocessor and logistic regression coefficients.
    """
    pre = pipeline.named_steps['preprocessor']
    clf = pipeline.named_steps['clf']
    X_trans = pre.transform(X_df)   # 2D array
    coefs = clf.coef_[0]
    try:
        feature_names = get_feature_names_from_preprocessor(pre, numeric_features, categorical_features)
        contrib = X_trans[0] * coefs
        contrib_df = pd.DataFrame({"feature": feature_names, "contribution": contrib})
        contrib_df = contrib_df.sort_values("contribution", ascending=False)
        return contrib_df
    except Exception:
        return None

# ---------- APP ----------
st.set_page_config(page_title="Loan Approval DSS", layout="centered")
st.title("Loan Approval Decision Support System")
st.write("Enter applicant details and get approval probability, decision and top reasons.")

# Load pipeline
pipeline = load_pipeline(PIPELINE_PATH)

# Attempt to detect which features were used during training
# We rely on the same feature lists used in training script.
# Update these lists if training script used different names.
numeric_features = []
for c in ["no_of_dependents", "income_annum", "loan_amount", "loan_term", "cibil_score", "total_assets", "loan_to_income_ratio", "loan_to_assets_ratio"]:
    # show the inputs even if not in pipeline training features; training script used subset detection
    numeric_features.append(c)

categorical_features = []
for c in ["education", "self_employed"]:
    categorical_features.append(c)

# ---------- USER INPUT ----------
st.header("Applicant Details")

# numeric inputs (use default 0 or typical values)
no_of_dependents = st.number_input("Number of dependents", min_value=0, max_value=10, value=0, step=1)
income_annum = st.number_input("Annual Income (INR)", min_value=0.0, value=300000.0, step=10000.0, format="%.2f")
loan_amount = st.number_input("Loan Amount (INR)", min_value=0.0, value=500000.0, step=1000.0, format="%.2f")
loan_term = st.number_input("Loan Term (months)", min_value=1, value=120, step=1)
cibil_score = st.number_input("CIBIL/Credit Score", min_value=0.0, max_value=1000.0, value=700.0, step=1.0)
residential_assets_value = st.number_input("Residential assets value (INR)", min_value=0.0, value=0.0, step=10000.0)
commercial_assets_value = st.number_input("Commercial assets value (INR)", min_value=0.0, value=0.0, step=10000.0)
luxury_assets_value = st.number_input("Luxury assets value (INR)", min_value=0.0, value=0.0, step=10000.0)
bank_asset_value = st.number_input("Bank asset value (savings) (INR)", min_value=0.0, value=50000.0, step=1000.0)

# categorical inputs
education = st.selectbox("Education", options=["Graduate", "Not Graduate", "Unknown"])
self_employed = st.selectbox("Self Employed", options=["No", "Yes", "Unknown"])

# when user clicks Predict
if st.button("Predict Approval"):
    # build DataFrame in same structure as training X (the training script used feature_list = numeric + categorical)
    # create the engineered features same as training
    total_assets = residential_assets_value + commercial_assets_value + luxury_assets_value + bank_asset_value
    loan_to_income_ratio = loan_amount / (income_annum + 1e-9)
    loan_to_assets_ratio = loan_amount / (total_assets + 1e-9)

    # assemble input dict - include the engineered features too
    data = {
        "no_of_dependents": [no_of_dependents],
        "income_annum": [income_annum],
        "loan_amount": [loan_amount],
        "loan_term": [loan_term],
        "cibil_score": [cibil_score],
        "residential_assets_value": [residential_assets_value],
        "commercial_assets_value": [commercial_assets_value],
        "luxury_assets_value": [luxury_assets_value],
        "bank_asset_value": [bank_asset_value],
        "total_assets": [total_assets],
        "loan_to_income_ratio": [loan_to_income_ratio],
        "loan_to_assets_ratio": [loan_to_assets_ratio],
        "education": [education],
        "self_employed": [self_employed]
    }

    X_input = pd.DataFrame(data)

    # Some training runs might not have used all fields; try to select matching columns
    # We will try to pass the full DataFrame; preprocessor will pick required columns
    try:
        proba = pipeline.predict_proba(X_input)[:, 1][0]
        pred = pipeline.predict(X_input)[0]
    except Exception as e:
        st.error(f"Model prediction failed: {e}")
        raise

    decision = "APPROVE" if pred == 1 else "REJECT"
    st.markdown(f"### Decision: **{decision}**")
    st.markdown(f"### Approval Probability: **{proba*100:.1f}%**")

    # explanation via linear contribution
    contrib_df = explain_contributions(pipeline, X_input, numeric_features, categorical_features)
    if contrib_df is not None:
        st.subheader("Top positive contributors (increase approval probability)")
        pos = contrib_df[contrib_df["contribution"] > 0].head(5)
        if not pos.empty:
            for _, row in pos.iterrows():
                st.write(f"- **{row['feature']}** : contribution {row['contribution']:.3f}")
        else:
            st.write("No strong positive contributors detected.")

        st.subheader("Top negative contributors (decrease approval probability)")
        neg = contrib_df[contrib_df["contribution"] < 0].head(5)
        if not neg.empty:
            for _, row in neg.iterrows():
                st.write(f"- **{row['feature']}** : contribution {row['contribution']:.3f}")
        else:
            st.write("No strong negative contributors detected.")
    else:
        st.write("Detailed contribution explanation not available (preprocessor feature names unavailable).")