# app.py
"""
Streamlit app for Loan Approval DSS.
Loads loan_pipeline.pkl and provides UI for user input, prediction and simple explanation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

# ---------- CONFIG ----------
MODEL_PATH = "best_model_pipeline.pkl"
THRESHOLD = 0.5   # probability threshold for approve

# Load model name and metrics
try:
    with open("best_model_info.txt", "r") as f:
        model_info = f.read()
except:
    model_info = "Model information not available"

# ---------- UTILS ----------
def load_pipeline():
    return joblib.load(MODEL_PATH)

def prettify_label(s):
    return s.replace("_", " ").title()

def get_feature_importance(pipeline, X_df, model_type=None):
    """Get feature importance based on model type and specific input"""
    pre = pipeline.named_steps["preprocessor"]
    clf = pipeline.named_steps["clf"]
    feature_names = pre.get_feature_names_out()
    X_transformed = pre.transform(X_df)
    
    # Convert sparse matrix to dense if needed
    X_transformed = X_transformed.toarray() if hasattr(X_transformed, 'toarray') else X_transformed
    
    # Use SHAP for feature importance (works for both RF and XGBoost)
    try:
        import shap
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_transformed)
        
        # Get SHAP values for class 1 (Approved)
        if isinstance(shap_values, list):
            feature_importance = np.abs(shap_values[1][0])  # Take absolute values for importance
            contribution = shap_values[1][0]
        else:
            feature_importance = np.abs(shap_values[0])
            contribution = shap_values[0]
            
        return pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance,
            'contribution': contribution
        }).sort_values('importance', ascending=False)
        
    except Exception as e:
        # Fallback to traditional feature importance if SHAP fails
        if hasattr(clf, 'feature_importances_'):
            importance = clf.feature_importances_
            return pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
        else:
            return None
    
    return None

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
    Returns per-feature contributions using SHAP for tree-based models.
    Falls back to feature_importances_ if SHAP fails.
    Always includes a 'type' column.
    """
    import shap
    pre = pipeline.named_steps["preprocessor"]
    clf = pipeline.named_steps["clf"]

    try:
        # Preprocess applicant
        X_trans = pre.transform(X_df)
        feature_names = pre.get_feature_names_out()

        # SHAP for tree-based model
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_trans)

        # Get shap values for class 1 (Approved)
        if isinstance(shap_values, list):
            shap_contrib = shap_values[1][0]
        else:
            shap_contrib = shap_values[0]

        contrib_df = pd.DataFrame({
            "feature": feature_names,
            "contribution": shap_contrib
        }).sort_values("contribution", ascending=False)
        contrib_df["type"] = "shap"
        return contrib_df

    except Exception as e:
        # Fallback: feature importances
        if hasattr(clf, "feature_importances_"):
            feature_names = pre.get_feature_names_out()
            contrib_df = pd.DataFrame({
                "feature": feature_names,
                "importance": clf.feature_importances_
            }).sort_values("importance", ascending=False)
            contrib_df["type"] = "tree"
            return contrib_df

        else:
            st.warning(f"Explanation unavailable: {e}")
            return None


# ---------- APP ----------
st.set_page_config(page_title="Loan Approval DSS", layout="centered")
st.title("Loan Approval Decision Support System")
st.write("Enter applicant details and get approval probability, decision and top reasons.")

# Display model info
st.markdown("### Model Information")
st.text(model_info)

# Load pipeline
pipeline = load_pipeline()

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
education = st.selectbox("Education", options=["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", options=["No", "Yes"])

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

    # Get feature importance
    importance_df = get_feature_importance(pipeline, X_input)
    
    if importance_df is not None:
        st.subheader("Feature Importance Analysis")
        
        # Display top 10 most important features
        top_features = importance_df.head(10)
        
        # Create a bar chart
        st.bar_chart(top_features.set_index("feature")["importance"])
        
        # List the top features with their importance values and contributions
        st.markdown("**Top influential features for this prediction:**")
        for _, row in top_features.iterrows():
            feature_name = prettify_label(row['feature'])
            importance = row['importance']
            
            st.markdown(
                f"- **{feature_name}** :     {importance:.4f}"
            )