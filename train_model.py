#pipeline for Loan Approval DSS using Logistic Regression

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score


DATA_PATH = "dataset1.csv"
PIPELINE_PATH = "loan_pipeline.pkl"
COEFS_PATH = "feature_coefs.csv"
RANDOM_STATE = 42

# ---------- HELPERS ----------
def map_status_to_binary(s):
    """Map many possible target encodings to 1 = approved, 0 = rejected."""
    if s =="Approved":
        return 1
    return 0


# ---------- LOAD ----------
df = pd.read_csv(DATA_PATH)
print("Loaded data:", DATA_PATH, "shape:", df.shape,"\nColumns:\n",df.columns)

# ---------- BASIC CLEANUP ----------
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
# drop identifier column
df = df.drop(columns=["loan_id"])

# standardize target column name if whitespace etc exists
target_col =  "loan_status"
print("Using target column:", target_col)

# map to binary
df["__target__"] = df[target_col].apply(map_status_to_binary)
print("Target distribution:\n", df["__target__"].value_counts(dropna=False))

# Drop rows with missing target
df = df[~df["__target__"].isna()].copy()


# show missing values
print("\nMissing values per column:")
print(df.isna().sum())

# ---------- FEATURE ENGINEERING ----------
# total assets
asset_cols = ["residential_assets_value", "commercial_assets_value", "luxury_assets_value", "bank_asset_value"]
df["total_assets"] = df[asset_cols].fillna(0).sum(axis=1)

# ratios
df["loan_to_income_ratio"] = df["loan_amount"] /df["income_annum"]
df["loan_to_assets_ratio"] = df["loan_amount"] / df["total_assets"]

# Binary encoding for categorical columns
df['self_employed'] = df['self_employed'].map({'Yes': 1, 'No': 0})
df['education'] = df['education'].map({'Graduate': 1, 'Not Graduate': 0})

# choose features
numeric_features = ["no_of_dependents", "income_annum", "loan_amount", "loan_term", "cibil_score", "total_assets", "loan_to_income_ratio", "loan_to_assets_ratio"]
categorical_features = ["education", "self_employed"]

print("\nNumeric features used:", numeric_features)
print("Categorical features used:", categorical_features)

# ---------- PREPROCESSOR ----------
numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

# ---------- MODEL PIPELINE ----------
clf = LogisticRegression(solver="liblinear", max_iter=1000)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("clf", clf),
])

# ---------- PREPARE X, y ----------
feature_list = numeric_features + categorical_features
X = df[feature_list].copy()
y = df["__target__"].astype(int).copy()

# split
try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE,stratify=y)
except ValueError:
    print("Stratify Failed!")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE)

print("\nTrain shape:", X_train.shape, "Test shape:", X_test.shape)
print("y_train value counts:", y_train.value_counts())

# ---------- TRAIN ----------
pipeline.fit(X_train, y_train)
print("\nModel trained.")

# ---------- EVALUATION ----------
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))
print("Accuracy:", accuracy_score(y_test, y_pred))
try:
    print("ROC AUC:", roc_auc_score(y_test, y_prob))
except Exception:
    pass
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ---------- SAVE PIPELINE ----------
joblib.dump(pipeline, PIPELINE_PATH)
print(f"\nSaved pipeline to {PIPELINE_PATH}")

# ---------- EXTRACT FEATURE NAMES & COEFFICIENTS ----------
def get_feature_names_from_preprocessor(preprocessor):
    # numeric names
    num_names = numeric_features
    # categorical one-hot names
    cat_ohe = []
    if categorical_features:
        # access onehot encoder inside transformer
        try:
            ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
            cat_ohe = list(ohe.get_feature_names_out(categorical_features))
        except Exception:
            # fallback (older sklearn)
            ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
            cat_ohe = []
    return num_names + cat_ohe

try:
    feature_names = get_feature_names_from_preprocessor(pipeline.named_steps["preprocessor"])
    coefs = pipeline.named_steps["clf"].coef_[0]
    if len(feature_names) == len(coefs):
        coef_df = pd.DataFrame({"feature": feature_names, "coef": coefs})
        coef_df = coef_df.reindex(coef_df.coef.abs().sort_values(ascending=False).index)
        coef_df.to_csv(COEFS_PATH, index=False)
        print(f"Saved feature coefficients to {COEFS_PATH}")
    else:
        print("Warning: feature name count doesn't match coefficient count. Skipping saving coefs.")
except Exception as e:
    print("Could not extract feature names/coeffs:", e)

print("\nTraining complete. Pipeline saved. Use app.py to interact with the model.")
