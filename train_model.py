# pipeline for Loan Approval DSS using Logistic Regression (fixed version)

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score

DATA_PATH = "dataset1.csv"
PIPELINE_PATH = "loan_pipeline.pkl"
COEFS_PATH = "feature_coefs.csv"
RANDOM_STATE = 42

def map_status_to_binary(s):
    return 1 if s == "Approved" else 0

# ---------- LOAD ----------
df = pd.read_csv(DATA_PATH)
print("Loaded data:", DATA_PATH, "shape:", df.shape)
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
df = df.drop(columns=["loan_id"])

target_col = "loan_status"
df["__target__"] = df[target_col].apply(map_status_to_binary)
df = df[~df["__target__"].isna()].copy()
print("Target distribution:\n", df["__target__"].value_counts())

# ---------- FEATURE ENGINEERING ----------
asset_cols = ["residential_assets_value", "commercial_assets_value", "luxury_assets_value", "bank_asset_value"]
df["total_assets"] = df[asset_cols].fillna(0).sum(axis=1)

# Safe ratio calculations
df["loan_to_income_ratio"] = (df["loan_amount"] / (df["income_annum"] + 1e-9)).clip(upper=5)
df["loan_to_assets_ratio"] = (df["loan_amount"] / (df["total_assets"] + 1e-9)).clip(upper=5)

# Replace inf/nan
df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

numeric_features = ["no_of_dependents", "income_annum", "loan_amount", "loan_term",
                    "cibil_score", "total_assets", "loan_to_income_ratio", "loan_to_assets_ratio"]
categorical_features = ["education", "self_employed"]

print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)

# ---------- PREPROCESSOR ----------
numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

# ---------- MODEL ----------
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    class_weight="balanced",
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("clf", clf),
])

# ---------- SPLIT ----------
X = df[numeric_features + categorical_features]
y = df["__target__"].astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# ---------- TRAIN ----------
pipeline.fit(X_train, y_train)
print("\nRandom Forest model trained.")

# ---------- EVALUATION ----------
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

import numpy as np
print("\nMean predicted probability (Approved class):", np.mean(y_prob))

# Debug probability sanity
print("\nMean predicted probability (Approved class):", np.mean(y_prob))
if np.mean(y_prob) < 0.05 or np.mean(y_prob) > 0.95:
    print("⚠️ Warning: Model outputs extremely confident probabilities — consider scaling or nonlinear model.")

# ---------- SAVE ----------
joblib.dump(pipeline, PIPELINE_PATH)
print(f"\nSaved pipeline to {PIPELINE_PATH}")
