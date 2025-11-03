# pipeline for Loan Approval DSS

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score

DATA_PATH = "dataset1.csv"
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

# ---------- MODELS ----------
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Initialize models
rf_clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    class_weight="balanced",
)

# Calculate class weight for XGBoost
neg_pos_ratio = len(df[df['__target__'] == 0]) / len(df[df['__target__'] == 1])

xgb_clf = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    scale_pos_weight=neg_pos_ratio  # Handle class imbalance
)

# Create pipelines for both models
rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("clf", rf_clf),
])

xgb_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("clf", xgb_clf),
])

# Dictionary to store all models and their results
models = {
    "Random Forest": rf_pipeline,
    "XGBoost": xgb_pipeline
}

# ---------- SPLIT ----------
X = df[numeric_features + categorical_features]
y = df["__target__"].astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# ---------- TRAIN AND EVALUATE MODELS ----------
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Function to evaluate a model
def evaluate_model(pipeline, X_test, y_test, model_name):
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    print(f"\n{model_name} Results:")
    print("Classification Report:\n", classification_report(y_test, y_pred, digits=4))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_prob))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    
    mean_prob = np.mean(y_prob)
    print(f"\nMean predicted probability (Approved class): {mean_prob:.4f}")
    if mean_prob < 0.05 or mean_prob > 0.95:
        print("⚠️ Warning: Model outputs extremely confident probabilities")
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    return {
        "y_pred": y_pred,
        "y_prob": y_prob,
        "accuracy": accuracy_score(y_test, y_pred),
        "auc": roc_auc,
        "fpr": fpr,
        "tpr": tpr
    }

# Dictionary to store results
results = {}

# Train and evaluate all models
plt.figure(figsize=(10, 8))
for model_name, pipeline in models.items():
    print(f"\nTraining {model_name}...")
    pipeline.fit(X_train, y_train)
    eval_results = evaluate_model(pipeline, X_test, y_test, model_name)
    results[model_name] = eval_results
    
    # Plot ROC curve for this model
    plt.plot(
        eval_results["fpr"], 
        eval_results["tpr"], 
        label=f'{model_name} (AUC = {eval_results["auc"]:.4f})'
    )

# Finish ROC plot
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for All Models')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig('roc_curves.png')
plt.close()

# Compare models
print("\nModel Comparison:")
print(f"{'Model':<20} {'Accuracy':<10} {'ROC AUC':<10}")
print("-" * 40)
for model_name, metrics in results.items():
    print(f"{model_name:<20} {metrics['accuracy']:.4f}    {metrics['auc']:.4f}")

# Find best model based on ROC AUC
best_model_name = max(results.items(), key=lambda x: x[1]['auc'])[0]
best_pipeline = models[best_model_name]
print(f"\nBest performing model: {best_model_name} (ROC AUC: {results[best_model_name]['auc']:.4f})")

# ---------- SAVE ----------
# Save only the best model
print(f"\nSaving best model ({best_model_name})...")
joblib.dump(best_pipeline, "best_model_pipeline.pkl")
with open("best_model_info.txt", "w") as f:
    f.write(f"Best Model: {best_model_name}\n")
    f.write(f"Accuracy: {results[best_model_name]['accuracy']:.4f}\n")
    f.write(f"ROC AUC: {results[best_model_name]['auc']:.4f}\n")
