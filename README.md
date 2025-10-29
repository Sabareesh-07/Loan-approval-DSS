# 💳 Loan Approval Decision Support System (DSS)

A Machine Learning-powered **Decision Support System** that predicts **loan approval** based on applicant details such as income, credit score, and assets.  
Built using **Streamlit**, **scikit-learn**, and **SHAP**, this app not only predicts approval outcomes but also explains **why** a loan is approved or rejected — making it transparent, interpretable, and useful for decision-makers.

---

## 🌐 Live App
👉 [**Launch the App on Streamlit Cloud**](https://sabareesh-07-loan-approval-dss-app-27y45u.streamlit.app/)

---

## 🧠 Project Overview

This project aims to build a **Decision Support System (DSS)** for loan approval using Machine Learning techniques.  
The system helps financial institutions assess loan applications efficiently while ensuring explainable decision-making.

### ✳️ Objectives
- Predict whether a loan should be **approved or rejected**.
- Compute the **approval probability** for each applicant.
- Provide **interpretable explanations** of feature influences (using SHAP).
- Serve as an easy-to-use **interactive Streamlit web app**.

---

## 🧩 Features

✅ Machine Learning model trained on real-world loan data  
✅ Computes applicant-based approval probability  
✅ Displays the **top influential factors** affecting each decision  
✅ Uses **Random Forest Classifier** for non-linear decision boundaries  
✅ Integrated **SHAP explanations** for model interpretability  
✅ Clean and responsive **Streamlit interface**  

---

## 🧱 Tech Stack

| Component | Description |
|------------|-------------|
| **Language** | Python 3 |
| **Framework** | Streamlit |
| **ML Library** | scikit-learn (Random Forest) |
| **Explainability** | SHAP |
| **Data Handling** | pandas, numpy |
| **Serialization** | joblib |

---

## ⚙️ How It Works

1. **train_model.py**
   - Loads and processes the dataset (`dataset1.csv`)
   - Engineers new features such as:
     - `total_assets`
     - `loan_to_income_ratio`
     - `loan_to_assets_ratio`
   - Trains a **Random Forest model**
   - Saves the pipeline as `loan_pipeline.pkl`

2. **app.py**
   - Interactive **Streamlit web UI**
   - Takes applicant details as inputs
   - Predicts loan approval probability
   - Displays decision outcome (Approve / Reject)
   - Shows **feature-level explanations** using SHAP

---

## 🧾 Example Input
| Field | Example Value |
|--------|----------------|
| No. of Dependents | 1 |
| Annual Income | 600000 |
| Loan Amount | 400000 |
| Loan Term | 120 |
| CIBIL Score | 750 |
| Education | Graduate |
| Self Employed | No |

**Predicted Output:**

Decision: APPROVE ✅
Approval Probability: 90.3%
Top Influential Features:
* cibil_score
* loan_to_income_ratio
* income_annum

---

## 🧩 Installation & Local Run

### 🔧 1. Clone this repository
```bash
git clone https://github.com/<your-username>/loan-approval-dss.git
cd loan-approval-dss
````

### 🧠 2. Install dependencies

```bash
pip install -r requirements.txt
```

### ▶️ 3. Run the Streamlit app

```bash
streamlit run app.py
```

---

## ☁️ Deployment

The app is hosted on **Streamlit Cloud**:
➡️ [https://sabareesh-07-loan-approval-dss-app-27y45u.streamlit.app/](https://sabareesh-07-loan-approval-dss-app-27y45u.streamlit.app/)

To deploy your own version:

1. Push this project to your GitHub repository
2. Go to [https://share.streamlit.io](https://share.streamlit.io)
3. Select your repo → Choose `app.py` → Deploy 🚀

---

## 📊 Model Performance (Training Metrics)

| Metric               | Value     |
| -------------------- | --------- |
| Accuracy             | **91.1%** |
| ROC AUC              | **0.97**  |
| Precision (Approved) | 0.92      |
| Recall (Approved)    | 0.94      |

---

## 🧠 Explainability Example (SHAP)

Positive contributions (increase approval):

* High CIBIL score
* High annual income

Negative contributions (decrease approval):

* Large loan amount
* High loan-to-income ratio

---

## 👨‍💻 Author

**Sabareesh**
- 📧 [GitHub](https://github.com/sabareesh-07)
- 🎓 Project: *Loan Approval Decision Support System*
- 🧩 Course: Integrated MSc in Computational Statistics and Data Analysis

---

## 🪪 License

This project is open-source and available under the **MIT License**.

---

## 🌟 Acknowledgments

* Streamlit for the web framework
* scikit-learn for ML pipeline tools
* SHAP for model explainability
* Dataset inspired by open loan prediction datasets (Kaggle-style)

---

> 💬 *“A decision support system is only as good as its transparency.”*
> This project bridges the gap between automation and interpretability in financial risk assessment.

---


Would you like me to also generate a **short project description and tags** (for GitHub’s sidebar & metadata), so your repo looks perfect in search results?
```

