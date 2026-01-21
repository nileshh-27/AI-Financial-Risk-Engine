import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
    accuracy_score,
    classification_report
)

# =====================
# 1. Load model & data
# =====================
model = joblib.load("xgb_financial_risk_model.pkl")
df = pd.read_csv("synthetic_financial_risk_dataset.csv")

TARGET = "risk_score"

# =====================
# 2. Prepare features
# =====================
X = df.drop(columns=["risk_score", "risk_class"])
y = df[TARGET]

if "customer_id" in X.columns:
    X = X.drop(columns=["customer_id"])

cat_cols = X.select_dtypes(include=["object"]).columns
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# =====================
# 3. Train / test split
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# =====================
# 4. Regression evaluation
# =====================
preds = model.predict(X_test)

rmse = mean_squared_error(y_test, preds) ** 0.5
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)
evs = explained_variance_score(y_test, preds)

print("\n📊 REGRESSION METRICS")
print("-" * 40)
print(f"RMSE               : {rmse:.6f}")
print(f"MAE                : {mae:.6f}")
print(f"R² Score           : {r2:.6f}")
print(f"Explained Variance : {evs:.6f}")

# =====================
# 5. Accuracy-style evaluation (binning)
# =====================
def risk_bucket(score):
    if score < 0.3:
        return "LOW"
    elif score < 0.6:
        return "MEDIUM"
    else:
        return "HIGH"

true_classes = y_test.apply(risk_bucket)
pred_classes = pd.Series(preds).apply(risk_bucket)

acc = accuracy_score(true_classes, pred_classes)

print("\n📌 CLASSIFICATION VIEW (Derived)")
print("-" * 40)
print(f"Bucket Accuracy: {acc * 100:.2f}%\n")
print(classification_report(true_classes, pred_classes))

# =====================
# 6. INTERACTIVE TERMINAL TEST
# =====================
print("\n🧪 MANUAL RISK SCORE TEST")
print("-" * 40)

def get_input(prompt, cast=float):
    return cast(input(f"{prompt}: ").strip())

sample = {
    "age": get_input("Age", int),
    "annual_income": get_input("Annual Income"),
    "credit_history_years": get_input("Credit History (years)"),
    "dependents": get_input("Dependents", int),
    "total_credit_limit": get_input("Total Credit Limit"),
    "used_credit": get_input("Used Credit"),
    "credit_utilization": get_input("Credit Utilization (0-1)"),
    "loan_count": get_input("Loan Count", int),
    "avg_monthly_spend": get_input("Avg Monthly Spend"),
    "total_monthly_debt": get_input("Total Monthly Debt"),
    "debt_to_income": get_input("Debt to Income"),
    "savings_rate": get_input("Savings Rate (0-1)"),
    "missed_payments_12m": get_input("Missed Payments (12m)", int),
    "job_loss_flag": get_input("Job Loss Flag (0/1)", int),
    "medical_expense_flag": get_input("Medical Expense Flag (0/1)", int),
    "income_drop_pct": get_input("Income Drop %")
}

# Default categorical values
sample["employment_type"] = "salaried"
sample["education_level"] = "graduate"
sample["region"] = "urban"

sample_df = pd.DataFrame([sample])
sample_df = pd.get_dummies(sample_df)
sample_df = sample_df.reindex(columns=X.columns, fill_value=0)

pred_score = model.predict(sample_df)[0]

risk_label = risk_bucket(pred_score)

print("\n🎯 RESULT")
print("-" * 40)
print(f"Predicted Risk Score : {pred_score:.4f}")
print(f"Risk Category        : {risk_label}")
