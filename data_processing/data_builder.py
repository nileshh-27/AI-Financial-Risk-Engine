import numpy as np
import pandas as pd

# -----------------------------
# Config
# -----------------------------
np.random.seed(42)
N = 500000  # number of customers

# -----------------------------
# 1. Demographics
# -----------------------------
age = np.clip(np.random.normal(35, 10, N).astype(int), 18, 70)

employment_type = np.random.choice(
    ["salaried", "self_employed", "unemployed"],
    size=N,
    p=[0.65, 0.25, 0.10]
)

annual_income = np.random.lognormal(mean=11, sigma=0.5, size=N).astype(int)

credit_history_years = np.array([
    np.random.randint(1, max(2, a - 17)) for a in age
])

dependents = np.clip(np.random.poisson(1.2, N), 0, 5)

education_level = np.random.choice(
    ["high_school", "graduate", "post_graduate"],
    size=N,
    p=[0.3, 0.45, 0.25]
)

region = np.random.choice(
    ["metro", "urban", "rural"],
    size=N,
    p=[0.4, 0.4, 0.2]
)

# -----------------------------
# 2. Credit Profile
# -----------------------------
total_credit_limit = (annual_income * np.random.uniform(0.3, 1.2, N)).astype(int)
used_credit = (total_credit_limit * np.random.beta(2, 3, N)).astype(int)

loan_count = np.clip(np.random.poisson(2, N), 0, 8)
missed_payments_12m = np.clip(np.random.poisson(0.6, N), 0, 6)

# -----------------------------
# 3. Financial Behaviour
# -----------------------------
monthly_income = annual_income / 12

avg_monthly_spend = (monthly_income * np.random.uniform(0.4, 0.9, N)).astype(int)
total_monthly_debt = (monthly_income * np.random.uniform(0.1, 0.7, N)).astype(int)

savings_rate = np.clip(
    1 - (avg_monthly_spend / monthly_income),
    0.01,
    0.6
)

# -----------------------------
# 4. Derived Metrics
# -----------------------------
credit_utilization = used_credit / total_credit_limit
debt_to_income = total_monthly_debt / monthly_income

# -----------------------------
# 5. Shock Events
# -----------------------------
job_loss_flag = np.random.binomial(1, 0.08, N)
medical_expense_flag = np.random.binomial(1, 0.12, N)

income_drop_pct = np.where(
    job_loss_flag == 1,
    np.random.uniform(0.2, 0.6, N),
    np.random.uniform(0.0, 0.1, N)
)

# -----------------------------
# 6. Risk Logic (Ground Truth)
# -----------------------------
risk_score = (
    0.05
    + 0.35 * (debt_to_income > 0.6)
    + 0.25 * (credit_utilization > 0.75)
    + 0.30 * (missed_payments_12m >= 2)
    + 0.15 * (savings_rate < 0.1)
    + 0.25 * job_loss_flag
)

risk_score = np.clip(risk_score, 0, 0.95)

default_12m = np.random.binomial(1, risk_score)

risk_class = np.where(
    risk_score < 0.3, "low",
    np.where(risk_score < 0.6, "medium", "high")
)

# -----------------------------
# 7. Build Dataset
# -----------------------------
df = pd.DataFrame({
    "customer_id": [f"C{100000+i}" for i in range(N)],
    "age": age,
    "employment_type": employment_type,
    "annual_income": annual_income,
    "credit_history_years": credit_history_years,
    "dependents": dependents,
    "education_level": education_level,
    "region": region,
    "total_credit_limit": total_credit_limit,
    "used_credit": used_credit,
    "credit_utilization": credit_utilization.round(3),
    "loan_count": loan_count,
    "avg_monthly_spend": avg_monthly_spend,
    "total_monthly_debt": total_monthly_debt,
    "debt_to_income": debt_to_income.round(3),
    "savings_rate": savings_rate.round(3),
    "missed_payments_12m": missed_payments_12m,
    "job_loss_flag": job_loss_flag,
    "medical_expense_flag": medical_expense_flag,
    "income_drop_pct": income_drop_pct.round(2),
    "risk_score": risk_score.round(3),
    "risk_class": risk_class,
    "default_12m": default_12m
})

# -----------------------------
# 8. Save
# -----------------------------
df.to_csv("synthetic_financial_risk_dataset1.csv", index=False)

print("Dataset generated:", df.shape)
