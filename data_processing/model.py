import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# =====================
# 1. Load dataset
# =====================
df = pd.read_csv("synthetic_financial_risk_dataset.csv")

TARGET = "risk_score"

# =====================
# 2. Split features / target
# =====================
X = df.drop(columns=["risk_score", "risk_class"])
y = df[TARGET]

# Drop ID column
if "customer_id" in X.columns:
    X = X.drop(columns=["customer_id"])

# =====================
# 3. Encode categorical features
# =====================
cat_cols = X.select_dtypes(include=["object"]).columns
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# =====================
# 4. Train / validation split
# =====================
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================
# 5. GPU XGBoost model
# =====================
model = xgb.XGBRegressor(
    objective="reg:squarederror",
    tree_method="hist",
    device="cuda",          # 🔥 GPU training
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=0.1,
    reg_lambda=1.0,
    random_state=42
)

# =====================
# 6. Train model
# =====================
model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=50
)

# =====================
# 7. Evaluate (CPU prediction — correct & safe)
# =====================
val_preds = model.predict(X_val)

rmse = mean_squared_error(y_val, val_preds) ** 0.5
r2 = r2_score(y_val, val_preds)

print(f"\nFinal RMSE: {rmse:.6f}")
print(f"Final R²  : {r2:.6f}")

# =====================
# 8. Save model
# =====================
joblib.dump(model, "xgb_financial_risk_model.pkl")
print("\n✅ Model saved as xgb_financial_risk_model.pkl")
