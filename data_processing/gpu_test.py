import xgboost as xgb

print("XGBoost version:", xgb.__version__)

# Simple GPU test
model = xgb.XGBClassifier(
    tree_method="hist",
    device="cuda"
)

print("GPU support OK")
