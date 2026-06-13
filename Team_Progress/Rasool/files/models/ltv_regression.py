"""
LTV Regression Model — Gradient Boosting Regressor.
Predicts predictive_ltv (future revenue) for each customer.
Run from: Team_Progress\Rasool\files\
Command:  python models/ltv_regression.py
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 55)
print("=== STEP 1: Loading LTV dataset ===")
df = pd.read_csv('data/processed/telco_ltv.csv')
print(f"Loaded {len(df)} rows")

# Drop columns that are targets or derived (not features)
drop_cols = [
    'customerid', 'churn', 'loyalty_segment', 'ltv_segment',
    'historical_ltv', 'total_ltv', 'predictive_ltv',
    'churn_probability', 'expected_remaining_months'
]
drop_cols = [c for c in drop_cols if c in df.columns]

X = df.drop(columns=drop_cols)
y = df['predictive_ltv']

print(f"\n=== STEP 2: Preparing features ===")
print(f"Features before encoding: {X.shape[1]}")
cat_cols = X.select_dtypes('object').columns.tolist()
X = pd.get_dummies(X, columns=cat_cols, drop_first=False)
print(f"Features after encoding:  {X.shape[1]}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

print("\n=== STEP 3: Training Gradient Boosting Regressor ===")
print("(1–3 minutes...)")
model = GradientBoostingRegressor(
    n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\n=== STEP 4: Model Evaluation ===")
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
print(f"MAE:  ${mae:.2f}   — average dollar error")
print(f"RMSE: ${rmse:.2f}   — root mean squared error")
print(f"R²:   {r2:.4f}    — aim for > 0.80")

print("\n=== STEP 5: Saving charts ===")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(y_test, y_pred, alpha=0.25, color='#6d28d9', s=12)
axes[0].plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0].set(xlabel='Actual LTV ($)', ylabel='Predicted LTV ($)',
            title=f'Predicted vs Actual (R²={r2:.3f})')
residuals = y_test - y_pred
axes[1].hist(residuals, bins=50, color='#6d28d9', alpha=0.7)
axes[1].axvline(0, color='red', ls='--')
axes[1].set(xlabel='Residual ($)', ylabel='Count',
            title='Residuals Distribution')
plt.tight_layout()
plt.savefig('reports/figures/17_ltv_regression.png', dpi=150, bbox_inches='tight')
print("Saved: reports/figures/17_ltv_regression.png")

print("\n=== STEP 6: Saving models ===")
joblib.dump(model, 'models/ltv_regression.pkl')
joblib.dump(list(X.columns), 'models/ltv_feature_columns.pkl')
print("Saved: models/ltv_regression.pkl")
print("Saved: models/ltv_feature_columns.pkl")
print("\n✅ Day 16 complete!")