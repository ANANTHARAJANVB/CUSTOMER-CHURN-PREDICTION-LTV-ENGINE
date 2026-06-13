import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib
import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '.')
from features.preprocessing import preprocess_for_ml

# Ensure output directory exists before saving charts
os.makedirs('reports/figures', exist_ok=True)

model = joblib.load('models/production_model.pkl')
df = pd.read_csv('data/processed/telco_features.csv')
X_train, X_test, y_train, y_test, _, _, _ = preprocess_for_ml(df)

print('Calculating SHAP values (30-60 seconds)...')
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# For binary classification, shap_values has shape (n_samples, n_features, 2).
# We isolate index 1 to focus specifically on the 'Churn' class impact.
shap_values_churn = shap_values[:, :, 1]
expected_value_churn = explainer.expected_value[1]

# Chart 14: SHAP Summary (beeswarm)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_churn, X_test, show=False)
plt.title('SHAP Summary Plot — Global Feature Impact (Churn)', fontweight='bold')
plt.tight_layout()
plt.savefig('reports/figures/14_shap_summary.png', dpi=150, bbox_inches='tight')
plt.show()

# Chart 15: SHAP Bar (mean absolute)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_churn, X_test, plot_type='bar', show=False)
plt.title('Mean |SHAP| — Feature Importance (Churn)', fontweight='bold')
plt.tight_layout()
plt.savefig('reports/figures/15_shap_bar.png', dpi=150, bbox_inches='tight')
plt.show()

# Chart 16: Waterfall for highest-risk customer
y_prob = model.predict_proba(X_test)[:, 1]
high_risk_idx = np.argmax(y_prob)
print(f'Highest risk customer index: {high_risk_idx}')
print(f'Predicted churn probability: {y_prob[high_risk_idx]:.3f}')
print(f'Actual label: {y_test.iloc[high_risk_idx]}')

plt.figure(figsize=(12, 7))
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values_churn[high_risk_idx],
        base_values=expected_value_churn,
        data=X_test.iloc[high_risk_idx],
        feature_names=X_test.columns.tolist()
    ), max_display=15, show=False
)
plt.title(f'SHAP Waterfall — Highest Risk Customer (P={y_prob[high_risk_idx]:.2f})', fontweight='bold')
plt.tight_layout()
plt.savefig('reports/figures/16_shap_waterfall_highrisk.png', dpi=150, bbox_inches='tight')
plt.show()

joblib.dump(explainer, 'models/shap_explainer.pkl')
print('SHAP explainer saved to models/shap_explainer.pkl')
print('=== SHAP Analysis Complete ===')