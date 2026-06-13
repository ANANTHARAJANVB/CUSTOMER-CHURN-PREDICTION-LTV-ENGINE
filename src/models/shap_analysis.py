import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib, sys
sys.path.insert(0, '.')
from tests.Files.src.features.preprocessing import preprocess_for_ml

model = joblib.load('models/production_model.pkl')
df = pd.read_csv('data/processed/telco_features.csv')
X_train, X_test, y_train, y_test, _, _, _ = preprocess_for_ml(df)

print('Calculating SHAP values (30-60 seconds)...')
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Chart 14: SHAP Summary (beeswarm)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, show=False)
plt.title('SHAP Summary Plot — Global Feature Impact',
          fontweight='bold')
plt.tight_layout()
plt.savefig('reports/figures/14_shap_summary.png', dpi=150,
            bbox_inches='tight')
plt.show()

# Chart 15: SHAP Bar (mean absolute)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, plot_type='bar', show=False)
plt.title('Mean |SHAP| — Feature Importance', fontweight='bold')
plt.tight_layout()
plt.savefig('reports/figures/15_shap_bar.png', dpi=150,
            bbox_inches='tight')
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
        values=shap_values[high_risk_idx],
        base_values=explainer.expected_value,
        data=X_test.iloc[high_risk_idx],
        feature_names=X_test.columns.tolist()
    ), max_display=15, show=False
)
plt.title(f'SHAP Waterfall — Highest Risk Customer '
          f'(P={y_prob[high_risk_idx]:.2f})', fontweight='bold')
plt.tight_layout()
plt.savefig('reports/figures/16_shap_waterfall_highrisk.png',
            dpi=150, bbox_inches='tight')
plt.show()

joblib.dump(explainer, 'models/shap_explainer.pkl')
print('SHAP explainer saved to models/shap_explainer.pkl')
print('=== SHAP Analysis Complete ===')