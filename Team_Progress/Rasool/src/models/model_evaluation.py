import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (precision_recall_curve,
    average_precision_score, f1_score)
import matplotlib.pyplot as plt
import joblib, sys
sys.path.insert(0, '.')
from tests.Files.src.features.preprocessing import preprocess_for_ml

df = pd.read_csv('data/processed/telco_features.csv')
X_train, X_test, y_train, y_test, _, _, _ = preprocess_for_ml(df)

# 5-Fold Cross-Validation
model = joblib.load('models/production_model.pkl')
cv_scores = cross_val_score(model, X_train, y_train,
                             cv=5, scoring='roc_auc', n_jobs=-1)
print(f'5-Fold CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')
print(f'Individual folds: {[round(s,4) for s in cv_scores]}')

# Threshold Optimisation
y_prob = model.predict_proba(X_test)[:, 1]
thresholds = np.arange(0.1, 0.9, 0.05)
f1_scores = [f1_score(y_test, (y_prob >= t).astype(int))
             for t in thresholds]
best_t = thresholds[np.argmax(f1_scores)]
best_f1 = max(f1_scores)
print(f'Best threshold: {best_t:.2f}  F1: {best_f1:.4f}')
joblib.dump(float(best_t), 'models/optimal_threshold.pkl')

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_prob)
ap = average_precision_score(y_test, y_prob)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(recall, precision, 'b-', lw=2, label=f'AP={ap:.3f}')
axes[0].set_xlabel('Recall')
axes[0].set_ylabel('Precision')
axes[0].set_title('Precision-Recall Curve')
axes[0].legend()

axes[1].plot(thresholds, f1_scores, 'g-', lw=2)
axes[1].axvline(best_t, color='red', ls='--',
                label=f'Best: {best_t:.2f}')
axes[1].set_xlabel('Threshold')
axes[1].set_ylabel('F1 Score')
axes[1].set_title('Threshold vs F1 Score')
axes[1].legend()

plt.tight_layout()
plt.savefig('reports/figures/13_threshold_optimisation.png', dpi=150)
plt.show()
print(f'Optimal threshold saved: {best_t:.2f}')