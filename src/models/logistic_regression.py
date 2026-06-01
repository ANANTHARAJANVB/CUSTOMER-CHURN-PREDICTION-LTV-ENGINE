import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, os, sys
sys.path.insert(0, '.')
from src.features.preprocessing import preprocess_for_ml
def train_logistic_regression():
    # Load and preprocess data
    df = pd.read_csv('data/processed/telco_features.csv')
    X_train, X_test, y_train, y_test, X_tr_sc, X_te_sc, _ = preprocess_for_ml(df)
    # Train model with class_weight='balanced' to handle imbalance
    model = LogisticRegression(
        class_weight='balanced',  # adjusts for 26.5% churn imbalance
        max_iter=1000,            # ensure convergence
        random_state=42
    )
    model.fit(X_tr_sc, y_train)
    # Predictions
    y_pred = model.predict(X_te_sc)
    y_prob = model.predict_proba(X_te_sc)[:, 1]  # probability of churn
    # Metrics
    print('=== Logistic Regression Results ===')
    print(classification_report(y_test, y_pred,
          target_names=['Not Churn','Churn']))
    auc = roc_auc_score(y_test, y_prob)
    print(f'AUC-ROC Score: {auc:.4f}')
    # Confusion Matrix plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['No Churn','Churn'])
    disp.plot(ax=axes[0], cmap='Blues')
    axes[0].set_title('Confusion Matrix — Logistic Regression')
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[1].plot(fpr, tpr, 'b-', lw=2, label=f'LR (AUC={auc:.3f})')
    axes[1].plot([0,1],[0,1],'k--', label='Random')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curve')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig('reports/figures/10_lr_evaluation.png', dpi=150, bbox_inches='tight')
    plt.show()
    return model, auc
if __name__ == '__main__':
    model, auc = train_logistic_regression()
    joblib.dump(model, 'models/logistic_regression.pkl')
    print(f'Model saved. AUC: {auc:.4f}')