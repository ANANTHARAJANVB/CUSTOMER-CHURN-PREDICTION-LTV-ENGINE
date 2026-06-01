import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import joblib, sys
sys.path.insert(0, '.')
from src.features.preprocessing import preprocess_for_ml

def train_xgboost():
    df = pd.read_csv('data/processed/telco_features.csv')
    X_train, X_test, y_train, y_test, _, _, _ = preprocess_for_ml(df)

    # scale_pos_weight handles imbalance:
    # ratio of non-churners to churners = 5174 / 1869 ≈ 2.77
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = neg / pos
    print(f'scale_pos_weight: {spw:.2f}')

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=50)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    print('=== XGBoost Results ===')
    print(classification_report(y_test, y_pred,
          target_names=['No Churn','Churn']))
    print(f'AUC-ROC: {auc:.4f}')
    return model, auc, y_prob, y_test

def compare_all_models(xgb_auc, y_test, xgb_y_prob):
    import warnings; warnings.filterwarnings('ignore')
    df = pd.read_csv('data/processed/telco_features.csv')
    _, X_test, _, y_test, _, X_te_sc, _ = preprocess_for_ml(df)

    lr = joblib.load('models/logistic_regression.pkl')
    rf = joblib.load('models/random_forest.pkl')
    lr_prob = lr.predict_proba(X_te_sc)[:, 1]
    rf_prob = rf.predict_proba(X_test)[:, 1]

    fig, ax = plt.subplots(figsize=(8, 6))
    for prob, label, color in [
        (lr_prob,     'Logistic Regression', '#2196F3'),
        (rf_prob,     'Random Forest',       '#4CAF50'),
        (xgb_y_prob,  'XGBoost',             '#F44336'),
    ]:
        fpr, tpr, _ = roc_curve(y_test, prob)
        a = roc_auc_score(y_test, prob)
        ax.plot(fpr, tpr, lw=2,
                label=f'{label} (AUC={a:.3f})', color=color)
    ax.plot([0,1],[0,1],'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Model Comparison — ROC Curves', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig('reports/figures/12_model_comparison_roc.png', dpi=150)
    plt.show()

if __name__ == '__main__':
    model, auc, y_prob, y_test = train_xgboost()
    compare_all_models(auc, y_test, y_prob)
    joblib.dump(model, 'models/xgboost_churn.pkl')
    joblib.dump(model, 'models/production_model.pkl')
    print('Saved production model')