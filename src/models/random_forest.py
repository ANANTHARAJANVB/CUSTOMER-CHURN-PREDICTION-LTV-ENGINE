import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import joblib, sys
sys.path.insert(0, '.')
from tests.Files.src.features.preprocessing import preprocess_for_ml
def train_random_forest():
    df = pd.read_csv('data/processed/telco_features.csv')
    X_train, X_test, y_train, y_test, _, _, _ = preprocess_for_ml(df)
    # Random Forest does NOT need scaling
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1  # use all CPU cores
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    print('=== Random Forest Results ===')
    print(classification_report(y_test, y_pred,
          target_names=['No Churn','Churn']))
    print(f'AUC-ROC: {auc:.4f}')
    # Feature Importance
    feat_imp = pd.Series(model.feature_importances_, index=X_train.columns)
    feat_imp = feat_imp.nlargest(20).sort_values()
    fig, ax = plt.subplots(figsize=(10, 8))
    feat_imp.plot.barh(ax=ax, color='#3949AB')
    ax.set_title('Top 20 Feature Importances — Random Forest', fontweight='bold')
    ax.set_xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('reports/figures/11_rf_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()
    return model, auc
if __name__ == '__main__':
    model, auc = train_random_forest()
joblib.dump(model, 'models/random_forest.pkl')
print(f'Random Forest AUC: {auc:.4f}')