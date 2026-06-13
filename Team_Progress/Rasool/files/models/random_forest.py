import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import joblib
import sys
import os

sys.path.insert(0, '.')
from features.preprocessing import preprocess_for_ml

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
    print(classification_report(y_test, y_pred, target_names=['No Churn','Churn']))
    print(f'AUC-ROC: {auc:.4f}')
    
    # Feature Importance
    feat_imp = pd.Series(model.feature_importances_, index=X_train.columns)
    feat_imp = feat_imp.nlargest(20).sort_values()
    fig, ax = plt.subplots(figsize=(10, 8))
    feat_imp.plot.barh(ax=ax, color='#3949AB')
    ax.set_title('Top 20 Feature Importances — Random Forest', fontweight='bold')
    ax.set_xlabel('Importance Score')
    plt.tight_layout()
    
    # Ensure the reports folder exists before saving the image
    os.makedirs('reports/figures', exist_ok=True)
    plt.savefig('reports/figures/11_rf_feature_importance.png', dpi=150, bbox_inches='tight')
    
    # NOTE: plt.show() will pause the terminal until you close the image window!
    plt.show() 
    
    # We now return the columns so they can be saved at the bottom
    return model, auc, list(X_train.columns)

if __name__ == '__main__':
    # Unpack all three variables returned by the function
    model, auc, feature_cols = train_random_forest()
    
    # Ensure the models folder exists
    os.makedirs('models', exist_ok=True)
    
    print("\nSaving model and features for LTV calculation...")
    
    # Save using the exact filenames required by ltv_model.py
    joblib.dump(model, 'models/production_model.pkl')
    joblib.dump(feature_cols, 'models/feature_columns.pkl')
    
    print(f"✅ Random Forest AUC: {auc:.4f}")
    print("✅ Model and feature columns saved successfully!")