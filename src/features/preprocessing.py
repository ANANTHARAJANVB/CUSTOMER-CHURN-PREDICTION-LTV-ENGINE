import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os, sys
sys.path.insert(0, '.')

def preprocess_for_ml(df: pd.DataFrame):
    df = df.copy()

    # Drop columns not useful for ML
    drop_cols = ['customerid', 'loyalty_segment']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Automatically find ALL text/categorical columns
    # This prevents the 'Male' string to float ValueError
    ohe_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Remove the target variable 'churn' from this list just in case it is still text
    if 'churn' in ohe_cols:
        ohe_cols.remove('churn')
        
    # Apply one-hot encoding (drop_first=True prevents the dummy variable trap for linear models)
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)
    print(f'After one-hot encoding: {df.shape}')

    # Separate features (X) from target (y)
    X = df.drop(columns=['churn'])
    y = df['churn']
    print(f'Features: {X.shape[1]}, Target: {y.name}')
    print(f'Class distribution: {dict(y.value_counts())}')

    # Split 80% train / 20% test (stratified = keep same churn ratio in both)
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
    print(f'Train: {X_train.shape}, Test: {X_test.shape}')
    print(f'Train churn %: {y_train.mean()*100:.1f}%')
    print(f'Test churn %:  {y_test.mean()*100:.1f}%')

    # Scale numeric features (ONLY fit on train data!)
    numeric_cols = X_train.select_dtypes(
        include=['float64','int64']).columns.tolist()
        
    # Ensure boolean columns (from get_dummies) are treated properly if not included in numeric
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled  = X_test.copy()
    
    # Scale only the truly numeric continuous columns
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols]  = scaler.transform(X_test[numeric_cols])

    # Save scaler and column list for use in API later
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(list(X.columns), 'models/feature_columns.pkl')
    print('Saved: models/scaler.pkl')
    print('Saved: models/feature_columns.pkl')

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler

if __name__ == '__main__':
    df = pd.read_csv('data/processed/telco_features.csv')
    result = preprocess_for_ml(df)
    print('Preprocessing complete!')