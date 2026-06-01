import pandas as pd
import numpy as np
import os, sys
sys.path.insert(0, '.')
from src.db.connection import get_engine
from sqlalchemy import text
def load_raw_data():
    engine = get_engine()
    df = pd.read_sql('SELECT * FROM customers', con=engine)
    print(f'Loaded {len(df):,} rows, {len(df.columns)} columns')
    return df
def audit_data_quality(df):
    """Print a full quality report for the dataset."""
    print('\n' + '='*60)
    print('DATA QUALITY AUDIT')
    print('='*60)
    # 1. Shape
    print(f'Shape: {df.shape}')
    # 2. True NaN values
    nan_counts = df.isnull().sum()
    print(f'\nColumns with NaN values:')
    print(nan_counts[nan_counts > 0] if nan_counts.sum() > 0 else '  None')
    # 3. Empty strings (often hidden 'missing' values)
    print(f'\nColumns with empty strings:')
    for col in df.select_dtypes('object').columns:
        empty = (df[col].str.strip() == '').sum()
        if empty > 0:
            print(f'  {col}: {empty} empty strings')
    # 4. Duplicate rows
    dups = df.duplicated().sum()
    print(f'\nDuplicate rows: {dups}')
    # 5. Unique values for categorical columns
    print(f'\nCategorical column unique values:')
    for col in df.select_dtypes('object').columns:
        uniq = df[col].unique()
        print(f'  {col}: {list(uniq)[:8]}')
    print('='*60)
def clean_data(df):
    """Apply all data cleaning transformations."""
    df = df.copy()
    #  Fix 1: TotalCharges — convert to numeric 
    # 11 rows have empty strings in TotalCharges
    # These are new customers (tenure=0) with no total yet
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # For the 11 rows with NaN TotalCharges:
    # Their tenure is 0, so TotalCharges logically = MonthlyCharges
    mask = df['TotalCharges'].isnull()
    df.loc[mask, 'TotalCharges'] = df.loc[mask, 'MonthlyCharges']
    print(f'Fixed TotalCharges: {mask.sum()} rows imputed with MonthlyCharges')
    #  Fix 2: Remove duplicate rows 
    before = len(df)
    df = df.drop_duplicates(subset=['customerID'])
    print(f'Removed {before - len(df)} duplicate rows')
    #  Fix 3: Standardise column names to lowercase 
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    print('Standardised column names to lowercase')
    #  Fix 4: Convert binary Yes/No to 1/0 for ML 
    binary_cols = ['partner', 'dependents', 'phoneservice',
                   'paperlessbilling', 'churn']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})
    df['seniorcitizen'] = df['seniorcitizen'].astype(int)
    print(f'Converted {len(binary_cols)} Yes/No columns to 1/0')
    #  Validation 
    assert df['totalcharges'].isnull().sum() == 0, 'TotalCharges still has NaN!'
    assert df.duplicated(subset=['customerid']).sum() == 0, 'Duplicates remain!'
    print(f'\nFinal dataset: {df.shape}')
    print(f'Missing values remaining: {df.isnull().sum().sum()}')
    return df
def save_cleaned_data(df):
    # Save to CSV
    os.makedirs('data/processed', exist_ok=True)
    path = 'data/processed/telco_churn_clean.csv'
    df.to_csv(path, index=False)
    print(f'Saved to {path}')
    # Save to PostgreSQL
    engine = get_engine()
    df.to_sql('customers_clean', con=engine, if_exists='replace', index=False)
    print(f'Saved to PostgreSQL table: customers_clean')
if __name__ == '__main__':
    df_raw = load_raw_data()
audit_data_quality(df_raw)
df_clean = clean_data(df_raw)
save_cleaned_data(df_clean)
print('\nData cleaning complete!')