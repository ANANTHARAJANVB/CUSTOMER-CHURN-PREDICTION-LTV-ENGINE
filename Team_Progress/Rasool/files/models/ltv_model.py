r"""
LTV Model — Customer Lifetime Value Calculation.
Calculates historical + predictive LTV and segments customers by value tier.
Run from: Team_Progress\Rasool\files\
Command:  python models/ltv_model.py
"""
import pandas as pd
import numpy as np
import joblib
import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '.')   # Adds 'files/' to Python path so db/ and features/ are found

ANNUAL_DISCOUNT_RATE = 0.10   # 10% discount rate for time value of money


def calculate_historical_ltv(df: pd.DataFrame) -> pd.DataFrame:
    """Historical LTV = total revenue received from this customer so far."""
    df = df.copy()
    df['historical_ltv'] = df['totalcharges']
    return df


def calculate_predictive_ltv(df, churn_probs):
    """
    Predictive LTV = MonthlyCharges * expected remaining months.
    Expected remaining months = 1 / churn_probability.
    Applies Present Value discount for time value of money.
    """
    df = df.copy()
    df['churn_probability'] = churn_probs
    df['churn_probability'] = df['churn_probability'].clip(lower=0.01)
    df['expected_remaining_months'] = (1.0 / df['churn_probability']).clip(upper=72)

    r = ANNUAL_DISCOUNT_RATE / 12
    n = df['expected_remaining_months']
    df['predictive_ltv'] = df['monthlycharges'] * (1 - (1 + r) ** (-n)) / r
    df['total_ltv'] = df['historical_ltv'] + df['predictive_ltv']
    return df


def segment_by_ltv(df):
    """Assign High / Medium / Low Value tier based on total_ltv percentiles."""
    p33 = df['total_ltv'].quantile(0.33)
    p67 = df['total_ltv'].quantile(0.67)
    def tier(v):
        if v >= p67:   return 'High Value'
        elif v >= p33: return 'Medium Value'
        else:          return 'Low Value'
    df['ltv_segment'] = df['total_ltv'].apply(tier)
    return df, p33, p67


if __name__ == '__main__':
    print("=" * 55)
    print("=== STEP 1: Loading clean data ===")
    try:
        df = pd.read_csv('data/processed/telco_features.csv')
        print(f"Loaded {len(df)} customers")
    except FileNotFoundError:
        print("ERROR: Could not find 'data/processed/telco_features.csv'.")
        print("Ensure you are running this script from the correct root directory.")
        sys.exit(1)

    print("\n=== STEP 2: Loading production model ===")
    try:
        model        = joblib.load('models/production_model.pkl')
        feature_cols = joblib.load('models/feature_columns.pkl')
        print("Model loaded OK")
    except FileNotFoundError:
        print("ERROR: Could not find 'models/production_model.pkl' or 'feature_columns.pkl'.")
        print("Please ensure you have run your model training script first to generate these files.")
        sys.exit(1)

    print("\n=== STEP 3: Preparing data for scoring ===")
    df_temp = df.copy()
    drop_cols = ['customerid', 'churn', 'loyalty_segment']
    df_temp = df_temp.drop(columns=[c for c in drop_cols if c in df_temp.columns])
    cat_cols = ['internetservice', 'contract', 'paymentmethod', 'multiplelines']
    df_temp  = pd.get_dummies(df_temp,
                               columns=[c for c in cat_cols if c in df_temp.columns],
                               drop_first=False)
    for col in feature_cols:
        if col not in df_temp.columns:
            df_temp[col] = 0
    df_temp = df_temp[feature_cols]

    print("\n=== STEP 4: Getting churn probabilities ===")
    churn_probs = model.predict_proba(df_temp)[:, 1]
    print(f"Mean churn prob: {churn_probs.mean():.3f}")

    print("\n=== STEP 5: Calculating LTV ===")
    df_ltv = calculate_historical_ltv(df)
    df_ltv = calculate_predictive_ltv(df_ltv, churn_probs)
    df_ltv, p33, p67 = segment_by_ltv(df_ltv)

    print(df_ltv[['total_ltv', 'churn_probability']].describe().round(2))
    print("\nSegment Distribution:")
    print(df_ltv['ltv_segment'].value_counts())
    print(f"\nThresholds: p33=${p33:.2f} | p67=${p67:.2f}")

    print("\n=== STEP 6: Saving outputs ===")
    # Ensure the directory exists before saving
    os.makedirs('data/processed', exist_ok=True)
    df_ltv.to_csv('data/processed/telco_ltv.csv', index=False)
    print("Saved: data/processed/telco_ltv.csv")

    # Save to PostgreSQL
    # NOTE: uses db.connection (NOT src.db.connection — no src/ in your project)
    try:
        from db.connection import get_engine
        engine = get_engine()
        df_ltv.to_sql('customers_ltv', con=engine, if_exists='replace', index=False)
        print("Saved: customers_ltv table in PostgreSQL")
    except Exception as e:
        print(f"DB save skipped: {e}")
        print("The CSV is saved — that is enough to continue.")

    print("\n✅ Day 15 complete!")