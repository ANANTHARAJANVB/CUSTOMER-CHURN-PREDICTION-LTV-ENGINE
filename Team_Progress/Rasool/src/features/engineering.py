import pandas as pd
import numpy as np
def create_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering transformations.
    Input: cleaned DataFrame (from customers_clean table)
    Output: DataFrame with additional engineered feature columns
    """
    df = df.copy()
    #  1. Ratio Features 
    # Charge per tenure month — how much each loyalty month costs
    # Avoid division by zero for tenure=0 customers
    df['charge_per_tenure'] = np.where(
        df['tenure'] > 0,
        df['monthlycharges'] / df['tenure'],
        df['monthlycharges']
    )
    # Total charges / monthly charges = effective tenure paid
    df['effective_tenure_paid'] = np.where(
        df['monthlycharges'] > 0,
        df['totalcharges'] / df['monthlycharges'],
        0
    )
    #  2. Interaction Features 
    # Revenue potential = tenure * monthly charges
    df['revenue_potential'] = df['tenure'] * df['monthlycharges']
    #  3. Service Count Features 
    # Map No internet service and No phone service to 0
    service_cols = ['onlinesecurity','onlinebackup','deviceprotection',
                    'techsupport','streamingtv','streamingmovies']
    for col in service_cols:
        # 'No internet service' → 0, 'Yes' → 1, 'No' → 0
        df[col + '_flag'] = (df[col] == 'Yes').astype(int)
    df['service_count'] = df[[c+'_flag' for c in service_cols]].sum(axis=1)
    df['has_premium_services'] = (df['service_count'] >= 3).astype(int)
    # Phone + Internet bundle flag
    df['has_phone'] = (df['phoneservice'] == 1).astype(int)
    df['has_internet'] = (df['internetservice'] != 'No').astype(int)
    df['has_bundle'] = (df['has_phone'] & df['has_internet']).astype(int)
    #  4. Loyalty Segment 
    df['loyalty_segment'] = pd.cut(
        df['tenure'],
        bins=[0, 12, 24, 48, 72, float('inf')],
        labels=['New (0-12m)', 'Developing (1-2yr)', 'Established (2-4yr)',
                'Loyal (4-6yr)', 'Champion (6yr+)'],
        include_lowest=True
    )
    #  5. Contract Risk Score 
    contract_risk = {
        'Month-to-month': 3,   # high risk
        'One year': 2,          # medium risk
        'Two year': 1           # low risk
    }
    df['contract_risk'] = df['contract'].map(contract_risk)
    #  6. Payment Risk Flag 
    # Electronic check customers churn most
    df['is_electronic_check'] = (
        df['paymentmethod'] == 'Electronic check').astype(int)
    #  7. Auto-payment Flag (reduces churn) 
    auto_pay = ['Bank transfer (automatic)', 'Credit card (automatic)']
    df['is_auto_payment'] = df['paymentmethod'].isin(auto_pay).astype(int)
    print(f'Feature engineering complete. New shape: {df.shape}')
    print(f'New features added: {df.shape[1] - 21}')
    return df
if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from tests.Files.src.db.connection import get_engine
    engine = get_engine()
    df = pd.read_sql('SELECT * FROM customers_clean', con=engine)
    df_eng = create_all_features(df)
    print(df_eng[['tenure','monthlycharges','charge_per_tenure',
                  'service_count','loyalty_segment','contract_risk']].head(10))
    df_eng.to_csv('data/processed/telco_features.csv', index=False)
    print('Saved: data/processed/telco_features.csv')