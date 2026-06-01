import pandas as pd, sys
sys.path.insert(0, '.')
from src.db.connection import get_engine

engine = get_engine()
df = pd.read_csv('data/raw/telco_churn.csv')
df.to_sql('customers', con=engine, if_exists='replace', index=False)
print(f'Loaded {len(df)} rows into database')