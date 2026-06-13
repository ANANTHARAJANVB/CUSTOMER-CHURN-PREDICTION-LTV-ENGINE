import pandas as pd
from sqlalchemy import create_engine

# 1. Connect to your database (Remember to put your real password back here!)
engine = create_engine('postgresql://postgres:postgres123@localhost:5432/postgres')

# 2. Point to the PROCESSED data file that actually contains your predictions!
file_path = 'Team_Progress/Rasool/files/data/processed/telco_ltv.csv'
print(f"Reading data directly from: {file_path}")

# 3. Read it and push it to Metabase
df = pd.read_csv(file_path)
df.to_sql('customers_ltv', con=engine, if_exists='replace', index=False)

print(f'SUCCESS! Loaded {len(df)} rows of PROCESSED data into the database!')