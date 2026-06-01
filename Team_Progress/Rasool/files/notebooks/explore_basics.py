import pandas as pd
import sys
sys.path.insert(0, '.')  # makes src/ importable
from src.db.connection import get_engine

engine = get_engine()

# Load ALL data from PostgreSQL into Python
df = pd.read_sql('SELECT * FROM customers', con=engine)

print('=== DATASET OVERVIEW ===')
print(f'Shape: {df.shape}')     # (rows, columns)
print(f'Rows: {df.shape[0]:,}')
print(f'Cols: {df.shape[1]}')

print('\n=== COLUMN NAMES AND TYPES ===')
print(df.dtypes)

print('\n=== FIRST 5 ROWS ===')
print(df.head())

print('\n=== NUMERIC STATISTICS ===')
print(df.describe())

print('\n=== CHURN DISTRIBUTION ===')
counts = df['Churn'].value_counts()
pct    = df['Churn'].value_counts(normalize=True) * 100
print(f'Not Churned: {counts["No"]:,} ({pct["No"]:.1f}%)')
print(f'Churned:     {counts["Yes"]:,} ({pct["Yes"]:.1f}%)')

print('\n=== CONTRACT TYPES ===')
print(df['Contract'].value_counts())

print('\n=== CHURN RATE BY CONTRACT ===')
churn_rate = df.groupby('Contract')['Churn'].apply(
    lambda x: (x == 'Yes').mean() * 100
).round(1)
print(churn_rate)

print('\n=== MISSING VALUES ===')
missing = df.isnull().sum()
missing = missing[missing > 0]
if len(missing) == 0:
    print('No missing values!')
else:
    print(missing)