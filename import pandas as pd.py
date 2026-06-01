import pandas as pd
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
#  Database connection 
DB_HOST     = os.getenv('DB_HOST',     'localhost')
DB_PORT     = os.getenv('DB_PORT',     '5432')
DB_NAME     = os.getenv('DB_NAME',     'churn_db')
DB_USER     = os.getenv('DB_USER',     'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'postgres123')
# Create the connection string (URL format)
# postgresql://user:password@host:port/database
DATABASE_URL = f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
def get_engine():
    """Create and return a SQLAlchemy engine (database connection pool)."""
    engine = create_engine(DATABASE_URL, echo=False)
    return engine
def load_csv_to_postgres():
    """Load the Telco CSV dataset into the PostgreSQL customers table."""
    # Path to the raw CSV file
    csv_path = 'data/raw/telco_churn.csv'
    print(f'Reading CSV from: {csv_path}')
    df = pd.read_csv(csv_path)
    print(f'Dataset shape: {df.shape}')  # rows x columns
    print(f'Columns: {list(df.columns)}')
    print('\nFirst 3 rows:')
    print(df.head(3))
    # Create database engine
    engine = get_engine()
    # Load DataFrame into PostgreSQL
    # if_exists='replace' drops and recreates the table each time
    # This is safe during development — in production use 'append'
    df.to_sql(
        name='customers',          # table name in PostgreSQL
        con=engine,                # connection
        if_exists='replace',       # drop table and recreate
        index=False,               # don't write DataFrame index as a column
        method='multi',            # use multi-row INSERT for speed
        chunksize=1000             # insert 1000 rows at a time
    )
    print(f'\nSuccessfully loaded {len(df)} rows into PostgreSQL customers table.')
    return df
def verify_load():
    """Run a quick SQL query to verify the data loaded correctly."""
    engine = get_engine()
    with engine.connect() as conn:
        # Count total rows
        result = conn.execute(text('SELECT COUNT(*) FROM customers'))
        count = result.fetchone()[0]
        print(f'Total rows in customers table: {count}')
        # Show first 5 rows
        result = conn.execute(text('SELECT * FROM customers LIMIT 5'))
        rows = result.fetchall()
        print('\nFirst 5 rows:')
        for row in rows:
            print(row)
        # Count churned customers
        result = conn.execute(text("SELECT Churn, COUNT(*) FROM customers GROUP BY Churn"))
        churn_counts = result.fetchall()
        print('\nChurn distribution:')
        for row in churn_counts:
            print(f'  Churn={row[0]}: {row[1]} customers')
if __name__ == '__main__':
    print('=== Telco Data Loader ===')
    load_csv_to_postgres()
    verify_load()
print('\n=== Done! ===')