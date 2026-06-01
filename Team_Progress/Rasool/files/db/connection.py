from sqlalchemy import create_engine
def get_engine():
    return create_engine(
	'postgresql://postgres:postgres123@localhost:5432/churn_db'
	)