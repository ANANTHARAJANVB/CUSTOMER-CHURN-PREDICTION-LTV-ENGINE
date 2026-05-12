import psycopg2

try:
    connection = psycopg2.connect(
        host="localhost",
        database="churn_prediction",
        user="postgres",
        password="1234",
        port="5432"
    )

    cursor = connection.cursor()

    cursor.execute("SELECT COUNT(*) FROM customers;")
    record_count = cursor.fetchone()

    print("Database connected successfully!")
    print("Total records in customers table:", record_count[0])

    cursor.close()
    connection.close()

except Exception as error:
    print("Error while connecting to PostgreSQL:", error)