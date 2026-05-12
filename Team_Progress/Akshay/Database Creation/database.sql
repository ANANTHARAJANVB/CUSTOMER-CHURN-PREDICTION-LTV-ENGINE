CREATE TABLE customers (
    customer_id VARCHAR(50),
    gender VARCHAR(10),
    senior_citizen INTEGER,
    partner VARCHAR(10),
    dependents VARCHAR(10),
    tenure INTEGER,
    phone_service VARCHAR(10),
    multiple_lines VARCHAR(30),
    internet_service VARCHAR(30),
    online_security VARCHAR(30),
    online_backup VARCHAR(30),
    device_protection VARCHAR(30),
    tech_support VARCHAR(30),
    streaming_tv VARCHAR(30),
    streaming_movies VARCHAR(30),
    contract VARCHAR(30),
    paperless_billing VARCHAR(10),
    payment_method VARCHAR(50),
    monthly_charges NUMERIC(10,2),
    total_charges NUMERIC(10,2),
    churn VARCHAR(10)
);
SELECT * FROM customers;

SELECT * FROM customers
LIMIT 10;

SELECT COUNT(*) FROM customers;

SELECT churn, COUNT(*) 
FROM customers
GROUP BY churn;

SELECT COUNT(*) 
FROM customers
WHERE total_charges IS NULL;