"""
API Tests — 5 automated tests.
Run from: Team_Progress\Rasool\files\
Command:  pytest tests/test_api.py -v
IMPORTANT: imports from api.main — NOT src.api.main (no src/ in this project)
"""
import sys
import pytest
sys.path.insert(0, '.')

from fastapi.testclient import TestClient
# YOUR project: api.main (not src.api.main)
from api.main import app

client = TestClient(app)

VALID_CUSTOMER = {
    'gender': 'Male', 'SeniorCitizen': 0, 'Partner': 'No',
    'Dependents': 'No', 'tenure': 24, 'PhoneService': 'Yes',
    'MultipleLines': 'No', 'InternetService': 'Fiber optic',
    'OnlineSecurity': 'No', 'OnlineBackup': 'No',
    'DeviceProtection': 'No', 'TechSupport': 'No',
    'StreamingTV': 'No', 'StreamingMovies': 'No',
    'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 70.35, 'TotalCharges': 1687.65
}


def test_health_endpoint():
    """GET /health must return 200 with status='healthy'."""
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'healthy'
    print("\n✅ Test 1: health endpoint works")


def test_predict_valid_customer():
    """POST /predict must return churn_probability in [0, 1]."""
    response = client.post('/predict', json=VALID_CUSTOMER)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data['churn_probability'] <= 1.0
    assert data['churn_prediction'] in ['Will Churn', 'Will Stay']
    assert data['risk_level'] in ['High', 'Medium', 'Low']
    print(f"\n✅ Test 2: prediction works — prob={data['churn_probability']:.3f}")


def test_predict_missing_field():
    """Missing 'tenure' must return 422 validation error."""
    bad = {k: v for k, v in VALID_CUSTOMER.items() if k != 'tenure'}
    response = client.post('/predict', json=bad)
    assert response.status_code == 422
    print("\n✅ Test 3: missing field returns 422")


def test_predict_invalid_tenure():
    """Negative tenure must return 422."""
    bad = {**VALID_CUSTOMER, 'tenure': -5}
    response = client.post('/predict', json=bad)
    assert response.status_code == 422
    print("\n✅ Test 4: invalid tenure returns 422")


def test_batch_prediction():
    """POST /predict/batch with 2 customers must return 2 predictions."""
    batch = {'customers': [VALID_CUSTOMER, VALID_CUSTOMER]}
    response = client.post('/predict/batch', json=batch)
    assert response.status_code == 200
    data = response.json()
    assert data['total_customers'] == 2
    assert len(data['predictions']) == 2
    print(f"\n✅ Test 5: batch works — {data['total_customers']} predictions")