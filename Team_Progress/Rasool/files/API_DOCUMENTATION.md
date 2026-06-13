# Customer Churn & LTV Prediction API Documentation

**Version:** 1.0.0  
**Base URL:** http://localhost:8000  
**Interactive Docs:** http://localhost:8000/docs  
**Project path:** Team_Progress\Rasool\files\api\

---

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | API health check |
| GET | / | API info and endpoint list |
| POST | /predict | Single customer prediction |
| POST | /predict/batch | Batch predictions |
| POST | /predict/csv | CSV file upload |

---

## How to Start the API

```powershell
# Navigate to files folder first
cd Team_Progress\Rasool\files

# Then run (no src/ — this project structure has no src folder)
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

---

## Error Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 422 | Validation error (missing/wrong field) |
| 500 | Server error (model failure) |