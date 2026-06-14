# Customer Churn Prediction & LTV Engine

[![Python](https://img.shields.io/badge/Python-3.14-blue)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.108-green)]()
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-blue)]()
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)]()

## Overview
This project predicts customer churn and estimates Customer Lifetime Value (LTV) using customer demographics, billing, and service usage data. It helps businesses improve retention strategies and focus on high-value customers. 

**Team:** Anantharajan V B, Bhavana P, Akshay M, Mohammed Rasool K M

## Project Structure
All Python code lives in: `Team_Progress/Rasool/files/`

## Key Results
| Metric | Value |
|--------|-------|
| Dataset | 7,043 customers |
| XGBoost AUC-ROC | ~0.86 |
| Month-to-month churn rate | 42.7% |
| LTV Regression R² | ~0.89 |
| API Endpoints | 5 |

## Quick Start
```powershell
# Navigate to working directory
cd Team_Progress\Rasool\files

# Install dependencies
pip install -r requirements.txt

# Start API
uvicorn api.main:app --reload --port 8000