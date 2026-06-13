"""
Pydantic schemas — request validation and response formatting for the API.
"""
from pydantic import BaseModel, Field
from typing import Optional, List


class CustomerInput(BaseModel):
    """19 fields required for a churn + LTV prediction."""
    gender:          str   = Field(..., example='Male')
    SeniorCitizen:   int   = Field(..., ge=0, le=1, example=0)
    Partner:         str   = Field(..., example='Yes')
    Dependents:      str   = Field(..., example='No')
    tenure:          int   = Field(..., ge=0, example=24)
    PhoneService:    str   = Field(..., example='Yes')
    MultipleLines:   str   = Field(..., example='No')
    InternetService: str   = Field(..., example='Fiber optic')
    OnlineSecurity:  str   = Field(..., example='No')
    OnlineBackup:    str   = Field(..., example='Yes')
    DeviceProtection:str   = Field(..., example='No')
    TechSupport:     str   = Field(..., example='No')
    StreamingTV:     str   = Field(..., example='Yes')
    StreamingMovies: str   = Field(..., example='Yes')
    Contract:        str   = Field(..., example='Month-to-month')
    PaperlessBilling:str   = Field(..., example='Yes')
    PaymentMethod:   str   = Field(..., example='Electronic check')
    MonthlyCharges:  float = Field(..., gt=0,  example=70.35)
    TotalCharges:    float = Field(..., ge=0,  example=1687.65)


class ChurnPredictionResponse(BaseModel):
    """What the API returns for each prediction."""
    customer_id:       Optional[str]
    churn_probability: float
    churn_prediction:  str    # 'Will Churn' or 'Will Stay'
    risk_level:        str    # 'High', 'Medium', 'Low'
    predictive_ltv:    float
    ltv_segment:       str    # 'High Value', 'Medium Value', 'Low Value'
    top_risk_factors:  list
    recommendation:    str


class BatchCustomer(BaseModel):
    """Input for the batch prediction endpoint."""
    customers: List[CustomerInput]