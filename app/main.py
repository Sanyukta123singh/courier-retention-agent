from fastapi import FastAPI
from pydantic import BaseModel
from app.model import predict_churn_risk

app = FastAPI()

# This defines what data the API expects to receive
class CourierData(BaseModel):
    courier_id: int
    trips_last_week: int
    login_days_last_week: int
    avg_earnings_last_week: float
    support_tickets_raised: int

# Health check endpoint
@app.get("/")
def read_root():
    return {"status": "Courier Retention Agent is running"}

# Risk prediction endpoint
@app.post("/predict")
def predict_risk(data: CourierData):
    risk_score, risk_level = predict_churn_risk(
        trips=data.trips_last_week,
        login_days=data.login_days_last_week,
        avg_earnings=data.avg_earnings_last_week,
        support_tickets=data.support_tickets_raised
    )
    
    return {
        "courier_id": data.courier_id,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "message": f"Courier {data.courier_id} has {risk_level} churn risk"
    }