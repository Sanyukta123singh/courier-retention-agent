import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Sample courier data to train the model
data = {
    'trips_last_week': [45, 5, 30, 2, 40, 8, 35, 1, 50, 3],
    'login_days_last_week': [6, 1, 5, 1, 6, 2, 5, 1, 7, 1],
    'avg_earnings_last_week': [3200, 400, 2100, 200, 2900, 600, 2400, 150, 3500, 250],
    'support_tickets_raised': [0, 3, 1, 5, 0, 4, 1, 6, 0, 5],
    'churned': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

FEATURES = ['trips_last_week', 'login_days_last_week', 
            'avg_earnings_last_week', 'support_tickets_raised']

X = df[FEATURES]
y = df['churned']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled, y)

def predict_churn_risk(trips, login_days, avg_earnings, support_tickets):
    input_data = pd.DataFrame([{
        'trips_last_week': trips,
        'login_days_last_week': login_days,
        'avg_earnings_last_week': avg_earnings,
        'support_tickets_raised': support_tickets
    }])
    
    input_scaled = scaler.transform(input_data)
    risk_score = model.predict_proba(input_scaled)[0][1]
    
    if risk_score > 0.6:
        risk_level = "HIGH"
    elif risk_score > 0.3:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"
    
    return round(risk_score, 2), risk_level