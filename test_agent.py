from app.agent import run_agent

# Test couriers with different risk levels
couriers = [
    {
        "courier_id": 1,
        "trips_last_week": 2,
        "login_days_last_week": 1,
        "avg_earnings_last_week": 200,
        "support_tickets_raised": 5,
        "risk_score": 0.0,
        "risk_level": "",
        "intervention_type": "",
        "outreach_message": ""
    },
    {
        "courier_id": 2,
        "trips_last_week": 30,
        "login_days_last_week": 4,
        "avg_earnings_last_week": 1800,
        "support_tickets_raised": 2,
        "risk_score": 0.0,
        "risk_level": "",
        "intervention_type": "",
        "outreach_message": ""
    },
    {
        "courier_id": 3,
        "trips_last_week": 50,
        "login_days_last_week": 7,
        "avg_earnings_last_week": 3500,
        "support_tickets_raised": 0,
        "risk_score": 0.0,
        "risk_level": "",
        "intervention_type": "",
        "outreach_message": ""
    }
]

print("=== Courier Retention Agent Running ===\n")

for courier in couriers:
    print(f"Processing Courier {courier['courier_id']}...")
    result = run_agent(courier)
    print(f"Risk: {result['risk_level']} ({result['risk_score']})")
    print(f"Intervention: {result['intervention']}")
    print(f"Message: {result['message']}")
    print("-" * 50)