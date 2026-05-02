from dotenv import load_dotenv
load_dotenv()
import requests
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from typing import TypedDict

# Initialize Claude
import os

llm = ChatAnthropic(
    model="claude-opus-4-6",
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

# Your FastAPI URL
API_URL = "http://127.0.0.1:8000/predict"

# Define what flows through the agent
class CourierState(TypedDict):
    courier_id: int
    trips_last_week: int
    login_days_last_week: int
    avg_earnings_last_week: float
    support_tickets_raised: int
    risk_score: float
    risk_level: str
    intervention_type: str
    outreach_message: str

# Node 1 — Call your FastAPI model
def get_risk_score(state: CourierState) -> CourierState:
    response = requests.post(API_URL, json={
        "courier_id": state["courier_id"],
        "trips_last_week": state["trips_last_week"],
        "login_days_last_week": state["login_days_last_week"],
        "avg_earnings_last_week": state["avg_earnings_last_week"],
        "support_tickets_raised": state["support_tickets_raised"]
    })
    
    result = response.json()
    state["risk_score"] = result["risk_score"]
    state["risk_level"] = result["risk_level"]
    
    print(f"Risk score fetched: {result['risk_score']} ({result['risk_level']})")
    return state

# Node 2 — Decide intervention
def decide_intervention(state: CourierState) -> CourierState:
    if state["risk_level"] == "HIGH":
        state["intervention_type"] = "incentive"
    elif state["risk_level"] == "MEDIUM":
        state["intervention_type"] = "nudge"
    else:
        state["intervention_type"] = "none"
    
    print(f"Intervention decided: {state['intervention_type']}")
    return state

# Node 3 — Generate personalized message using Claude
def generate_message(state: CourierState) -> CourierState:
    if state["intervention_type"] == "none":
        state["outreach_message"] = "No outreach needed."
        return state
    
    prompt = f"""
    You are a courier engagement specialist at Uber.
    
    Courier ID: {state['courier_id']}
    Churn Risk: {state['risk_level']} ({state['risk_score']})
    Trips last week: {state['trips_last_week']}
    Earnings last week: {state['avg_earnings_last_week']}
    Intervention: {state['intervention_type']}
    
    Write a short friendly WhatsApp message (2-3 sentences) to re-engage this courier.
    If incentive: offer a bonus for completing trips this week.
    If nudge: remind them of high demand in their area.
    """
    
    response = llm.invoke(prompt)
    state["outreach_message"] = response.content
    
    print(f"Message generated for courier {state['courier_id']}")
    return state

# Build the agent graph
graph = StateGraph(CourierState)

graph.add_node("get_risk_score", get_risk_score)
graph.add_node("decide_intervention", decide_intervention)
graph.add_node("generate_message", generate_message)

graph.set_entry_point("get_risk_score")
graph.add_edge("get_risk_score", "decide_intervention")
graph.add_edge("decide_intervention", "generate_message")
graph.add_edge("generate_message", END)

agent = graph.compile()


def run_agent(courier_data: dict):
    result = agent.invoke(courier_data)
    return {
        "courier_id": result["courier_id"],
        "risk_score": result["risk_score"],
        "risk_level": result["risk_level"],
        "intervention": result["intervention_type"],
        "message": result["outreach_message"]
    }