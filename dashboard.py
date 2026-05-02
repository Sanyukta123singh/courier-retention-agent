import streamlit as st
import plotly.express as px
import pandas as pd
from app.database import get_all_logs

st.set_page_config(page_title="Courier Retention Agent", layout="wide")

st.title("Courier Retention Agent — Live Dashboard")
st.markdown("Monitoring courier churn risk and outreach performance")

# Fetch data from database
logs = get_all_logs()

if not logs:
    st.warning("No data yet. Run test_agent.py first.")
else:
    # Convert to dataframe
    df = pd.DataFrame([{
        "courier_id": log.courier_id,
        "risk_score": log.risk_score,
        "risk_level": log.risk_level,
        "intervention_type": log.intervention_type,
        "message_sent": log.message_sent,
        "timestamp": log.timestamp
    } for log in logs])

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Couriers Processed", len(df))

    with col2:
        high_risk = len(df[df["risk_level"] == "HIGH"])
        st.metric("High Risk Couriers", high_risk)

    with col3:
        messages_sent = len(df[df["message_sent"] == True])
        st.metric("Messages Sent", messages_sent)

    with col4:
        avg_risk = round(df["risk_score"].mean(), 2)
        st.metric("Avg Risk Score", avg_risk)

    st.divider()

    # Risk distribution chart
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk Level Distribution")
        risk_counts = df["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["risk_level", "count"]
        fig1 = px.pie(
            risk_counts,
            names="risk_level",
            values="count",
            color="risk_level",
            color_discrete_map={
                "HIGH": "#E24B4A",
                "MEDIUM": "#EF9F27",
                "LOW": "#639922"
            }
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Intervention Type Distribution")
        intervention_counts = df["intervention_type"].value_counts().reset_index()
        intervention_counts.columns = ["intervention_type", "count"]
        fig2 = px.bar(
            intervention_counts,
            x="intervention_type",
            y="count",
            color="intervention_type",
            color_discrete_map={
                "incentive": "#E24B4A",
                "nudge": "#EF9F27",
                "none": "#639922"
            }
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # Risk score per courier
    st.subheader("Risk Score per Courier")
    fig3 = px.bar(
        df,
        x="courier_id",
        y="risk_score",
        color="risk_level",
        color_discrete_map={
            "HIGH": "#E24B4A",
            "MEDIUM": "#EF9F27",
            "LOW": "#639922"
        }
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.divider()

    # Raw data table
    st.subheader("All Courier Logs")
    st.dataframe(df, use_container_width=True)