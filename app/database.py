import os
from sqlalchemy import create_engine, Column, Integer, Float, String, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

DATABASE_URL = "sqlite:///./courier_retention.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# Table to log every agent run
class CourierLog(Base):
    __tablename__ = "courier_logs"

    id = Column(Integer, primary_key=True, index=True)
    courier_id = Column(Integer)
    risk_score = Column(Float)
    risk_level = Column(String)
    intervention_type = Column(String)
    message_sent = Column(Boolean)
    timestamp = Column(DateTime, default=datetime.utcnow)

# Create tables
Base.metadata.create_all(bind=engine)

def log_courier_result(result: dict):
    db = SessionLocal()
    log = CourierLog(
        courier_id=result["courier_id"],
        risk_score=result["risk_score"],
        risk_level=result["risk_level"],
        intervention_type=result["intervention"],
        message_sent=result["message_sent"]
    )
    db.add(log)
    db.commit()
    db.close()

def get_all_logs():
    db = SessionLocal()
    logs = db.query(CourierLog).all()
    db.close()
    return logs