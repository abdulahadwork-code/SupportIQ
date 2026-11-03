from sqlalchemy import Column, Integer, String, Text, DateTime
from datetime import datetime
from .database import Base

class Ticket(Base):
    __tablename__ = "tickets"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text, nullable=False)
    predicted_category = Column(String, index=True)
    sentiment = Column(String)
    priority = Column(String)
    model_version = Column(String, default="distilbert-v1.0")
    created_at = Column(DateTime, default=datetime.utcnow)