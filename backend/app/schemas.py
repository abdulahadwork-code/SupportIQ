from pydantic import BaseModel
from typing import Optional

class TicketInput(BaseModel):
    text: str
    order_id: Optional[str] = None

class TicketPrediction(BaseModel):
    category: str
    sentiment: str
    priority: str
    suggested_response: str
    model_version: str