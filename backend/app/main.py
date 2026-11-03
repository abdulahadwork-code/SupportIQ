import os
import torch
import joblib
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from contextlib import asynccontextmanager
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from groq import Groq

load_dotenv()

from . import models, schemas, database

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
if not GROQ_API_KEY or not GROQ_API_KEY.startswith("gsk_"):
    print(" APIKEY is invalid or missing in .env")
else:
    print(f"APIKEY loaded ")
groq_client = Groq(api_key=GROQ_API_KEY)
LLM_MODEL = "openai/gpt-oss-20b" 
classifier = None
label_encoder = None
MODEL_VERSION = "distilbert-v1.0"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models and initialize database on startup."""
    global classifier, label_encoder
    # 1. Loading Label Encoder
    encoder_path = "ml/models/label_encoder.pkl"
    if os.path.exists(encoder_path):
        label_encoder = joblib.load(encoder_path)
    else:
        print("label_encoder.pkl not found.")
    # 2. Loading here DistilBERT
    model_path = "ml/models/distilbert_ticket_classifier"
    if os.path.exists(model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        device = 0 if torch.cuda.is_available() else -1
        classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)
        print("DistilBERT model loaded")
    else:
        print("DistilBERT model not found")
    # 3. Creating DB Tables
    models.Base.metadata.create_all(bind=database.engine)
    yield  
app = FastAPI(title="SupportIQ API", version="1.0.0", lifespan=lifespan)

def get_sentiment(text: str) -> str:
    text_lower = text.lower()
    if any(w in text_lower for w in ["not", "never", "hate", "terrible", "issue", "problem", "angry", "furious"]):
        return "Negative"
    elif any(w in text_lower for w in ["great", "excellent", "thank", "happy", "love", "good"]):
        return "Positive"
    return "Neutral"

def get_priority(text: str, category: str) -> str:
    text_lower = text.lower()
    if any(w in text_lower for w in ["urgent", "immediately", "asap", "now"]) or category in ["Refund", "Technical Issue"]:
        return "High"
    return "Medium"

def get_suggested_response(text: str, category: str) -> str:
    if not GROQ_API_KEY:
        return "GROQAPIKEY is missing."

    prompt = f"""You are a professional customer support agent for an e-commerce company.
The customer's issue is classified as: {category}.
The customer wrote: "{text}"
Write a polite, concise, helpful response in maximum 2 sentences. Do not include placeholders."""

    templates = {
        "Billing": "Thank you for reaching out. We are checking your billing details and will resolve this promptly.",
        "Delivery": "We apologize for the delay. We are tracking your shipment and will provide an update shortly.",
        "Refund": "We're sorry to hear about this. Our billing team is reviewing your refund request and will update you within 24 hours.",
        "Account Access": "We understand how frustrating this is. Please reset your password via the link we just emailed you, or we can verify your identity manually.",
        "Technical Issue": "We're sorry you're experiencing this issue. Our technical team has been notified and is investigating.",
    }

    try:
        response = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024,  
        )
        msg = response.choices[0].message
        content = (getattr(msg, "content", None) or "").strip()
        if not content:
            print("⚠️ LLM returned empty content (reasoning-only). Falling back to template.")
            return templates.get(category, "Thank you for contacting support. We are looking into your request.")

        return content

    except Exception as e:
        print(f"⚠️ LLM API failed: {str(e)}")
        return templates.get(category, "Thank you for contacting support. We are looking into your request.")
# --- API Endpoints ---
@app.post("/predict", response_model=schemas.TicketPrediction)
def predict_ticket(ticket: schemas.TicketInput, db: Session = Depends(database.get_db)):
    if classifier is None or label_encoder is None:
        raise HTTPException(status_code=500, detail="ML Models not loaded. Check server logs.")
    
    try:
        # 1. Predict Category using DistilBERT
        prediction = classifier(ticket.text)[0]
        # HF pipeline returns {'label': 'LABEL_0', 'score': 0.99}. We map 'LABEL_X' to the string.
        label_idx = int(prediction['label'].split('_')[-1])
        category = label_encoder.classes_[label_idx]
        
        # 2. Derive Sentiment & Priority
        sentiment = get_sentiment(ticket.text)
        priority = get_priority(ticket.text, category)
        
        # 3. Generate AI Reply using Groq LLM
        suggested_response = get_suggested_response(ticket.text, category)
        
        # 4. Save to Database for analytics
        db_ticket = models.Ticket(
            text=ticket.text,
            predicted_category=category,
            sentiment=sentiment,
            priority=priority,
            model_version=MODEL_VERSION
        )
        db.add(db_ticket)
        db.commit()
        db.refresh(db_ticket)
        
        return schemas.TicketPrediction(
            category=category,
            sentiment=sentiment,
            priority=priority,
            suggested_response=suggested_response,
            model_version=MODEL_VERSION
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.get("/")
def root():
    return {"message": "SupportIQ API is running!", "status": "healthy"}