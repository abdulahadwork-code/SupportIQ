from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from unittest.mock import patch, MagicMock

from backend.app.main import app
from backend.app import database, models

TEST_ENGINE = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=TEST_ENGINE)

def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[database.get_db] = override_get_db
models.Base.metadata.create_all(bind=TEST_ENGINE)

client = TestClient(app)  


def test_root_endpoint():
    """Health check works."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "SupportIQ API is running!", "status": "healthy"}


@patch('backend.app.main.GROQ_API_KEY', 'gsk_test_dummy_key')   
@patch('backend.app.main.groq_client')                          
@patch('backend.app.main.classifier')                           
@patch('backend.app.main.label_encoder')                        
def test_predict_endpoint(mock_label_encoder, mock_classifier, mock_groq, mock_key):
    """Predict works with ML + LLM fully mocked: fast, no model files, no API key, no network."""
    mock_classifier.return_value = [{"label": "LABEL_1", "score": 0.99}]
    mock_label_encoder.classes_ = ["Billing", "Delivery", "Refund", "Account Access", "Technical Issue"]
    mock_message = MagicMock()
    mock_message.content = "We've escalated your delivery and will send a tracking update shortly."
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_groq.chat.completions.create.return_value = mock_response

    payload = {
        "text": "My order is late and I need a refund immediately!",
        "order_id": "ORD-123",
    }
    response = client.post("/predict", json=payload)

    assert response.status_code == 200          
    data = response.json()
    assert data["category"] == "Delivery"
    assert data["priority"] == "High"                                     
    assert "escalated your delivery" in data["suggested_response"]        
    assert data["model_version"] == "distilbert-v1.0"