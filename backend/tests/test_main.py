from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from backend.app.main import app

client = TestClient(app)

def test_root_endpoint():
    """Health check works."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "SupportIQ API is running!", "status": "healthy"}

@patch('backend.app.main.groq_client')
@patch('backend.app.main.classifier')
@patch('backend.app.main.label_encoder')
def test_predict_endpoint(mock_label_encoder, mock_classifier, mock_groq):
    """Predict endpoint works with the ML pipeline + LLM fully mocked (fast, no network, no model files)."""

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
        "order_id": "ORD-123"
    }
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["category"] == "Delivery"
    assert data["priority"] == "High"                 
    assert "escalated your delivery" in data["suggested_response"]  
    assert data["model_version"] == "distilbert-v1.0"