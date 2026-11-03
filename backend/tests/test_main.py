# backend/tests/test_main.py
from fastapi.testclient import TestClient
from unittest.mock import patch
from backend.app.main import app

client = TestClient(app)

def test_root_endpoint():
    """Test that the health check endpoint works."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "SupportIQ API is running!", "status": "healthy"}

@patch('backend.app.main.qwen_client')
@patch('backend.app.main.classifier')
@patch('backend.app.main.label_encoder')
def test_predict_endpoint(mock_label_encoder, mock_classifier, mock_qwen):
    """Test the ML prediction endpoint by mocking the heavy models."""
    
    # 1. Mock DistilBERT
    mock_classifier.return_value = [{"label": "LABEL_1", "score": 0.99}]
    mock_label_encoder.classes_ = ["Billing", "Delivery", "Refund", "Account Access", "Technical Issue"]

    # 2. Mock Qwen LLM response
    mock_message = type('Message', (), {'content': 'We are so sorry your order is late! We have expedited your shipment and it will arrive tomorrow.'})()
    mock_choice = type('Choice', (), {'message': mock_message})()
    mock_qwen.chat_completion.return_value = type('Response', (), {'choices': [mock_choice]})()

    payload = {
        "text": "My order is late and I need a refund immediately!",
        "order_id": "ORD-123"
    }
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["category"] == "Delivery"
    assert data["priority"] == "High"
    # Verify Qwen's mocked response was returned
    assert "expedited your shipment" in data["suggested_response"]