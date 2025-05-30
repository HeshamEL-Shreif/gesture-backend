from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Hand geasure recognition API!"}

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model"] == "RandomForest_model"
    assert response.json()["prediction_check"] == "passed"

def test_get_model():
    response = client.get("/home")
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "random_forest_hand_gesture_model"
    assert "features" in data
    assert len(data["features"]) == 63  # 21 landmarks * 3 (x, y, z)
    assert data["model_type"] == "RandomForestClassifier"
    assert data["accuracy"] == 0.88
    assert data["f1_score"] == 0.86

def test_valid_prediction():
    test_input = {
        "features": [1] * 63  # Assuming the model expects 63 features
    }
    response = client.post("/predict", json=test_input)
    assert response.status_code == 200
    assert "predicted_class" in response.json()
    assert isinstance(response.json()["predicted_class"], int)


def test_invalid_prediction_too_few_features():
    test_input = {
        "features": [600.0, 1, 0] 
    }
    response = client.post("/predict", json=test_input)
    assert response.status_code == 400
    assert "error" in response.json()