from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import logging
from prometheus_fastapi_instrumentator import Instrumentator
from fastapi.responses import JSONResponse
from preprocess_landmark import preprocess_landmarks
from prometheus_client import Summary, Counter

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus custom metrics
inference_duration = Summary('inference_duration_seconds', 'Time spent on inference')
feature_length_errors = Counter('input_feature_length_errors', 'Requests with incorrect feature length')

# Load the model
path = 'model/RandomForest_model.pkl'
try:
    model = joblib.load(path)
    logging.info(f"Model loaded from {path}")
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    raise RuntimeError("Model loading failed")

# Initialize FastAPI
app = FastAPI(title="Hand Gesture Recognition API")
logging.info("FastAPI app initialized")

# Expose Prometheus metrics
instrumentator = Instrumentator().instrument(app).expose(app)

# Request and Response Models
class InputData(BaseModel):
    features: list[float]

class PredictionResponse(BaseModel):
    predicted_class: int

# Root Endpoint
@app.get("/")
def root():
    logging.info("Root endpoint accessed")
    return {"message": "Welcome to the Hand Gesture Recognition API!"}

# Prediction Endpoint
@app.post("/predict", response_model=PredictionResponse)
@inference_duration.time()  
def predict(data: InputData):
    expected_features = 63
    if len(data.features) != expected_features:
        logging.warning(f"Expected {expected_features} features, got {len(data.features)}")
        feature_length_errors.inc()  
        return JSONResponse(status_code=400, content={"error": f"Expected {expected_features} features"})

    logging.info(f"Prediction request received with features: {data.features}")
    features = np.array(data.features).reshape(1, -1)
    try:
        features = preprocess_landmarks(features)
        prediction = model.predict(features)
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")
    
    return {"predicted_class": int(prediction[0])}

# Health Check Endpoint
@app.get("/health")
def health_check():
    logging.info("Health check endpoint accessed")

    if model is None:
        logging.error("Model not loaded")
        raise HTTPException(status_code=500, detail="Model not loaded")
    try:
        dummy_input = np.zeros((1, 63))  # 21 landmarks * 3 coords
        _ = model.predict(dummy_input)
    except Exception as e:
        logging.error(f"Prediction check failed: {e}")
        raise HTTPException(status_code=500, detail="Model prediction failed")

    return {
        "status": "healthy",
        "model": "RandomForest_model",
        "prediction_check": "passed"
    }

# Model Metadata Endpoint
@app.get("/home")
def get_model():
    logging.info("Home endpoint accessed")
    return {
        "model": "random_forest_hand_gesture_model",
        "version": "1.0",
        "description": "Random Forest model for hand gesture classification using 3D hand landmark data",
        "data": "hand_landmark_data.csv",
        "features": [f"{coord}{i}" for i in range(1, 22) for coord in ["x", "y", "z"]],
        "target": "gesture_label",
        "model_type": "RandomForestClassifier",
        "accuracy": 0.88,
        "f1_score": 0.86,
        "num_classes": 18
    }