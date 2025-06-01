from fastapi import FastAPI, HTTPException
import numpy as np
import joblib
import logging
from prometheus_fastapi_instrumentator import Instrumentator
from fastapi.responses import JSONResponse
from prometheus_client import Summary, Counter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
from typing import List, Optional

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus custom metrics
inference_duration = Summary('inference_duration_seconds', 'Time spent on inference')
feature_length_errors = Counter('input_feature_length_errors', 'Requests with incorrect feature length')


def preprocess_landmarks(points, scaler):
    
    if points.shape != (21, 3):  
        logging.debug(f"Error: Expected 21 landmarks, but got shape {points.shape}. Skipping frame.")
        return None  
    wrist = points[0, :2]
    mid_finger_tip = points[12, :2]
    points[:, :2] -= wrist 
    points[:, :2] /= mid_finger_tip  
    
    points = points.flatten().reshape(1, -1) 

    points = scaler.transform(points) 

    return points.flatten() 



# Load the model
path = 'model/RandomForest_model.pkl'
scaler_path = "scaler/scaler.pkl"
try:
    model = joblib.load(path)   
    scaler = joblib.load(scaler_path) 
    logging.info(f"Model loaded from {path}")
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    raise RuntimeError("Model loading failed")

# Initialize FastAPI
app = FastAPI(title="Hand Gesture Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)
logging.info("FastAPI app initialized")


instrumentator = Instrumentator().instrument(app).expose(app)


class InputData(BaseModel):
    features: list[float]

class PredictionResponse(BaseModel):
    predicted_class: int


@app.get("/")
def root():
    logging.info("Root endpoint accessed")
    return {"message": "Welcome to the Hand Gesture Recognition API!"}


class Landmark(BaseModel):
    x: float
    y: float
    z: float
    visibility: Optional[float] = 0.0

class InputData(BaseModel):
    features: List[Landmark]

    @staticmethod
    def validate_length(features):
        if len(features) != 21:
            raise ValueError(f"Expected 21 features, got {len(features)}")

    @classmethod
    def validate(cls, value):
        cls.validate_length(value.features)
        return value

class PredictionResponse(BaseModel):
    predicted_class: int

@app.post("/predict", response_model=PredictionResponse)
def predict(data: InputData):

    try:
        features_np = np.array([[lm.x, lm.y, lm.z] for lm in data.features])
        processed = preprocess_landmarks(features_np, scaler).reshape(1, -1)
        prediction = model.predict(processed)[0]
        logging.info(f"Prediction made: {prediction}")
        if prediction == 'like':
           return {"predicted_class": int(0)}
        elif prediction == 'dislike':
           return {"predicted_class": int(1)}
        elif prediction == 'peace':
           return {"predicted_class": int(2)}
        elif prediction == 'peace_inverted':
              return {"predicted_class": int(3)}
       
        return {"predicted_class": int(4)}

    except ValidationError as ve:
        logging.warning(f"Validation error: {ve}")
        return JSONResponse(status_code=400, content={"error": str(ve)})

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")
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