import numpy as np
import joblib
from sklearn.model_selection import train_test_split

scaler_path = "scaler/scaler.pkl"
scaler = joblib.load(scaler_path) 

def preprocess_landmarks(hand_landmarks):
    points = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])

    if points.shape != (21, 3):  
        print(f"Error: Expected 21 landmarks, but got shape {points.shape}. Skipping frame.")
        return None  
    wrist = points[0, :2]
    mid_finger_tip = points[12, :2]
    points[:, :2] -= wrist 
    points[:, :2] /= mid_finger_tip  
    
    points = points.flatten().reshape(1, -1) 

    points = scaler.transform(points) 

    return points.flatten() 