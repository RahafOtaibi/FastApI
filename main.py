from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load the fitted scaler and model
scaler = joblib.load('scaler.joblib')
model = joblib.load('knn_model.joblib')

class InputFeatures(BaseModel):
    minutes_played: int
    appearance: int
    highest_value: int

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI prediction service!"}

def preprocessing(input_features: InputFeatures):
    # Prepare a dictionary with specific column names
    dict_f = {
        'minutes played': input_features.minutes_played,
        'appearance': input_features.appearance,
        'highest_value': input_features.highest_value
    }
    return dict_f

@app.post("/predict")
def predict(data: InputFeatures):
    try:
        # Call preprocessing to get a formatted dictionary
        processed_data = preprocessing(data)
        
        # Prepare input data for prediction
        input_data = np.array([[processed_data['minutes played'], processed_data['appearance'], processed_data['highest_value']]])
        input_scaled = scaler.transform(input_data)  # Use the fitted scaler
        
        # Make prediction
        prediction = model.predict(input_scaled)
        
        # Convert numpy types to native Python types
        return {"prediction": int(prediction[0])}  # Ensuring the prediction is an int
    except Exception as e:
        return {"error": str(e)}
