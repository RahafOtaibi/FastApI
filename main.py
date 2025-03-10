from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

model = joblib.load('knn_model.joblib')
scaler = joblib.load('scaler.joblib')


class InputFeatures(BaseModel):
    minutes_played: int
    appearance: int
    highest_value: int


def preprocessing(input_features: InputFeatures):
    dict_f = {
        'minutes_played': input_features.minutes_played,
        'appearance': input_features.appearance,
        'highest_value': input_features.highest_value
    }
    return dict_f


@app.post("/predict")
def predict(data: InputFeatures):
    input_data = preprocessing(data)
    input_array = np.array([list(input_data.values())]).reshape(1, -1)

    # Transform input data using the scaler
    input_scaled = scaler.transform(input_array)

    # Make a prediction
    prediction = model.predict(input_scaled)

    # Return the prediction as a string instead of converting to int
    return {"prediction": prediction[0]}  # No `int()` conversion needed
