from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from preprocess import preprocess_input

# Initialize FastAPI app
app = FastAPI(title="Forecast Automation API")

# Load model
with open("./model/model.pkl", "rb") as f:
    model = pickle.load(f)

# Define input format
class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    # Add as many features as needed

# Define health check
@app.get("/")
def read_root():
    return {"message": "Forecast API is running"}

# Define prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    input_array = np.array([[data.feature1, data.feature2, data.feature3]])
    
    # Preprocess data
    processed_input = preprocess_input(input_array)
    
    # Predict
    prediction = model.predict(processed_input)
    
    return {"forecast_result": prediction.tolist()}
