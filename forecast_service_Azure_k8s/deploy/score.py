import joblib
import json
import os
from preprocess import preprocess_input

def init():
    global model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    model = joblib.load(model_path)

def run(raw_data):
    try:
        data = json.loads(raw_data)
        processed_data = preprocess_input(data['data'])
        predictions = model.predict(processed_data)
        return predictions.tolist()
    except Exception as e:
        return str(e)
