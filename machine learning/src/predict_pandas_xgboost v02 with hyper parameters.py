# predict.py
import os
import mlflow
import xgboost as xgb
import dask.dataframe as dd
import yaml

# Load config
with open("config/train_config.yaml", "r") as f:
    config = yaml.safe_load(f)

model_uri = config["model"]["uri"]
data_path = config["data"]["inference_path"]
output_path = config["data"]["output_path"]

# Load model
model = mlflow.xgboost.load_model(model_uri)

# Load inference data
X_infer = dd.read_csv(data_path)

# Predict
preds = model.predict(X_infer.compute())

# Save predictions
output_df = X_infer.copy()
output_df["predicted_sales"] = preds
output_df.to_csv(output_path, index=False)

print(f"Inference complete! Predictions saved to {output_path}")
