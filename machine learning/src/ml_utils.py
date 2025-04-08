import json
import os
import time
from datetime import datetime, timedelta
import pandas as pd
import yaml

HYPERPARAMS_FILE = "best_hyperparameters.json"
MAX_AGE_DAYS = 7

def save_hyperparameters(params: dict, filepath: str = HYPERPARAMS_FILE):
    """Save hyperparameters to a JSON file."""
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(params, f, indent=4)
    print(f"Hyperparameters saved to {filepath}.")

def load_hyperparameters(filepath: str = HYPERPARAMS_FILE):
    """Load hyperparameters if the file is not older than MAX_AGE_DAYS."""
    if not os.path.exists(filepath):
        print("No hyperparameters file found.")
        return None

    # Check file modification time
    file_mod_time = datetime.fromtimestamp(os.path.getmtime(filepath))
    if datetime.now() - file_mod_time > timedelta(days=MAX_AGE_DAYS):
        print("Hyperparameters file is older than a week. Ignoring it.")
        return None

    with open(filepath, 'r') as f:
        params = json.load(f)
    print(f"Hyperparameters loaded from {filepath}.")
    return params

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def load_data_pd(data_path, data_format, features, target):
    if data_format.lower() == "csv":
        df = pd.read_csv(data_path)
    elif data_format.lower() == "parquet":
        df = pd.read_parquet(data_path)
    else:
        raise ValueError(f"Unsupported data format: {data_format}")
    
    X = df[features]
    y = df[target]
    print("x ::", X.shape)
    print("y ::", y.shape)
    return X, y
