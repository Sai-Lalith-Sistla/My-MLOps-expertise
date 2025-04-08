from datetime import datetime
import os
import yaml
import joblib
import mlflow
import mlflow.xgboost
import pandas as pd
import xgboost_ray as xgbr  # ← use Ray's XGBoost interface
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import ray
from ray import train

# Initialize Ray
ray.init(ignore_reinit_error=True)

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_data(data_path, data_format, features, target):
    if data_format.lower() == "csv":
        df = pd.read_csv(data_path)
    elif data_format.lower() == "parquet":
        df = pd.read_parquet(data_path)
    
    X = df[features]
    y = df[target]
    return X, y

def train_model(X_train, y_train, X_valid, y_valid, model_params, early_stopping_rounds):
    # Ray expects DMatrix from local pandas DataFrames
    train_set = xgbr.RayDMatrix(X_train, y_train)
    valid_set = xgbr.RayDMatrix(X_valid, y_valid)

    evals_result = {}

    # params = {
    #     'tree_method': 'hist',
    #     'objective': 'binary:logistic',
    #     'eval_metric': 'logloss',
    #     'verbosity': 1
    # }
    # params.update(model_params)  # override if you have extra params

    output = xgbr.train(
        params=params,
        dtrain=train_set,
        evals=[(train_set, "train"), (valid_set, "validation")],
        num_boost_round=model_params.get('n_estimators', 50),
        early_stopping_rounds=early_stopping_rounds,
        evals_result=evals_result,
        # Ray specific options
        ray_params=xgbr.RayParams(
            num_actors=4,  # Number of parallel actors (adjust as per CPUs)
            cpus_per_actor=1
        )
    )

    return output['booster'], evals_result

def evaluate_model(model, X_test, y_test):
    dtest = xgbr.RayDMatrix(X_test, y_test)
    y_pred = xgbr.predict(
        model=model,
        data=dtest
    )
    
    y_true = y_test.values

    mse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {"mse": mse, "mae": mae, "r2": r2}

def save_model(model, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.save_model(output_path)

def main():
    # Load configuration
    config = load_config("configs/train_config.yaml")
    experiment_name = config['experiment_name']
    model_params = config['model_params']
    train_params = config['train_params']
    features = config['features']
    target = config['target']
    data_path = config['data_path']
    model_output_path = config['model_output_path']
    data_format = config['data_format']
    early_stopping_rounds = train_params.get('early_stopping_rounds', 10)

    # Set up MLflow
    mlflow.set_experiment(experiment_name)

    # Create a run name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"experiment_run_{timestamp}"
    
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_params(model_params)

        # Load data
        X, y = load_data(data_path, data_format, features, target)

        # Train-validation-test split
        test_size = train_params.get('test_size', 0.2)
        valid_size = train_params.get('valid_size', 0.1)
        n = len(X)

        train_end = int((1 - test_size - valid_size) * n)
        valid_end = int((1 - test_size) * n)

        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_valid, y_valid = X.iloc[train_end:valid_end], y.iloc[train_end:valid_end]
        X_test, y_test = X.iloc[valid_end:], y.iloc[valid_end:]

        # Train the model
        model, evals_result = train_model(X_train, y_train, X_valid, y_valid, model_params, early_stopping_rounds)

        # Evaluate the model
        metrics = evaluate_model(model, X_test, y_test)

        # Log metrics
        mlflow.log_metrics(metrics)

        # Save model
        save_model(model, model_output_path)

        print("Training and evaluation completed.")
        print("Metrics:", metrics)

if __name__ == "__main__":
    main()
