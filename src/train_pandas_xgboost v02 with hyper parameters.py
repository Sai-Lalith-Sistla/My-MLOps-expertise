from datetime import datetime
import os
import yaml
import joblib
import mlflow
import mlflow.xgboost
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import optuna
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_data(data_path, data_format, features, target):
    if data_format.lower() == "csv":
        df = pd.read_csv(data_path)
    elif data_format.lower() == "parquet":
        df = pd.read_parquet(data_path)
    else:
        raise ValueError(f"Unsupported data format: {data_format}")
    
    X = df[features]
    y = df[target]
    return X, y

def train_model(X_train, y_train, model_params, num_boost_round=50):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    booster = xgb.train(model_params, dtrain, num_boost_round=num_boost_round)
    return booster

def evaluate_model(model, X_test, y_test):
    dtest = xgb.DMatrix(X_test)
    y_pred = model.predict(dtest)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {"mse": mse, "mae": mae, "r2": r2}

def save_model(model, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.save_model(output_path)

def objective(trial, X_train, y_train, X_valid, y_valid):
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True)
    }
    booster = train_model(X_train, y_train, params)
    metrics = evaluate_model(booster, X_valid, y_valid)
    return metrics['mse']

def main():
    config = load_config("configs/train_pd_hp_config.yaml")
    experiment_name = config['experiment_name']
    model_params = config['model_params']
    features = config['features']
    target = config['target']
    data_path = config['data_path']
    model_output_path = config['model_output_path']
    data_format = config.get('data_format', 'csv')
    tune_hyperparameters = config.get('tune_hyperparameters', False)

    mlflow.set_experiment(experiment_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"experiment_run_{timestamp}"

    with mlflow.start_run(run_name=run_name):
        X, y = load_data(data_path, data_format, features, target)

        # Simple 80-10-10 split
        n = len(X)
        train_end = int(n * 0.8)
        valid_end = int(n * 0.9)

        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_valid, y_valid = X.iloc[train_end:valid_end], y.iloc[train_end:valid_end]
        X_test, y_test = X.iloc[valid_end:], y.iloc[valid_end:]

        if tune_hyperparameters:
            logger.info("Starting hyperparameter tuning with Optuna...")
            study = optuna.create_study(direction="minimize")
            study.optimize(lambda trial: objective(trial, X_train, y_train, X_valid, y_valid), n_trials=30)

            best_params = study.best_params
            logger.info(f"Best hyperparameters found: {best_params}")
            model_params.update(best_params)

        model = train_model(pd.concat([X_train, X_valid]), pd.concat([y_train, y_valid]), model_params)

        # Validation metrics
        val_metrics = evaluate_model(model, X_valid, y_valid)
        logging.info(f"Validation Metrics: {val_metrics}")
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})

        # Test metrics
        test_metrics = evaluate_model(model, X_test, y_test)
        logging.info(f"Test Metrics: {test_metrics}")
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        mlflow.log_params(model_params)

        save_model(model, model_output_path)
        mlflow.xgboost.log_model(model, artifact_path="xgb_model", input_example=X_train.sample(1))

if __name__ == "__main__":
    main()
