from datetime import datetime
import os
import joblib
import mlflow
import mlflow.lightgbm
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import optuna
import logging
import ml_utils

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model(X_train, y_train, model_params, num_boost_round=50, 
                X_valid=None, y_valid=None, early_stopping_rounds=None):
    dtrain = lgb.Dataset(X_train, label=y_train)
    
    valid_sets = None
    if X_valid is not None and y_valid is not None:
        dvalid = lgb.Dataset(X_valid, label=y_valid)
        valid_sets = [dtrain, dvalid]
    
    booster = lgb.train(
        model_params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        valid_names=["train", "valid"] if valid_sets else None,
        callbacks=[lgb.early_stopping(early_stopping_rounds)] if early_stopping_rounds and valid_sets else None
    )
    return booster


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    mse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {"mse": mse, "mae": mae, "r2": r2}

def save_model(model, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.save_model(output_path)

def objective(trial, X_train, y_train, X_valid, y_valid):
    # params = {
    #     'objective': 'regression',
    #     'eval_metric': 'rmse',
    #     'max_depth': trial.suggest_int('max_depth', 3, 10),
    #     'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
    #     'subsample': trial.suggest_float('subsample', 0.5, 1.0),
    #     'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
    #     'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
    #     'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True)
    # }

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-3, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-3, 10.0, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'bagging_freq': 1,
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 10.0, log=True),
        'max_bin': trial.suggest_int('max_bin', 100, 500),
    }
    booster = train_model(X_train, y_train, params, early_stopping_rounds=50)
    metrics = evaluate_model(booster, X_valid, y_valid)
    return metrics['mse']

def main():
    config = ml_utils.load_config("configs/train_pd_hp_config.yaml")
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
        X, y = ml_utils.load_data_pd(data_path, data_format, features, target)

        # Simple 80-10-10 split
        n = len(X)
        train_end = int(n * 0.8)
        valid_end = int(n * 0.9)

        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_valid, y_valid = X.iloc[train_end:valid_end], y.iloc[train_end:valid_end]
        X_test, y_test = X.iloc[valid_end:], y.iloc[valid_end:]

        if tune_hyperparameters:
            # To load
            loaded_params = ml_utils.load_hyperparameters(filepath = './data/lgbm_optuna_hp_config.json')
            if loaded_params:
                print(loaded_params)
                logger.info(f"hyperparameters retreived from file: {loaded_params}")
                model_params.update(loaded_params)
            else:
                print("No valid hyperparameters available.")
                logger.info("Starting hyperparameter tuning with Optuna...")
                study = optuna.create_study(direction="minimize")
                study.optimize(lambda trial: objective(trial, X_train, y_train, X_valid, y_valid), n_trials=30)

                best_params = study.best_params
                logger.info(f"Best hyperparameters found: {best_params}")
                model_params.update(best_params)
                ml_utils.save_hyperparameters(params = best_params, filepath = './data/lgbm_optuna_hp_config.json')


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
        mlflow.lightgbm.log_model(model, artifact_path="lightgbm_model", input_example=X_train.sample(1))

if __name__ == "__main__":
    main()
