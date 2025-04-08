from datetime import datetime
import os
import yaml
import joblib
import mlflow
import mlflow.xgboost
import dask.dataframe as dd
import dask.distributed
import xgboost as xgb
import xgboost.dask as dxgb
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

import os
os.environ["DASK_DISTRIBUTED__DIAGNOSTICS__NVML"] = "False"

import logging

# Mute the NVML GPU errors
logging.getLogger("distributed.diagnostics.nvml").setLevel(logging.CRITICAL)


# Start Dask client
cluster = dask.distributed.LocalCluster()
client = dask.distributed.Client(cluster)

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_data(data_path, data_format, features, target):
    if data_format.lower() == "csv":
        df = dd.read_csv(data_path)
    elif data_format.lower() == "parquet":
        df = dd.read_parquet(data_path)

    
    X = df[features]
    y = df[target]
    return X, y

def train_model(X_train, y_train, X_valid, y_valid, model_params, early_stopping_rounds):
    # xgb.dask
    dtrain = dxgb.DaskDMatrix(client, X_train, y_train)
    dvalid = dxgb.DaskDMatrix(client, X_valid, y_valid)

    evals_result = {}
    
    # Define XGBoost params for CPU
    params = {
        'tree_method': 'hist',  # 'hist' is fast and CPU-optimized
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'verbosity': 1
    }

    # Train the model
    output = dxgb.train(
        client,
        params,
        dtrain,
        num_boost_round=50,
        evals=[(dtrain, 'train')]
    )
    # output = dxgb.train(
    #     client=client,
    #     params=model_params,
    #     dtrain=dtrain,
    #     num_boost_round=model_params.get('n_estimators', 100),
    #     evals=[(dtrain, 'train'), (dvalid, 'validation')],
    #     early_stopping_rounds=early_stopping_rounds,
    #     evals_result=evals_result
    # )
    return output['booster'], evals_result

def evaluate_model(model, X_test, y_test):
    dtest = dxgb.DaskDMatrix(client, X_test, y_test)
    y_pred = dxgb.predict(client=client, model=model, data=dtest)
    y_pred = y_pred.compute()
    y_true = y_test.compute()

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
    with mlflow.start_run(run_name):
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

        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        X_train = X.loc[:train_end]
        y_train = y.loc[:train_end]

        X_valid = X.loc[train_end:valid_end]
        y_valid = y.loc[train_end:valid_end]

        X_test = X.loc[valid_end:]
        y_test = y.loc[valid_end:]

        X_train.head()
        # # Train
        # model, evals_result = train_model(X_train, y_train, X_valid, y_valid, model_params, early_stopping_rounds)

        # # Evaluate
        # metrics = evaluate_model(model, X_test, y_test)
        # print(f"Test Metrics: {metrics}")
        
        # # Log metrics
        # mlflow.log_metrics(metrics)

        # # Save
        # save_model(model, model_output_path)
        # mlflow.xgboost.log_model(model, artifact_path="model")

if __name__ == "__main__":
    main()
