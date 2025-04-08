import mlflow
from mlflow.tracking import MlflowClient

# Setup MLflow client
client = MlflowClient()

# Define parameters
MODEL_NAME = "forecast_model"   # Change this to your model name
METRIC_TO_COMPARE = "rmse"      # Example metric
IMPROVEMENT_THRESHOLD = 0.05    # Challenger must be 5% better

def get_model_info(model_name, stage):
    """Fetch model info (version, run_id) from registry."""
    model_versions = client.get_latest_versions(name=model_name, stages=[stage])
    if not model_versions:
        return None
    model_version = model_versions[0]
    run_id = model_version.run_id
    return model_version.version, run_id

def get_metric_from_run(run_id, metric_key):
    """Fetch specific metric from model run."""
    run = client.get_run(run_id)
    metrics = run.data.metrics
    return metrics.get(metric_key)

def promote_model(model_name, challenger_version):
    """Promote challenger model to Production."""
    client.transition_model_version_stage(
        name=model_name,
        version=challenger_version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"Promoted challenger model v{challenger_version} to Production.")

def compare_models():
    # Get Champion and Challenger info
    champion_info = get_model_info(MODEL_NAME, stage="Production")
    challenger_info = get_model_info(MODEL_NAME, stage="Staging")

    if not challenger_info:
        print("No challenger model in staging to compare.")
        return

    if not champion_info:
        print("No existing champion. Directly promoting challenger...")
        promote_model(MODEL_NAME, challenger_info[0])
        return

    # Extract metrics
    champion_metric = get_metric_from_run(champion_info[1], METRIC_TO_COMPARE)
    challenger_metric = get_metric_from_run(challenger_info[1], METRIC_TO_COMPARE)

    print(f"Champion {METRIC_TO_COMPARE}: {champion_metric}")
    print(f"Challenger {METRIC_TO_COMPARE}: {challenger_metric}")

    # Comparison logic
    if challenger_metric < champion_metric * (1 - IMPROVEMENT_THRESHOLD):
        print("Challenger model outperforms Champion. Promoting Challenger...")
        promote_model(MODEL_NAME, challenger_info[0])
    else:
        print("Challenger did not sufficiently outperform Champion. No promotion.")

if __name__ == "__main__":
    compare_models()
