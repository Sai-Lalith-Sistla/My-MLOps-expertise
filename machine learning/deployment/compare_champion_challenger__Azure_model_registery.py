from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
import argparse
import requests
import os
import traceback

def send_notification(message, webhook_url):
    headers = {'Content-Type': 'application/json'}
    payload = {
        "text": message
    }
    response = requests.post(webhook_url, json=payload, headers=headers)
    if response.status_code != 200:
        print(f"Failed to send notification: {response.text}")

def compare_models(model_name, metric_name="rmse", mode="min", improvement_threshold=0.01, webhook_url=None):
    try:
        # Authenticate
        credential = DefaultAzureCredential()
        ml_client = MLClient(
            credential=credential,
            subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID"),
            resource_group=os.getenv("AZURE_RESOURCE_GROUP"),
            workspace_name=os.getenv("AZURE_WORKSPACE_NAME")
        )

        # Fetch all versions of the model
        models = ml_client.models.list(name=model_name)
        models = sorted(models, key=lambda x: x.created_on, reverse=True)

        if len(models) < 2:
            msg = "Not enough models to compare."
            print(msg)
            if webhook_url:
                send_notification(msg, webhook_url)
            return

        champion = models[1]
        challenger = models[0]

        champion_metric = float(champion.properties.get(metric_name))
        challenger_metric = float(challenger.properties.get(metric_name))

        print(f"Champion ({champion.version}) {metric_name}: {champion_metric}")
        print(f"Challenger ({challenger.version}) {metric_name}: {challenger_metric}")

        if mode == "min":
            improvement = (champion_metric - challenger_metric) / champion_metric
        else:
            improvement = (challenger_metric - champion_metric) / champion_metric

        print(f"Improvement: {improvement:.2%}")

        if improvement > improvement_threshold:
            msg = f"✅ Challenger (v{challenger.version}) BEATS Champion (v{champion.version}) with {metric_name} improvement of {improvement:.2%}!"
            print(msg)
            if webhook_url:
                send_notification(msg, webhook_url)

            # Update tags
            ml_client.models.update(challenger.name, challenger.version, tags={"champion": "true"})
            ml_client.models.update(champion.name, champion.version, tags={"champion": "false"})
        else:
            msg = f"❌ Challenger (v{challenger.version}) DID NOT outperform Champion (v{champion.version}). Improvement: {improvement:.2%}."
            print(msg)
            if webhook_url:
                send_notification(msg, webhook_url)

    except Exception as e:
        err_msg = f"🚨 ERROR during model comparison: {str(e)}\n{traceback.format_exc()}"
        print(err_msg)
        if webhook_url:
            send_notification(err_msg, webhook_url)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--metric_name", default="rmse")
    parser.add_argument("--mode", choices=["min", "max"], default="min")
    parser.add_argument("--improvement_threshold", type=float, default=0.01)
    parser.add_argument("--webhook_url", required=False)
    args = parser.parse_args()

    compare_models(
        model_name=args.model_name,
        metric_name=args.metric_name,
        mode=args.mode,
        improvement_threshold=args.improvement_threshold,
        webhook_url=args.webhook_url
    )
