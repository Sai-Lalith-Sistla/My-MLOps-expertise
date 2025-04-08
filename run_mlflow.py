# run_mlflow.py
import subprocess

print("Running MLflow experiment...")

subprocess.run(["python", "train.py"], check=True)

print("Run completed!")
