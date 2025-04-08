#!/bin/bash
echo "Starting MLflow Training Run..."
mlflow run . -P train=true