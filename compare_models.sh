#!/bin/bash
echo "comparing champion challenger ...."
python src/deployment/compare_champion_challenger.py --model_name my_forecast_model --metric_name rmse --mode min --improvement_threshold 0.02
