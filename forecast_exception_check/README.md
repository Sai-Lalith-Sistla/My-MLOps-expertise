# Forecast Exception Report Automation

## Overview
This project automates the monitoring of forecast model outputs and sends alerts for exceptional scenarios.  
It uses Kubernetes CronJobs to periodically:
- Fetch the best model (Champion) from Azure ML Registry.
- Run forecast predictions on fresh data.
- Generate an Exception Report.
- Send Email Notifications if exceptions are detected.

---

## Project Structure

| File | Purpose |
|:-----|:--------|
| `forecast.py` | Fetch Champion model from Azure ML Registry, Run model inference and calculate results  |
| `data_engineering.py` | Fetch actual results captured over time |
| `forecast_exception_report.py` | Detect exception scenarios, end notification emails |
| `send_email.py` | S |
| `forecast_exception_k8s_cronjob.yaml` | Kubernetes CronJob definition |
| `requirements.txt` | Python dependencies |

---

