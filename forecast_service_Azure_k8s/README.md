# Forecast Automation API — AzureML Managed AKS Deployment

This repository contains all necessary files to deploy a trained ML model as a **FastAPI** service on **Azure Kubernetes Service (AKS)** managed by **Azure Machine Learning (AzureML)**.

The system supports:
- Model deployment from AzureML Model Registry
- Automated preprocessing of input data
- Real-time scoring via REST API
- Containerization with Docker
- Production-grade deployment with Kubernetes

---

## 📦 Project Structure

```bash
/forecast-automation/
├── docker/
│   └── Dockerfile         # Container build for FastAPI server
├── k8s/
│   ├── deployment.yaml    # Kubernetes deployment spec
│   └── service.yaml       # Kubernetes service spec
├── deploy/
│   ├── app.py             # FastAPI application
│   └── preprocess.py      # Preprocessing logic
├── model/
│   └── model.pkl          # Trained model artifact
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation



## ⚙️ AzureML Deployment Architecture

1. **Register the trained model** into AzureML workspace.

2. **Create an Inference Environment** using `Dockerfile` or AzureML `Environment`.

3. **Define an Inference Configuration** using a FastAPI app (`app.py`).

4. **Deploy the model** onto an AzureML-managed AKS cluster.

5. **Expose the API endpoint** for real-time scoring.
