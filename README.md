# 🛠️ MLOps POC - End-to-End Retail Forecasting


## 🚀 Tools & Skills Displayed

- **Python** (`Pandas`, `Dask`, `NumPy`)
- **MLOps** (`MLflow`, `Experiment Tracking`, `Metric Comparison`)
- **Machine Learning** (`LightGBM`, `Random Forests`)
- **Data Engineering** (Scalable Pipelines, Feature Engineering, Memory Optimization)
- **Time Series Forecasting** (Lag Features, Temporal Data Handling)
- **API Deployment** (`Flask` / `FastAPI` — upcoming in Day 3)
- **Model Monitoring** (Performance Tracking, Retraining Automation — upcoming)
- **Cloud-Ready Architecture** (Modular, Scalable Design)


## 📈 High-Level Workflow

1️⃣ **Data Ingestion → Preprocessing → Feature Engineering → Data Quality Report**  
2️⃣ **Model Training → Experiment Tracking with MLflow → Metric Comparison**  
3️⃣ **Champion vs. Challenger Selection → Best Model Deployment via API**  
4️⃣ **Model Monitoring → Performance Reports → Automated Retraining**

---

## 🧩 Data Engineering

### Summary
Designed a **robust and scalable data pipeline** that transforms complex retail datasets into actionable insights, establishing a strong foundation for predictive analytics and operational excellence.

### Key Steps
- **Large-Scale Data Processing:**  
  Efficiently processed the **M5 Forecasting** dataset using **Dask**, enabling scalable handling of millions of records.
  
- **Data Filtering:**  
  Focused on the **HOBBIES** product category to optimize resource utilization and reduce noise.

- **Data Reshaping:**  
  Transformed sales data from **wide** to **long** format, enabling advanced **time-series modeling**.

- **Missing Value Handling:**  
  Applied robust **forward** and **backward** filling strategies to impute missing values accurately.

- **Feature Engineering:**  
  Created **lag features** (lags of 1, 7, and 28 days) to capture temporal sales patterns and enhance model performance.

- **Memory Optimization:**  
  Reduced memory footprint via **intelligent repartitioning** and storage in **Parquet** format.

- **Scalability & Modularity:**  
  Architected the pipeline for **easy scaling** and **modular integration** with downstream ML workflows.

---

## 🤖 Model Training & Experiment Tracking

### Summary
Built a **modular, extensible training framework** capable of handling multiple model families.  
Integrated **MLflow** for seamless experiment tracking, ensuring traceability, reproducibility, and informed decision-making.

### Key Steps
- **Baseline Model Development:**  
  Developed a **LightGBM Regressor** as the baseline model for initial performance benchmarking.

- **Experiment Tracking with MLflow:**  
  Logged **parameters**, **metrics** (like RMSE), **feature importance**, and **model artifacts** for each experiment run.

- **Hyperparameter Variations:**  
  Trained multiple model versions with different hyperparameter settings to observe their impact on performance.

- **Metric Comparison:**  
  Compared all experiments using MLflow’s tracking UI, enabling **easy identification of top-performing models**.

- **Reproducibility:**  
  Ensured that every run can be **replicated** with tracked inputs, outputs, and model versions.

---
## 📂 Dataset Instructions

Due to the size of the datasets, they are **not included** in this repository.  
Please manually download the required files from the [M5 Forecasting Accuracy Kaggle competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data):

- `sales_train_evaluation.csv`
- `sales_train_validation.csv`

Place the downloaded files into the following directory structure:

