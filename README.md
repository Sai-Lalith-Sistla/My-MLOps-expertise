# 🛠️ MLOps POC - End-to-End Retail Forecasting

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



---
## 📂 Dataset Instructions

Due to the size of the datasets, they are **not included** in this repository.  
Please manually download the required files from the [M5 Forecasting Accuracy Kaggle competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data):

- `sales_train_evaluation.csv`
- `sales_train_validation.csv`

Place the downloaded files into the following directory structure:

