#  What is Performance Degradation Detection?

When a model is deployed in production, over time its predictions might become less accurate because:

- **Data Drift**: The input data distribution changes.
- **Prediction Drift**: The model’s output distribution changes.
- **Concept Drift**: The relationship between input and output changes.

These changes can cause performance degradation, and detecting it early is critical.



## How is it Implemented?

Typically, the approach is **monitoring + statistics + triggers**:

### 1. Monitoring Signals

**A. Data Drift Detection**: Compare current incoming feature distributions vs training distributions.

📍 **Metrics**:
- Kolmogorov-Smirnov Test (K-S Test)
- Population Stability Index (PSI)
- Wasserstein Distance (Earth Mover’s Distance)



| Feature Type                       | Best Method                           | Notes                                  |
|-------------------------------------|---------------------------------------|----------------------------------------|
| **Continuous**                     | KS Test, Wasserstein Distance         | For quick alerts                       |
| **Categorical**                    | Chi-Square Test, PSI                  | Need enough samples                    |
| **High-Dimensional Vectors** (e.g., embeddings) | MMD, Kullback-Leibler Divergence | More complex, can be slower             |
| **Time Series**                    | CUSUM, Drift Detection Models (concept drift) | Time-aware methods             |



**B. Prediction Drift Detection**: Compare current output distributions vs expected outputs.

- It occurs when the distribution of model predictions shifts from what was expected at deployment.

- This can happen even if input features don't drift — for example, the data might still "look" the same, but the relationships between features and outcomes have changed (concept drift).

📍 **Metrics**:
- KL Divergence
- JS Divergence
- Chi-Square Test

**C. Model Metrics Monitoring**: If labels are available later (delayed feedback), monitor:
- Accuracy
- Precision
- Recall
- AUC, etc.

### 2. Sampling / Windowing (for Huge Datasets)

You don't process everything in real-time when data is huge:

- **Sliding Windows**: Analyze only the recent N days/hours of data.
- **Sampling**: Smart sampling (e.g., stratified sampling) to ensure smaller but representative sets.
- **Mini-batch Evaluation**: Evaluate performance on small batches periodically.

### 3. Statistical Testing

For each window:
- Perform statistical tests between training vs recent data.
- Set thresholds: If drift score exceeds the threshold → raise an alert.
- Use control charts (like in quality management) to monitor metric trends.

### 4. Automation and Alerting

Use MLOps tools to automate:
- Batch jobs to compute drift daily/hourly (on windowed data).
- Dashboards for visualization (e.g., Prometheus + Grafana, AWS CloudWatch, Azure Monitor).
- Auto-alerts (Slack, Email, PagerDuty) when drift exceeds thresholds.

### 5. Tools Used

| Purpose | Tools |
|:--------|:------|
| **Data/Prediction Drift Monitoring** | WhyLabs, Evidently AI, Fiddler AI, Arize AI |
| **Model Monitoring** | MLflow, Seldon Core, BentoML, Vertex AI Model Monitoring |
| **Metrics Visualization** | Grafana, Kibana, Prometheus |





## Quick Implementation Flow
*(for a real-world huge-data system)*

1. Collect incoming feature & prediction data (store temporarily).
2. Sample / window the data (e.g., last 10k records daily).
3. Compute drift scores (PSI, KL divergence, K-S Test).
4. Compare scores with pre-decided thresholds.
5. If threshold crossed:
   - a. Send alert.
   - b. Trigger automatic model evaluation or retraining workflows (optional).
6. Visualize historical drifts to see trends.



### Advanced Techniques

- **Active Drift Detection**:  
  Use methods like **ADWIN (Adaptive Windowing)** algorithm for real-time drift detection.

- **Unsupervised Monitoring**:  
  Use clustering or anomaly detection when labels aren't available.

- **Concept Drift Detection**:  
  Techniques like **DDM (Drift Detection Method)**, **EDDM (Early Drift Detection Method)**, or **ADWIN** are used if you expect fundamental target shifts.

