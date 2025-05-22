
#  Fraud Detection Using Isolation Forest

This repository contains a Jupyter Notebook that demonstrates how to detect fraudulent transactions using the **Isolation Forest** algorithm, an unsupervised machine learning method for anomaly detection.

##  Project Overview

Fraudulent transactions are typically rare and differ from normal transactions. This project uses the Isolation Forest algorithm to flag such anomalies based on transaction features without requiring labeled data. Due to the absence of ground truth labels, we treat this as an **unsupervised learning problem**.

##  Key Features

- Unsupervised learning using Isolation Forest
- Feature engineering and scaling
- KDE plots and dimensionality reduction (PCA and t-SNE) for result validation

---

##  File Structure

```bash
fraud_detection_using_isolation_forests.ipynb   # Main notebook
README.md                                       # Project documentation
```

---

##  Concepts and Tools Used

- **Isolation Forest** from `sklearn.ensemble`
- **KDE plots** for density-based visualization
- **PCA** (Principal Component Analysis) and **t-SNE** for dimensionality reduction

---

##  Dataset and Features

- **Rows**: 20,000 transactions
- **Columns/Features Used**:
  - `TransactionID ` [int64]
  - `AccountID `[int64]
  - `TransactionType `[object]
  - `Amount `[float64]
  - `TransactionDate `[datetime64]
  - `Season `[object]
  - `customerID `[int64]
  - `AccountType `[object]
  - `Balance `[int64]
  - `CreatedDate `[datetime64]
  - `TimeBetweenTransactions`: A custom column was created specifically for training to capture unique interactions between existing transactions

- **Preprocessing**:
  - Missing value handling
  - Dropping irrelevant columns

---

##  Model Training Details

```python
from sklearn.ensemble import IsolationForest

model = IsolationForest(
    contamination='auto',  # Automatically determines the threshold based on the dataset
    random_state=42
)
model.fit(X)
```

###  Why `contamination='auto'`?
Since the dataset does not include labels indicating which transactions are fraudulent, we used the `'auto'` setting for `contamination`. This allows the model to infer an appropriate threshold for identifying anomalies based on the distribution of the input data.

---

---

##  Results & Evaluation

- **Anomalies Detected**: 8,263 out of 20,000 transactions
- **Percentage**: \( rac{8263}{20000} 	imes 100 = 41.3\% \)

###  Visual Validation

To evaluate the quality of the model:
- **KDE (Kernel Density Estimation)** plots were used to visualize density and identify anomaly distributions.
- **PCA and t-SNE** were applied to reduce dimensions and visually inspect clustering of anomalies.

The results indicated clear and consistent grouping of anomalies, suggesting high model effectiveness in distinguishing suspicious transactions.

---

##  Full Processing Pipeline

```text
1. Load data
2. Clean & preprocess features
3. Create custom training column
4. Apply feature scaling and encoding
5. Train Isolation Forest with contamination='auto'
6. Predict anomalies and analyze results
7. Visualize using KDE, PCA, and t-SNE
8. Save and export model (.pkl)
```

---

##  Limitations

- No ground truth → model validation is visual/manual
- False positives possible due to unsupervised nature
- Contamination estimation may vary with new data

---

##  Future Improvements

- Incorporate labeled data for hybrid modeling (semi-supervised)
- Deploy model via REST API (Flask/FastAPI)
- Schedule periodic retraining for adaptive detection

---

##  References

- [Scikit-learn: Isolation Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
- [t-SNE Visualization](https://distill.pub/2016/misread-tsne/)
