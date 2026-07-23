# Assignment 4: KNN Classification — Breast Cancer Wisconsin Diagnostic Dataset

## Objective
Build a **K-Nearest Neighbors (KNN)** classification model to predict whether a breast tumor is **Malignant (M)** or **Benign (B)** based on 30 diagnostic measurements computed from digitized fine needle aspirate (FNA) images of breast masses.

---

## Dataset Link
[Breast Cancer Wisconsin (Diagnostic) Data Set – Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

| Property | Value |
|---|---|
| Samples | 569 |
| Features | 30 numerical |
| Target Classes | Malignant (212) / Benign (357) |
| Missing Values | Only in `Unnamed: 32` (dropped) |

---

## Libraries Used

| Library | Version | Purpose |
|---|---|---|
| `pandas` | ≥ 1.5 | Data loading & manipulation |
| `numpy` | ≥ 1.23 | Numerical computation |
| `matplotlib` | ≥ 3.6 | Plotting & visualization |
| `seaborn` | ≥ 0.12 | Statistical visualizations |
| `scikit-learn` | ≥ 1.2 | Preprocessing, KNN model, metrics |

---

## Methodology

### Task 1 — Data Understanding
- Loaded dataset using Pandas
- Displayed first 5 records
- Identified 30 numerical features and `diagnosis` as the target variable
- Explored dataset info and summary statistics

### Task 2 — Data Preprocessing
- **Missing Values:** `Unnamed: 32` column was entirely NaN → dropped
- **Unnecessary Columns:** `id` column (row identifier) → dropped
- **Encoding:** `LabelEncoder` → Benign = 0, Malignant = 1
- **Normalization:** `StandardScaler` (zero mean, unit variance) — critical for distance-based KNN
- **Split:** 80% training (455 samples) / 20% testing (114 samples), `random_state=42`

### Task 3 — Model Development
- Trained `KNeighborsClassifier(n_neighbors=5)` with Euclidean distance (Minkowski p=2)
- Generated predictions on the test set
- Plotted Elbow Method (K=1 to 25) to justify K=5

### Task 4 — Model Evaluation
Evaluated using accuracy, precision, recall, F1-score, and confusion matrix.

### Task 5 — Conclusion
Summarized findings, importance of feature scaling in KNN, and one key limitation.

---

## Results

| Metric | Score |
|---|---|
| **Accuracy** | **94.74%** |
| **Precision** (Malignant) | 93.02% |
| **Recall** (Malignant) | 93.02% |
| **F1-Score** (Malignant) | 93.02% |

### Confusion Matrix

|  | Predicted Benign | Predicted Malignant |
|---|---|---|
| **Actual Benign** | 68 (TN) | 3 (FP) |
| **Actual Malignant** | 3 (FN) | 40 (TP) |

---

## Observations
1. **High Overall Accuracy (94.74%):** Cell nucleus measurements are highly discriminative for tumor classification.
2. **Balanced Precision & Recall (93.02% each for Malignant):** The model is symmetric in errors — 3 false positives and 3 false negatives.
3. **Benign class outperforms Malignant class** due to more training samples and tighter clustering in feature space.

---

## Conclusion

The KNN classifier (K=5) achieved strong results on the Wisconsin Breast Cancer dataset with 94.74% accuracy. Feature standardization using `StandardScaler` was essential — without it, features with large ranges (like *area*) would dominate Euclidean distance calculations and skew predictions. The key limitation of KNN is its O(n) inference cost: it must compare every test point against all training samples, which becomes prohibitively slow on large datasets. Additionally, KNN suffers from the curse of dimensionality in high-dimensional spaces, where distance metrics become less meaningful.

---

## How to Run

```bash
# Clone the repository
git clone <your-repo-url>
cd <repo-folder>

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn notebook

# Launch Jupyter
jupyter notebook Assignment-4.ipynb
```
"# Breast-Cancer-Classification-using-K-Nearest-Neighbors" 
