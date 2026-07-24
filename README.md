# Loan Approval Prediction Model

A complete Machine Learning classification project designed to predict loan approval decisions based on applicant demographics, financial information, and credit history. Developed as part of the Machine Learning Internship for **Chand Web Technology Private Limited**.

---

## Project Structure

```text
loan_approval_prediction/
├── data/
│   └── loan_approval_dataset.csv     # Raw dataset
├── src/
│   ├── __init__.py                   # Package initialization
│   ├── preprocessing.py              # Cleaning and feature engineering
│   ├── model.py                      # Model training, comparison, and serialization
│   └── utils.py                      # Plotting and visualization utilities
├── models/
│   └── best_loan_model.joblib        # Saved model pipeline (preprocessor, scaler, model)
├── reports/
│   ├── correlation_heatmap.png       # Correlation matrix of numeric features
│   ├── distribution_plots.png        # Income and Loan Amount distributions (Raw vs Log)
│   ├── confusion_matrix.png          # Test set confusion matrix
│   ├── roc_curve.png                 # ROC curve & AUC score
│   └── feature_importance.png        # Relative importance of features
├── main.py                           # End-to-end model pipeline runner
├── predict.py                        # Interactive prediction CLI
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

---

## Machine Learning Pipeline

### 1. Data Cleaning
- **Imputation Strategy**:
  - **Median** imputation is used for numerical features with missing values (`LoanAmount`, `Loan_Amount_Term`) to handle skewness robustly.
  - **Mode** (most frequent value) imputation is used for categorical features (`Gender`, `Married`, `Dependents`, `Self_Employed`, `Credit_History`).
- **Identifier Removal**: `Loan_ID` is removed to prevent overfitting on arbitrary ID strings.

### 2. Feature Engineering
Four high-impact features were engineered to improve class separation:
- **`TotalIncome`**: Combining `ApplicantIncome` and `CoapplicantIncome` to capture the overall household repayment capacity.
- **`Loan_Income_Ratio`**: `LoanAmount / TotalIncome` to measure the requested loan amount relative to total household earnings.
- **`Income_Per_Dependent`**: `TotalIncome / (Dependents_Numeric + 1)` to capture the disposable income share per household member.
- **Log Transformations**: Applied `log1p` to heavily right-skewed columns (`ApplicantIncome`, `TotalIncome`, `LoanAmount`) to align them closer to normal distributions, improving linear model performance.

### 3. Model Training & Evaluation
The dataset was split into **80% training** and **20% testing** sets using stratified sampling to preserve the target class balance (`Loan_Status`). Features were scaled using `StandardScaler`. 

Three classification algorithms were compared:
1. **Logistic Regression** (Linear Baseline)
2. **Random Forest Classifier** (Ensemble Bagging)
3. **Gradient Boosting Classifier** (Ensemble Boosting)

---

## Model Performance Results

| Model Name | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | **86.18%** | **84.00%** | **98.82%** | **90.81%** | **83.90%** |
| **Random Forest** | 85.37% | 84.54% | 96.47% | 90.11% | 86.04% |
| **Gradient Boosting** | 82.11% | 84.62% | 90.59% | 87.50% | 81.27% |

* **Selected Model**: **Logistic Regression** was selected as the best overall classifier. It achieved the highest accuracy (**86.18%**) and F1-Score (**90.81%**), showing excellent classification capability on the test set.

---

## Installation & Setup

Ensure you have Python 3.9+ installed. Follow these steps to run the project:

### 1. Clone the repository (if applicable) and Navigate to the folder:
```bash
cd loan_approval_prediction
```

### 2. Create and Activate a Virtual Environment:
* **macOS / Linux**:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```
* **Windows**:
  ```bash
  python -m venv .venv
  .venv\Scripts\activate
  ```

### 3. Install Dependencies:
```bash
pip install -r requirements.txt
```

---

## Usage Guide

### 1. Run the Training Pipeline
To run data preprocessing, train/evaluate all models, save the best classifier, and output the report plots:
```bash
python main.py
```
This script will output the training logs and save the visual plots to the `reports/` folder.

### 2. Make Predictions on New Data
To run predictions interactively using the CLI:
```bash
python predict.py
```
The script will first output predictions for two pre-configured test profiles (high-approval vs low-approval), and then prompt you to enter details for a custom applicant to get immediate predictions!

---

## Visual Reports (Saves in `reports/`)
- **`correlation_heatmap.png`**: Inspects pairwise linear relationships between numerical attributes.
- **`distribution_plots.png`**: Visualizes the density of income/loan attributes before and after log transformations.
- **`confusion_matrix.png`**: Displays true positives, true negatives, false positives, and false negatives of the selected model.
- **`roc_curve.png`**: Evaluates the true-positive rate vs false-positive rate across decision thresholds, displaying the AUC score.
- **`feature_importance.png`**: Shows the absolute coefficients of features, indicating their relative importance to the prediction.
