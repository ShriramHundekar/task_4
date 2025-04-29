# task_4
Binary classification using Logistic Regression on the Breast Cancer Wisconsin dataset. Includes data preprocessing, model training, evaluation with confusion matrix, precision, recall, ROC-AUC, and sigmoid visualization.
#  Breast Cancer Classification using Logistic Regression

This project implements **binary classification** using **Logistic Regression** on the **Breast Cancer Wisconsin Diagnostic Dataset**.  
It is created as part of **Task 4** for the **AI & ML Internship** program.

---

## Dataset Overview

- **Dataset Source**: Breast Cancer Wisconsin (Diagnostic) Dataset (uploaded as `data.csv`)
- **Target Variable**:
  - `diagnosis`:
    - `M` (Malignant) → Cancer present
    - `B` (Benign) → Cancer absent
- **Feature Variables**:  
  Numeric measurements of tumor size, shape, texture, compactness, etc.

| Column Type | Description |
|:------------|:------------|
| ID, Unnamed: 32 | Dropped (not useful for modeling) |
| Diagnosis | Target variable (to be predicted) |
| Other columns | Numerical features for tumor characteristics |

---

## Project Objectives

- Load and preprocess the dataset
- Encode target variable for binary classification
- Train a **Logistic Regression** model
- Evaluate model performance using:
  - Confusion Matrix
  - Precision, Recall, F1-Score
  - ROC Curve and AUC Score
- Visualize model outputs
- Understand and explain the **Sigmoid Function** used in Logistic Regression

---

## Tools and Libraries Used

- **Python 3.x**
- Pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn (LogisticRegression, metrics, preprocessing)

---

## Project Workflow

### Step 1: Import Libraries
- Imported Pandas, NumPy, Matplotlib, Seaborn, scikit-learn packages.

### Step 2: Load Dataset
- Loaded `data.csv` into a Pandas DataFrame.
- Inspected dataset structure.

### Step 3: Preprocessing
- Dropped irrelevant columns: `id` and `Unnamed: 32`
- Encoded `diagnosis` column: 
  - `M` → 1 (Malignant)
  - `B` → 0 (Benign)
- Checked for missing/null values.
- Defined independent features `X` and target variable `y`.

### Step 4: Train-Test Split
- Split dataset into:
  - 80% for training
  - 20% for testing
- Used `train_test_split()` from scikit-learn.

### Step 5: Feature Scaling
- Applied **StandardScaler** to standardize feature values.

### Step 6: Model Training
- Trained a **Logistic Regression** model on the scaled training data.

### Step 7: Model Evaluation
- Evaluated predictions using:
  - **Confusion Matrix**
  - **Classification Report**: Precision, Recall, F1-Score, Support
  - **ROC Curve** and **AUC Score**

### Step 8: Visualization
- Plotted:
  - **ROC Curve** with AUC value
  - **Sigmoid Function** (explaining probability mapping)

### Step 9: Understanding the Sigmoid Function
- Plotted sigmoid curve:
  - Converts any real number into a probability between 0 and 1.
  - Key function behind Logistic Regression probability outputs.

---

## Evaluation Metrics

| Metric | Description |
|:-------|:------------|
| **Confusion Matrix** | Shows True Positive, True Negative, False Positive, False Negative |
| **Precision** | How many predicted positives were actually correct |
| **Recall** | How many actual positives were correctly predicted |
| **F1-Score** | Harmonic mean of precision and recall |
| **ROC Curve** | Graph of True Positive Rate vs False Positive Rate |
| **AUC (Area Under Curve)** | Overall performance summary, 1.0 = perfect model |

---

## Files Included

| File | Description |
|:-----|:------------|
| `BreastCancer_LogisticRegression.ipynb` | Full Jupyter Notebook with code, graphs, and outputs |
| `data.csv` | Breast cancer dataset used for training and testing |
| `README.md` | This detailed project explanation |

---

## How to Run the Project

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/breast-cancer-logistic-regression.git
   cd breast-cancer-logistic-regression
