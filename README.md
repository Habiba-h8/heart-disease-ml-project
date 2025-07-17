#  Heart Disease Prediction - Machine Learning Project

This project is a complete machine learning pipeline to **predict heart disease** based on patient medical records. It includes data preprocessing, feature selection, dimensionality reduction, model training, hyperparameter tuning, evaluation, and clustering analysis.

---

##  Dataset

The dataset was obtained from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) and contains 918 rows and 12 features (both numerical and categorical), including the target variable: `HeartDisease`.

---

##  Project Workflow

1. **Data Cleaning**:
   - Removed unrealistic values (e.g. 0 for `RestingBP`, `Cholesterol`)
   - Handled missing values with median imputation

2. **Preprocessing**:
   - Categorical encoding using One-Hot Encoding
   - Feature scaling using StandardScaler

3. **Feature Selection**:
   - Used both `SelectKBest` (ANOVA F-test) and `RandomForest` feature importance

4. **Dimensionality Reduction**:
   - Applied PCA to reduce feature dimensionality while retaining ~98% of variance

5. **Model Training & Evaluation**:
   - Trained and evaluated:
     - Logistic Regression
     - Decision Tree (with hyperparameter tuning)
     - Random Forest (with tuning using RandomizedSearchCV)
     - Support Vector Machine (SVM)
   - Used metrics: Accuracy, Confusion Matrix, ROC Curve, AUC

6. **Clustering**:
   - K-Means Clustering with Elbow Method
   - Hierarchical Clustering with dendrogram visualization

7. **Model Export**:
   - Final model pipeline (preprocessing + model + PCA) exported as `.pkl` using `joblib` for reproducibility

---

##  Final Accuracy Comparison

| Model               | Accuracy |
|--------------------|----------|
| Logistic Regression| 0.8043   |
| Decision Tree      | 0.7772   |
| Random Forest      | 0.8478   |
| SVM                | 0.8300   |

---

## Requirements

- Python 3.x
- scikit-learn
- pandas, numpy
- seaborn, matplotlib
- joblib
- Jupyter Notebook

You can install all required packages using:
```bash

