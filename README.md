a. Problem Statement
Heart disease is one of the leading causes of death worldwide. Early diagnosis plays a critical role in preventing severe complications.This project aims to predict the presence of heart disease using multiple machine learning classification algorithms and compare their performance using standard evaluation metrics.

b. Dataset Description
Dataset: Heart Disease Dataset
Source: The Heart Disease dataset was obtained from the UCI Machine Learning Repository and accessed via Kaggle.
Total Instances: 1025
Total Features: 13
Target Variable: Presence of heart disease (Binary Classification)

c. Models Used & Evaluation Metrics

| ML Model            | Accuracy  |   AUC     | Precision  |  Recall   |  F1       |  MCC      |
| ------------------- | --------  | --------  | ---------  |  -------- |  -------- |  -------- |
| Logistic Regression | 0.857     | 0.923     | 0.831      | 0.905     | 0.866     | 0.715     |
| Decision Tree       | 0.997     | 0.997     | 1.000      | 0.994     | 0.997     | 0.994     |
| KNN                 | 0.926     | 0.984     | 0.915      | 0.943     | 0.929     | 0.852     |
| Naive Bayes         | 0.831     | 0.910     | 0.811      | 0.875     | 0.842     | 0.664     |
| Random Forest       | 0.997     | 1.000     | 1.000      | 0.994     | 0.997     | 0.994     |
| XGBoost             | 0.997     | 0.998     | 1.000      | 0.994     | 0.997     | 0.994 `   |


Model Observations

| ML Model Name       | Observation about the model performance                |
| ------------------- | ------------------------------------------------------ |
| Logistic Regression | Performed well with interpretable results              |
| Decision Tree       | Easy to interpret but prone to overfitting             |
| KNN                 | Performance depends on optimal k value                 |
| Naive Bayes         | Fast and efficient but assumes feature independence    |
| Random Forest       | Strong ensemble performance with reduced overfitting   |
| XGBoost             | Best overall performance with highest accuracy and MCC |
