# Heart Disease Prediction Pipeline
# Overview

This project builds an end-to-end machine learning pipeline to predict the likelihood of heart disease using patient clinical data.
The dataset comes from Kaggle: Heart Failure Prediction Dataset.

# The workflow covers:

Data preprocessing (encoding, scaling, transformations)

Building a scikit-learn pipeline

Training Logistic Regression and Random Forest models

Hyperparameter tuning with GridSearchCV

Model evaluation with accuracy, precision, recall, F1-score, confusion matrix, and ROC AUC

Packaging into a deployable prediction function

# Dataset

Rows: 918 patient records

Features (11 total):

Numerical: Age, RestingBP, Cholesterol, MaxHR, Oldpeak

Categorical: Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope, FastingBS

Target: HeartDisease (binary: 1 = disease, 0 = no disease)

# Methodology

# Preprocessing

One-hot encoding for multi-class categorical variables

Label encoding for binary categorical variables

StandardScaler for numerical features

Managed using ColumnTransformer

# Modeling

Baseline: Logistic Regression

Tuned model: Random Forest Classifier (best with 50 estimators)

# Pipeline

Combined preprocessing + modeling into a single scikit-learn pipeline

Enabled easy cross-validation and deployment

# Results

Logistic Regression: ~84.9% CV accuracy

Random Forest (final model): ~86.0% CV accuracy

# Test Set Performance

Accuracy: 86%

Precision: 0.87 (class 0), 0.86 (class 1)

Recall: 0.82 (class 0), 0.90 (class 1)

F1-score: 0.84 (class 0), 0.88 (class 1)

ROC AUC: 0.926

Confusion Matrix

[[67, 15],
 [10, 92]]

The model achieves high recall (90%) for heart disease cases, which is critical in healthcare applications.

[Demo Video](https://drive.google.com/file/d/1SXzvP1QcFhTZTLYLmy8PM0Svhsh46G0r/view?usp=drive_link)

With a strong pipeline, high recall, and deployable design, this project demonstrates end-to-end ML skills for healthcare prediction tasks.
