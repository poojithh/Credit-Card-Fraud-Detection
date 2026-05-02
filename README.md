# Credit Card Fraud Detection using Hybrid Machine Learning

## Project Overview
This project detects fraudulent credit card transactions using Autoencoders, XGBoost, Bagging Classifier, and Optuna hyperparameter tuning.

## Key Features
- Autoencoder-based anomaly detection
- XGBoost and Bagging Classifier
- Optuna Bayesian hyperparameter tuning
- ROC-AUC, precision, recall, F1-score, and confusion matrix evaluation
- Feature importance analysis

## Results
XGBoost achieved around 0.97 ROC-AUC and detected 80 out of 98 fraud cases in the test set.

## How to Run
```bash
pip install -r requirements.txt
python main.py

