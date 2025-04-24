# Credit Card Fraud Detection

This project implements machine learning algorithms to detect fraudulent credit card transactions based on transaction time and amount.

## Project Overview

Credit card fraud is a significant concern in e-commerce. This project uses various machine learning techniques to identify potentially fraudulent transactions and help reduce financial losses.

## Features

- Data exploration and visualization of credit card transactions
- Handling of imbalanced dataset using:
  - TomekLinks
  - Random Undersampling
  - SMOTE
- Implementation of multiple ML models:
  - Support Vector Classifier
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - K-Nearest Neighbors
  - Artificial Neural Network

## Model Performance

| ML Algorithm | Cross Validation Score | ROC AUC Score | F1 Score (Fraud) | GMean Score |
|--------------|------------------------|---------------|------------------|-------------|
| SVC | 91.08% | 89.68% | 75% | 90.54% |
| Logistic Regression | 97.62% | 89.69% | 79% | 90.65% |
| Decision Tree | 90.69% | 91.12% | 66% | 90.15% |
| Random Forest | 97.48% | 93.37% | 84% | 91.87% |
| KNN | 93.52% | 91.89% | 79% | 92.41% |
| ANN | 93.52% | 91.89% | 79% | 92.41% |

## Key Findings

- Random Forest showed the best overall performance with high cross-validation score (97.48%) and F1 score (84%)
- The dataset was highly imbalanced and required special handling
- No missing values were found in the dataset

## Requirements

- Python 3.x
- Libraries:
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - scikit-learn
  - tensorflow
  - imblearn
  - optuna
  - plotly

## Usage
1. Download data at `https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data`
2. Clone the repository
3. Ensure you have the required dependencies installed
4. Open `CreditFraud.ipynb` in Jupyter Notebook
5. Run the cells sequentially to reproduce the analysis

## Data

The project uses a credit card transaction dataset containing:
- Time
- Transaction Amount
- Anonymous Features (V1-V28)
- Class (0 for legitimate, 1 for fraudulent)
