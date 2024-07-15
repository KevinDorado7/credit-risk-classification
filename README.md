# credit-risk-classification
Module 20 - Challenge

# Credit Risk Classification

## Overview

This project involves using various machine learning techniques to train and evaluate a model based on loan risk. The primary goal is to predict the creditworthiness of borrowers using a dataset of historical lending activity from a peer-to-peer lending services company.

## Background

The analysis was conducted to assess the feasibility of using a logistic regression model to classify loans as either healthy or high-risk. This classification helps the lending company make informed decisions about loan approvals and manage potential risks.

## Steps Performed

### 1. Data Preparation
- **Loading the Data:**
  - The dataset `lending_data.csv` was read into a Pandas DataFrame.
  
  ```python
  lending_data = pd.read_csv("Resources/lending_data.csv")

### Creating Labels and Features:

The target variable y was created from the loan_status column.

y = lending_data["loan_status"]

The feature set X was created from the remaining columns.

X = lending_data.drop(columns=["loan_status"])

### Splitting the Data:
The data was split into training and testing sets using train_test_split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

### 2. Model Training
Logistic Regression:

A logistic regression model was instantiated and trained using the training data.

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='lbfgs', random_state=1, max_iter=200)
model.fit(X_train, y_train)

### 3. Model Evaluation
Predictions:

- Predictions were made on the testing data.
y_pred = model.predict(X_test)

### Performance metrics
The model performance was evaluated using a confusion matrix and a classification report.

from sklearn.metrics import confusion_matrix, classification_report
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

### Confusion Matrix
[[14926    75]
 [   46   461]]

### Classification Report
              precision    recall  f1-score   support
           0       1.00      1.00      1.00     15001
           1       0.86      0.91      0.88       507
    accuracy                           0.99     15508
   macro avg       0.93      0.95      0.94     15508
weighted avg       0.99      0.99      0.99     15508

### 4. Analysis Summary

The logistic regression model performed exceptionally well, with an overall accuracy of 99%. It demonstrated perfect performance in predicting healthy loans and strong performance in identifying high-risk loans. This model is recommended for use in evaluating loan applications due to its high accuracy and reliability.

In this project, I made use of the assistance of ChatGPT to help me with debugging and optimizing code functionality and with the predictions and evaluatinons of the model's performance with confusion matrix and classification report and also summarizing the report.

- OpenAI. (n.d.). ChatGPT by OpenAI from https://openai.com/chatgpt