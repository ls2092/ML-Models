# 📊 ML Models Collection

This repository contains three end-to-end machine learning projects using datasets from [Kaggle](https://www.kaggle.com/). Each project demonstrates data preprocessing, model training, evaluation, and visualization.

## 🧠 Projects Overview

- **Score Predictor** – Predict student exam scores based on study hours and related features
- **Loan Approval Predictor** – Predict whether a loan application will be approved or not
- **Sales Forecast** – Forecast future Walmart sales using historical and time-series data
- **Traffic Sign Predictor** - Classifies traffic sign images into their respective categories using a convolutional neural network (CNN) trained  on labeled image data.

---

## Project Structure

ML-Models/
│
├── Sales_Forecast/
| └── archive
│ └── main.py
│
├── Loan_Predictor/
| └── archive
│ └── main.py
│
├── Score_Predictor/
| └── archive
│ └── main.py
| 
├── Traffic_Sign_Predictor/
| └── archive
│ └── main.py 
│
└── README.md

## 🧪 Prerequisites

Ensure you have **Python 3.8.2** or later installed.

Install the necessary Python packages per project using the commands below.

---

## 🚀 How to Run Each Project

### ✅ Score Predictor

> Predicts students' exam scores using Linear Regression.

````bash
cd Score_Predictor
pip install pandas matplotlib scikit-learn
python main.py


### ✅ Loan Approval Predictor

> Classifies whether a loan application will be approved based on applicant features.

```bash
cd Loan_Predictor
pip install pandas matplotlib scikit-learn
python main.py


### ✅ Sales Forecast Predictor

> Uses historical Walmart sales data and time-series features to forecast future weekly sales.

```bash
cd Sales_Forecast
pip install pandas matplotlib scikit-learn xgboost
python main.py


### ✅ Traffic Sign Predictor

> Classifies traffic sign images into their respective categories using a convolutional neural network (CNN) trained on labeled image data.

```bash
cd Traffic_Sign_Predictor
pip install pandas numpy matplotlib seaborn opencv-python scikit-learn tensorflow
python main.py
````
