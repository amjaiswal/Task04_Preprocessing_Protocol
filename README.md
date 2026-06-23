# Task 04 - AI/ML Preprocessing Protocol

## 📌 Project Overview

This project demonstrates a reusable and leak-free machine learning preprocessing pipeline using Scikit-learn.

The preprocessing pipeline is trained only on the training dataset and reused during inference to prevent data leakage.

---

## Features

- Missing Value Imputation
- Feature Scaling
- One-Hot Encoding
- ColumnTransformer
- Scikit-learn Pipeline
- Train/Test Split
- Saved Preprocessor using Joblib
- Reusable Inference Pipeline

---

## Technologies

- Python
- Pandas
- Scikit-learn
- Joblib

---

## Project Structure

```
Task04_Preprocessing_Protocol
│
├── data/
│   └── employee_attrition.csv
│
├── saved_models/
│   └── preprocessor.pkl
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
│
├── main.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Run Training

```bash
python src/train.py
```

---

## Run Prediction

```bash
python src/predict.py
```

---

## Run Complete Project

```bash
python main.py
```

---

## Concepts Used

- Data Preprocessing
- Data Imputation
- Standard Scaling
- One-Hot Encoding
- Machine Learning Pipeline
- ColumnTransformer
- Feature Engineering
- Data Leakage Prevention

---

## Author

Amar Jaiswal