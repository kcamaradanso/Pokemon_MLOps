#  Pokémon MLOps Project

## Project Overview

This project implements an end-to-end MLOps pipeline using the Pokémon dataset from Kaggle.  
The objective is to build a machine learning model capable of predicting whether a Pokémon is **legendary or not**, following a structured and reproducible workflow.

The project integrates:

- Data engineering
- Machine learning
- Experiment tracking with MLflow
- Modular Python architecture

---

##  Objective

Predict the variable:

👉 **is_legendary** (binary classification)

Using Pokémon statistics and characteristics.

---

## 📂 Project Structure
<img width="183" height="384" alt="image" src="https://github.com/user-attachments/assets/105b7191-7855-45a2-94f2-46e3506639c3" />

pokemon_mlops/
│
├── application/
│   ├── data_collection.py
│   ├── data_preparation.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   ├── modeling_mlflow.py
│
├── domain/
│   ├── transform.py
│   ├── feature_engineering.py
│   ├── model.py
│
├── notebooks/
│   ├── analysis.ipynb
│
├── data/ (ignored in Git)
│
├── README.md
└── requirements.txt


---

##  Pipeline Stages

The project follows five main stages:

### 1️ Data Collection  
- Load Pokémon dataset  
- Initial exploration

### 2️ Data Preparation  
- Cleaning  
- Handling missing values  
- Formatting

### 3️ Feature Engineering  
- Selection of numeric features  
- Scaling  
- Transformation

### 4️ Modeling  
- Logistic Regression  
- Random Forest  
- GridSearch optimization  
- Evaluation with:
  - F1-score  
  - ROC-AUC  
  - Precision/Recall  

### 5️ MLflow Integration  
- Experiment tracking  
- Parameter logging  
- Metric comparison  
- Model registry

---

## Results

Final selected model:

👉 **Random Forest**

Based on:

- Higher F1-score  
- Better ROC-AUC  
- Robustness to class imbalance

---

## 🛠 How to Run the Project

### Install dependencies

pip install -r requirements.txt


### Run the pipeline (without MLflow)

python -m pokemon_mlops.application.data_collection
python -m pokemon_mlops.application.data_preparation
python -m pokemon_mlops.application.feature_engineering
python -m pokemon_mlops.application.modeling


### Run with MLflow

mlflow ui
python -m pokemon_mlops.application.modeling_mlflow


---

## Author

**Khady Camara Danso and Jean Bacquet **  
Master in Data Science / MLOps Project

---

## Dataset

Source: Kaggle – Pokémon Dataset
