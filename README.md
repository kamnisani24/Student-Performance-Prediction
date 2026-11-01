
# 🎓 Student Performance Prediction and Early Warning System

## 📌 Project Overview

This project is an AI-powered Student Performance Prediction System built using Python and Machine Learning.

It predicts students' academic risk based on attendance, internal marks, assignment completion, previous semester GPA, study hours and class participation.

The project uses a synthetically generated dataset of 1000 students for educational and demonstration purposes.

## 🎯 Objectives

- Predict student academic risk.
- Identify High Risk and Low Risk students.
- Evaluate Machine Learning model performance.
- Provide an interactive dashboard.
- Generate downloadable student reports.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit

## 🤖 Machine Learning Algorithm

Random Forest Classifier

## 📊 Model Evaluation

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## ✨ Key Features

1. Student Academic Risk Prediction
2. Prediction Confidence
3. Important Prediction Factors
4. Academic Risk Distribution
5. Student Attendance Analysis
6. Student Dataset Preview
7. High Risk and Low Risk Filtering
8. Downloadable Student Reports

## 📁 Project Structure

- app.py – Streamlit dashboard
- generate_data.py – Student dataset generation
- train_model.py – Machine Learning model training
- student_data.csv – Generated student dataset
- student_risk_model.pkl – Trained ML model
- README.md – Project documentation

## ▶️ How to Run

### 1. Install Required Libraries

```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib
```

### 2. Generate Student Dataset

```bash
python generate_data.py
```

### 3. Train the Machine Learning Model

```bash
python train_model.py
```

### 4. Run the Streamlit Dashboard

```bash
streamlit run app.py
```

## ⚠️ Disclaimer

This project uses a synthetically generated dataset for educational and demonstration purposes. Predictions are not validated for real-world academic decisions.

## 👩‍💻 Developed By

Kamni Sani

B.Tech Computer Science Engineering
