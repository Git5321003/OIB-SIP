# Housing Price Prediction (Linear Regression)

## Project Overview
This project focuses on predicting house prices in the **Delhi region** using **multiple linear regression**. The aim is to model the relationship between property prices and key factors such as **area, number of bedrooms, bathrooms, and parking spaces**.  

By building this model, the real estate company can make data-driven decisions to **optimize pricing strategies** and understand the factors that most influence property values.

---

## Problem Statement
The dataset contains property listings with features affecting price. The company wants to:  

- Identify which factors (area, rooms, bathrooms, etc.) impact house prices.  
- Create a linear regression model that quantitatively relates these variables to property prices.  
- Measure model accuracy and understand how well the selected variables predict prices.

---

## Key Concepts and Challenges
1. **Data Collection:** Obtain a dataset of Delhi properties with numerical features and target prices.  
2. **Data Exploration and Cleaning:** Inspect the dataset, handle missing values, and ensure data quality.  
3. **Feature Selection:** Identify which variables contribute significantly to predicting house prices.  
4. **Model Training:** Implement multiple linear regression using **Scikit-Learn**.  
5. **Model Evaluation:** Assess performance using metrics such as **R-squared (R²)** and **Mean Squared Error (MSE)**.  
6. **Visualization:** Illustrate relationships between predicted and actual prices using scatter plots and residual plots.

---

## Dataset Description
- **Features (Sample):**  
  - `Area (sq.ft)`  
  - `Bedrooms`  
  - `Bathrooms`  
  - `Parking Spaces`  
  - `Age of Property`  
- **Target Variable:**  
  - `Price (INR)`  
- **Dataset Size (Example):**  
  - 1,500 properties  
  - Area: 500–3,500 sq.ft  
  - Price: ₹25 lakh – ₹5 crore  

---

## Learning Objectives
- Understand the principles of **linear regression**.  
- Gain hands-on experience in **building predictive models**.  
- Develop skills in **data cleaning, feature selection, and model evaluation**.  
- Learn to **interpret model coefficients** and performance metrics.  

---

## Folder Structure
```text
housing_price_prediction/
│
├── data/
│   ├── raw/
│   │   └── delhi_housing_data.csv
│   └── processed/
│       └── cleaned_data.csv
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_model_training_evaluation.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── evaluation.py
│   └── visualization.py
│
├── requirements.txt
├── main.py
└── README.md
