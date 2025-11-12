# 🍷 Wine Quality Prediction

## 🧠 Project Overview
This project aims to **predict the quality of wine** using its **chemical characteristics** through **Machine Learning classification models**.  
The project demonstrates how data-driven insights can be applied to viticulture — analyzing the relationship between physicochemical properties (like acidity, density, and alcohol content) and perceived wine quality.

The dataset consists of multiple **chemical test results** for red and white wines, and the target variable (`quality`) represents a score between **0 and 10**, based on sensory evaluations.

---

## 📊 Dataset Description

**Source:** UCI Machine Learning Repository — *Wine Quality Dataset*  
**Link:** [Wine Quality Dataset - UCI Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality)

The dataset contains **4,898 white wine samples** and **1,599 red wine samples**.  
Each sample is described by 11 physicochemical attributes and one sensory score (quality).

| Feature | Description |
|----------|-------------|
| `fixed acidity` | Non-volatile acids in wine (tartaric acid, etc.) |
| `volatile acidity` | Amount of acetic acid (vinegar-like smell) |
| `citric acid` | Adds freshness and flavor to wine |
| `residual sugar` | Remaining sugar after fermentation |
| `chlorides` | Amount of salt content |
| `free sulfur dioxide` | SO₂ in free form, helps prevent microbial growth |
| `total sulfur dioxide` | Total SO₂ concentration (free + bound) |
| `density` | Mass per unit volume; related to sugar/alcohol content |
| `pH` | Acidity level of the wine |
| `sulphates` | Contribute to wine preservation and flavor |
| `alcohol` | Alcohol content by volume |
| `quality` | Sensory rating (integer between 0 and 10) |

---

## Project Structure
wine-quality-prediction/
│
├── data/
│   └── WineQT.csv
│
├── notebooks/
│   └── wine_quality_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── evaluation.py
│
├── models/
│   └── (saved models)
│
├── results/
│   ├── feature_importance.png
│   ├── confusion_matrices.png
│   └── model_comparison.png
│
├── requirements.txt
├── main.py
└── README.md

---

## 🧩 Objectives

1. **Predict** wine quality using supervised learning algorithms.  
2. **Compare** the performance of multiple classifier models:
   - Random Forest Classifier  
   - Stochastic Gradient Descent (SGD) Classifier  
   - Support Vector Classifier (SVC)  
3. **Visualize** relationships between chemical features and wine quality.  
4. **Evaluate** model accuracy and tune hyperparameters for optimal results.

---

## 🛠️ Tech Stack

| Category | Tools & Libraries |
|-----------|------------------|
| Programming Language | Python 3.x |
| Data Analysis | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn |
| Machine Learning | scikit-learn |
| Environment | Jupyter Notebook / VS Code |

---

## 🔍 Exploratory Data Analysis (EDA)

- Visualized **correlation heatmaps** to identify key influencing features.  
- Observed that **alcohol**, **volatile acidity**, and **sulphates** had the strongest correlations with wine quality.  
- Checked for **missing values**, **outliers**, and **feature distributions**.  
- Applied **scaling and normalization** for model consistency.

---

## ⚙️ Model Training & Evaluation

Three models were implemented:

| Model | Accuracy (approx.) | Key Insights |
|--------|--------------------|---------------|
| **Random Forest Classifier** | 83.2% | High performance with balanced precision and recall |
| **SVC (Support Vector Classifier)** | 81.7% | Performs well after feature scaling |
| **SGD Classifier** | 78.4% | Fast training, suitable for large datasets |

*Metrics used:*  
- Accuracy  
- Precision, Recall, F1-Score  
- Confusion Matrix  
- ROC Curve (for model comparison)

---

## 📈 Key Insights

- Wines with **higher alcohol** and **moderate acidity** tend to score better.  
- Excessive **volatile acidity** negatively affects quality.  
- Ensemble methods (like Random Forest) outperform linear models for this dataset.  
- Feature scaling significantly improves the performance of SVC and SGD models.

---

## 🚀 How to Run the Project

1. Clone the repository
   ```bash
   git clone https://github.com/<your-username>/wine-quality-prediction.git
   cd wine-quality-prediction
   
2. Install dependencies
   pip install -r requirements.txt

3. Run the notebook or script
   jupyter notebook wine_quality_prediction.ipynb

4. View results
   Model evaluation metrics and visualizations will be displayed inline.

---

## 🧪 Future Improvements

Add hyperparameter tuning using GridSearchCV.

Implement deep learning models (e.g., ANN) for non-linear feature mapping.

Deploy the best model as a web app using Streamlit or Flask.

Integrate explainable AI (XAI) to interpret model predictions.

---

## 📚 References

UCI Machine Learning Repository — Wine Quality Data Set

Scikit-learn documentation

J. Paulo Cortez et al., "Modeling wine preferences by data mining from physicochemical properties", Decision Support Systems, 2009.

---

## 🏁 Conclusion

This project demonstrates how machine learning can effectively classify wine quality using measurable physicochemical indicators.
Among the models tested, the Random Forest Classifier yielded the most reliable and interpretable results, 
showing that ensemble techniques are particularly powerful for multi-feature datasets.
