# NYC Airbnb Data Cleaning and Analysis

## Project Idea
Cleaning Data

---

## Project Overview
This project focuses on cleaning and exploring the **New York City Airbnb dataset (2019)**. 
The dataset includes Airbnb listings across NYC, providing insights into hosts, locations, pricing, and review patterns.  
The goal is to prepare the dataset for analysis, identify trends, and explore patterns in host activity and guest behavior.

---

## Project Structure
<img width="619" height="596" alt="image" src="https://github.com/user-attachments/assets/126a3d18-c33c-4400-9408-abafad24f20a" />

---

## Dataset Description

- **Year:** 2019  
- **Location:** New York City, NY, USA  
- **Entries:** ~49,500 listings  
- **Features:** Listing IDs, host information, neighborhood, property type, price, reviews, availability, and ratings  

### Key Questions
- Which areas have the highest concentration of listings?  
- How do prices vary across neighborhoods and property types?  
- Who are the busiest hosts and what factors contribute to their activity?  
- Can we predict booking patterns based on location and property characteristics?  

---

## Project Goals
1. **Data Cleaning**  
   - Handle missing and inconsistent values  
   - Standardize numeric and categorical columns  
   - Convert dates and categorical data to suitable formats  

2. **Exploratory Data Analysis (EDA)**  
   - Summary statistics of price, availability, and review scores  
   - Distribution of listings across neighborhoods and property types  
   - Identify outliers and anomalies  

3. **Feature Engineering**  
   - Calculate metrics such as `Average_Review_Score`, `Booking_Frequency`, `Price_per_Night`  
   - Encode categorical variables for analysis  

4. **Visualization & Insights**  
   - Create charts and plots to illustrate trends and patterns  
   - Identify the busiest hosts and neighborhoods  
   - Provide actionable insights for Airbnb operations  

---

## Installation

### Clone Repository
```bash
git clone https://github.com/yourusername/NYC-Airbnb-Data-Cleaning.git
cd NYC-Airbnb-Data-Cleaning

Create Virtual Environment

for macOS/Linux:

bash
python -m venv venv
source venv/bin/activate

for Windows:

bash
python -m venv venv
venv\Scripts\activate

Install Dependencies

bash
pip install -r requirements.txt

Usage

Run Analysis Script

bash
python scripts/run_analysis.py

This script will:

Load the raw dataset from data/raw/

Clean and preprocess the data

Perform basic EDA and generate summary statistics

Save cleaned data in data/processed/

Output charts and figures in reports/figures/

Run Jupyter Notebooks

Data Exploration:

bash
jupyter notebook notebooks/01_data_exploration.ipynb

Data Cleaning:

bash
jupyter notebook notebooks/02_data_cleaning.ipynb

Analysis & Visualization:

bash
jupyter notebook notebooks/03_analysis.ipynb

Sample Insights (Updated Values)

Average price per night across NYC listings: $145

Average review score: 4.6/5

Top 3 busiest neighborhoods: Manhattan, Brooklyn, Queens

Number of active hosts with >50 listings: 220

Folder Structure

bash
data/           # Raw and processed datasets
notebooks/      # Jupyter notebooks for EDA and cleaning
scripts/        # Python scripts for cleaning and analysis
reports/        # Visualizations and summaries
requirements.txt
README.md

Tools & Technologies

Python Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn

Environment: Jupyter Notebook, VS Code

Future Work

Predict listing occupancy based on location and price

Analyze seasonal trends in bookings

Identify correlations between host activity and guest reviews

Explore clustering techniques to segment neighborhoods by listing activity

