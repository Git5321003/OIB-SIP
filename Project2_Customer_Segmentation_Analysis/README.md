# Marketing Analytics: Customer Segmentation

## Project Idea
Customer Segmentation Analysis

## Project Overview
Customer segmentation is a key marketing strategy that involves dividing a customer base into distinct segments based on shared characteristics, behaviors, or demographics. This allows businesses to understand customer needs more deeply and deliver personalized, targeted marketing campaigns.  

In this project, we performed **Customer Segmentation Analysis** using K-Means clustering to identify meaningful customer groups and propose actionable marketing strategies.

---

## Objectives
The main objectives of this project are:  
- To explore and clean customer data for analysis.  
- To perform feature engineering for meaningful insights.  
- To apply K-Means clustering to identify distinct customer segments.  
- To interpret cluster characteristics and recommend targeted marketing strategies.

---

## Dataset
- **Size:** 2,205 observations  
- **Columns:** 39  
- **Features:** Includes demographic information, purchase behaviors, income levels, relationship status, and more.  
- **Note:** Some column descriptions in the original dataset do not match the actual column names; assumptions were made based on feature names.

---

## Methodology

### 1. Data Preparation and Cleaning
- Missing values handled appropriately.  
- Relevant features selected for clustering analysis.  
- Categorical features encoded as required.  

### 2. Exploratory Data Analysis (EDA)
- Visualized distribution of features such as income, purchase totals, and relationship status.  
- Identified trends and anomalies in the data.

### 3. Feature Engineering
- Created meaningful features to improve clustering results.  
- Focused on features like `Income`, `In_relationship`, and `MntTotal` (total purchase amount).

### 4. K-Means Clustering
- Determined the optimal number of clusters using:  
  - **Elbow Method**  
  - **Silhouette Analysis**  
- Applied K-Means clustering algorithm to group customers.

### 5. Cluster Exploration
- Interpreted clusters to understand customer characteristics.  
- Analyzed each cluster based on income, relationship status, and purchase behavior.

---

## Results

- **Optimal number of clusters:** 4  
- **Cluster Characteristics:**
  | Cluster | Description | % of Customers |
  |---------|-------------|----------------|
  | 0       | High-value customers in a relationship (married/together) | 26% |
  | 1       | Low-value single customers | 21% |
  | 2       | High-value single customers | 15% |
  | 3       | Low-value customers in a relationship | 39% |

---

## Marketing Recommendations

| Cluster | Strategy |
|---------|----------|
| 0 (High-value in relationship) | Promote high-quality wines and fruits. Use family-oriented marketing visuals. |
| 1 (Low-value single) | Offer discounts, coupons, and loyalty programs to increase engagement. |
| 2 (High-value single) | Promote wines, fruits, and social experiences. Use marketing visuals featuring friends, parties, or solo travel. |
| 3 (Low-value in relationship) | Use family-oriented offers and discounts to encourage purchases. |

---

## Future Opportunities
- Analyze influence of children on purchase behavior.  
- Explore impact of education on buying patterns.  
- Study frequent buyers and their product preferences.  
- Examine sales channels (store, website, etc.) and response to marketing campaigns.  
- Incorporate gender data for more nuanced segmentation.  
- Test alternative clustering algorithms for improved segmentation.

---

## Technologies Used
- Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn)  
- Jupyter Notebook  

---

## Conclusion
This project demonstrates how **customer segmentation** can help businesses tailor marketing strategies for different customer groups. By analyzing income, relationship status, and purchasing behavior, we identified four meaningful customer segments, each with unique characteristics and marketing needs. Implementing targeted campaigns based on these insights can improve customer engagement, sales, and overall marketing efficiency.

