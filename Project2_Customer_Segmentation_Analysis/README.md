# Marketing Analytics: Customer Segmentation

## Project Idea
Customer Segmentation Analysis

---

## Project Overview
Understanding customer behavior is critical for effective marketing. Customer segmentation involves grouping customers based on shared attributes, purchase habits, or demographic information. By identifying distinct segments, companies can tailor their marketing campaigns to improve engagement, sales, and customer satisfaction.  

This project performs **customer segmentation analysis** using clustering techniques to identify meaningful customer groups and provide actionable marketing insights.

---

## Project Structure
<img width="582" height="221" alt="image" src="https://github.com/user-attachments/assets/120f6bb0-43e7-4ded-9e9a-95e4afb29a3d" />

---

## Project Goals
- Clean and prepare customer data for analysis.  
- Explore the dataset to identify patterns and trends.  
- Engineer features to enhance segmentation accuracy.  
- Apply clustering algorithms to detect distinct customer groups.  
- Interpret results and suggest targeted marketing strategies.

---

## Dataset
- **Number of entries:** 2,300+ customers  
- **Number of columns:** 39+ features  
- **Key features:** Income, purchase amounts, relationship status, demographic information, product preferences  
- **Notes:** Some column names differ from original descriptions; assumptions were made based on feature names.

---

## Methodology

### 1. Data Cleaning and Preparation
- Removed missing and inconsistent values.  
- Selected relevant features for clustering analysis.  
- Encoded categorical variables for modeling.

### 2. Exploratory Analysis
- Visualized distributions of key features like income, total purchases, and relationship status.  
- Detected outliers and trends influencing customer behavior.

### 3. Feature Engineering
- Created aggregate features like `Total_Spend` and `Purchase_Frequency` to improve segmentation.  
- Standardized numeric features for clustering stability.

### 4. Clustering with K-Means
- Determined the optimal number of clusters using the **Elbow Method** and **Silhouette Score**.  
- Applied K-Means to group customers into meaningful segments.

### 5. Cluster Analysis
- Examined cluster profiles to understand the characteristics of each group.  
- Focused on income, relationship status, and total purchase behavior.

---

## Results

- **Optimal clusters:** 4  
- **Cluster Summary:**
| Cluster | Description | Approx. % of Customers |
|---------|-------------|------------------------|
| 0       | High-value couples | 25% |
| 1       | Budget-conscious singles | 20% |
| 2       | High-income singles | 18% |
| 3       | Average spenders in relationships | 37% |

- Cluster 0: Customers with higher income and in relationships, showing strong spending on premium products.  
- Cluster 1: Singles with lower income, making smaller, occasional purchases.  
- Cluster 2: High-income singles who tend to spend significantly on select product categories.  
- Cluster 3: Customers in relationships with moderate income and spending patterns.

---

## Marketing Recommendations

| Cluster | Recommended Strategy |
|---------|--------------------|
| 0 (High-value couples) | Targeted campaigns promoting premium products and family-oriented offers. |
| 1 (Budget-conscious singles) | Offer discounts, loyalty programs, and occasional promotions to boost engagement. |
| 2 (High-income singles) | Focus on premium experiences and products; use social or lifestyle-centric promotions. |
| 3 (Average spenders in relationships) | Provide bundle deals and seasonal promotions to encourage repeat purchases. |

---

## Future Enhancements
- Analyze impact of children and household size on purchase behavior.  
- Study influence of education level on spending patterns.  
- Investigate most active channels (online, in-store) for each segment.  
- Examine response to previous marketing campaigns for improved targeting.  
- Test alternative clustering algorithms (e.g., Hierarchical, DBSCAN) for comparison.  

---

## Tools and Technologies
- **Programming Language:** Python  
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn  
- **Environment:** Jupyter Notebook  

---

## Conclusion
This analysis highlights the importance of customer segmentation in marketing strategy. By identifying four distinct customer segments, companies can deliver tailored campaigns that maximize engagement and sales. The insights generated from this project provide actionable directions for marketing, product promotion, and customer relationship management.


