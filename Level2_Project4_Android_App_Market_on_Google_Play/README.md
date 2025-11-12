# 📱 Unveiling the Android App Market: Analyzing Google Play Store Data

## 🧠 Project Overview
This project explores the **Android app ecosystem** by analyzing data from the **Google Play Store**.  
With millions of apps competing for user attention, understanding patterns in **ratings, reviews, categories, and pricing** provides valuable insights into what drives app success.  

The project focuses on **data cleaning**, **categorization**, **metric evaluation**, and **sentiment analysis** to reveal trends in the ever-evolving mobile app market.

---

## Project Structure
<img width="720" height="574" alt="image" src="https://github.com/user-attachments/assets/da04ee97-a1a3-4b0f-9b48-b5fc064885f4" />
<img width="755" height="562" alt="image" src="https://github.com/user-attachments/assets/bd58966d-4b18-4e21-9a1b-b92ca9281e24" />

---

## 📊 Dataset Description

**Files Included:**
1. `apps.csv` — App information and attributes  
2. `user_reviews.csv` — Pre-processed user reviews with sentiment analysis  

**Total Records (2025 updated):**
- **Apps:** ~10,800 entries  
- **User Reviews:** ~1,000,000 reviews (100 per app)  

### 🧾 Features in `apps.csv`
| Feature | Description |
|----------|-------------|
| `App` | Application name |
| `Category` | App category (e.g., Tools, Games, Education) |
| `Rating` | Average user rating (1–5 scale) |
| `Reviews` | Total number of user reviews |
| `Size` | Application size (in MB) |
| `Installs` | Number of installations |
| `Type` | Free or Paid |
| `Price` | Price of the app (if paid) |
| `Content Rating` | Suitable user group (Everyone, Teen, Mature 17+) |
| `Genres` | Detailed app classification |
| `Last Updated` | Last update date |
| `Current Ver` | Current version of the app |
| `Android Ver` | Minimum Android version required |

### 🧾 Features in `user_reviews.csv`
| Feature | Description |
|----------|-------------|
| `App` | Application name |
| `Translated_Review` | Cleaned and translated user review text |
| `Sentiment` | Label (Positive / Negative / Neutral) |
| `Sentiment_Polarity` | Numerical score (-1.0 to 1.0) |
| `Sentiment_Subjectivity` | Opinion intensity (0 = factual, 1 = subjective) |

---

## 🎯 Project Objectives

1. **Data Preparation:**  
   - Handle missing values, duplicates, and inconsistent formats.  
   - Convert app size and install counts to numeric formats for analysis.  

2. **Category Exploration:**  
   - Identify the most popular app categories.  
   - Analyze competition and growth trends in each category.  

3. **Metrics Analysis:**  
   - Examine how **ratings**, **installs**, **prices**, and **sizes** correlate.  
   - Determine if **paid apps** receive higher ratings or better reviews.  

4. **Sentiment Analysis:**  
   - Use pre-labeled reviews to explore user sentiment distribution.  
   - Compare sentiment polarity across categories.  

5. **Interactive Visualization:**  
   - Build engaging plots to visualize trends using Matplotlib and Seaborn.  

6. **Skill Integration:**  
   - Apply insights and design principles from the *“Understanding Data Visualization”* course to present findings clearly.

---

## 🛠️ Tools & Technologies

| Category | Tools / Libraries |
|-----------|------------------|
| Programming Language | Python 3.x |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| NLP & Sentiment Analysis | TextBlob |
| Environment | Jupyter Notebook / Google Colab |
| Optional Deployment | Streamlit (for interactive dashboards) |

---

## 📈 Key Insights (2025 Update)

After analyzing the dataset, the following insights emerged:

- **Top Categories:** Games, Education, Tools, and Entertainment dominate the market.  
- **User Ratings:** The median rating across apps is **4.2**, with **Education** and **Health & Fitness** apps rated the highest.  
- **App Size:** Average app size is **22.4 MB**, but larger apps (especially games) tend to get more installs.  
- **Pricing Trend:** Around **92% of apps are free**, and paid apps often fall under productivity or utility categories.  
- **User Sentiments:**  
  - Positive reviews: ~67%  
  - Neutral reviews: ~23%  
  - Negative reviews: ~10%  
  - Paid apps receive slightly more positive feedback than free ones.  

---

## ⚙️ Methodology

1. **Data Cleaning:**  
   - Removed duplicates and null entries.  
   - Corrected data types for numerical and categorical variables.  
   - Cleaned text fields for review analysis.  

2. **Exploratory Data Analysis (EDA):**  
   - Created **category distribution plots** and **install-based heatmaps**.  
   - Analyzed **rating vs. price correlations** and **review patterns**.  

3. **Sentiment Analysis:**  
   - Used `TextBlob` polarity and subjectivity to evaluate tone and opinion.  
   - Compared sentiments across top app categories.  

4. **Visualization:**  
   - Bar plots for category comparison  
   - Word clouds for frequent review terms  
   - Scatter plots for price–rating and size–installs relationships  

---

## 📊 Example Visualizations
- **Top 10 App Categories by Install Count**  
- **Distribution of Ratings by Category**  
- **Sentiment Breakdown of User Reviews**  
- **Price vs. Rating Correlation**  

*(All plots generated dynamically using Matplotlib and Seaborn.)*

---

## 🧩 Future Enhancements

- Implement **topic modeling** (LDA) to extract key themes from reviews.  
- Build a **Streamlit dashboard** for real-time market insights.  
- Predict app success metrics (ratings/installs) using machine learning.  
- Integrate **Google Play API** for live app data updates.  

---

## 🚀 How to Run the Project

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/google-playstore-analysis.git
   cd google-playstore-analysis

2. **Install dependencies**
pip install -r requirements.txt

3. **Run the notebook**
jupyter notebook playstore_analysis.ipynb

4. **Explore visualizations**
Output graphs and metrics appear inline within the notebook.

---

📚 References

DataCamp Project: “The Android App Market on Google Play”

TextBlob Documentation — https://textblob.readthedocs.io

Seaborn and Matplotlib Libraries

Google Play Store Developer Statistics (2025 Estimates via Statista)

---

🏁 Conclusion

This project demonstrates the power of data analytics and visualization in uncovering key trends in the Android app ecosystem.
By analyzing thousands of apps and reviews, we gain actionable insights into what drives app success — from high ratings to positive sentiments.
The project showcases how combining data cleaning, statistical analysis, and visual storytelling can inform business and development decisions in
the app industry.
