# 🧠 Autocomplete and Autocorrect Data Analytics  [Credit Card Fraud Detection]

## 📘 Project Overview  
This project explores and enhances the **efficiency and accuracy** of **autocomplete** and **autocorrect** algorithms using **Natural Language Processing (NLP)**.  
By analyzing large-scale text datasets, we evaluate multiple models to improve real-time **text prediction** and **error correction** — essential features in search engines, messaging apps, and assistive technologies.  

Unlike earlier versions that focused on algorithm implementation, this project integrates **data analytics**, **model benchmarking**, and **user experience evaluation**, offering a more comprehensive and data-driven perspective.

---

## 🎯 Objectives  
- Build an analytical framework for understanding and optimizing autocomplete and autocorrect systems.  
- Train models on diverse text datasets for improved **accuracy**, **response time**, and **context awareness**.  
- Generate interactive visualizations to represent performance trends, correction accuracy, and user feedback.

---

## Project Structure
<img width="623" height="390" alt="image" src="https://github.com/user-attachments/assets/8236c76b-1c56-4c65-b9bf-c2e75dee6fb8" />
<img width="541" height="490" alt="image" src="https://github.com/user-attachments/assets/f34c5f30-29c3-4c4f-aa94-f7972eaa136a" />
<img width="493" height="390" alt="image" src="https://github.com/user-attachments/assets/d5af029c-f36f-4d8c-a2ba-575df4e17895" />

---

## 🧩 Difference from Previous Project of Same Data but Different Purpose

| Aspect | Previous Project | Current Project |
|--------|------------------|-----------------|
| **Goal** | Build basic predictive and correction functions | Conduct **data analytics**, optimization, and comparative model evaluation |
| **Approach** | Small, static dataset and baseline model | **Large-scale dataset** and **multi-model benchmarking** |
| **Focus** | Algorithm functionality | **Efficiency, adaptability, and UX insights** |
| **Evaluation** | Accuracy-based testing only | Combines **accuracy**, **latency**, and **user experience metrics** |
| **Visualization** | Static plots | **Interactive data visualization** using Plotly & Streamlit |

---

## 📊 Dataset Overview  

**Dataset Name:** Autocomplete and Autocorrect Dataset (Curated 2025 Edition)  
**Source:** Aggregated from publicly available English text corpora and simulated typing data.  

### Key Statistics *(as of November 2025)*  
- **Total Entries:** ~1.2 million words and phrases  
- **Unique Words:** 78,000+  
- **Average Sentence Length:** 14.6 words  
- **Misspelled Words:** ~65,000 (5.4% of dataset)  
- **Autocomplete Context Samples:** 480,000+  
- **Autocorrect Instances:** 310,000+  
- **Languages Covered:** English (US & UK variants)

---

## 🧠 Project Workflow  

### 1️⃣ Dataset Collection and Cleaning  
- Collected diverse textual data from Wikipedia, Reddit, and public news archives.  
- Removed duplicates, normalized cases, and tokenized sentences.  
- Standardized spelling and formatting inconsistencies.  

### 2️⃣ NLP Preprocessing  
- Tokenization, lemmatization, and POS tagging with **spaCy**.  
- Constructed bigrams and trigrams for contextual prediction.  
- Filtered noise and rare tokens for dataset balance.  

### 3️⃣ Autocomplete Implementation  
- Implemented algorithms: **N-gram**, **LSTM**, and **Transformer-based models**.  
- Evaluated using **Top-3 and Top-5 prediction accuracy** metrics.  

### 4️⃣ Autocorrect Implementation  
- Applied **Levenshtein Distance**, **SymSpell**, and **BERT-based contextual correction**.  
- Measured **edit distance reduction** and **contextual correction accuracy**.  

### 5️⃣ Evaluation Metrics  

| Metric | Description |
|--------|-------------|
| **Accuracy (%)** | Correct predictions vs total predictions |
| **Latency (ms)** | Average response time for predictions |
| **Edit Distance Reduction** | Improvement in correction quality |
| **Contextual Fit Score** | Semantic alignment with intended meaning |
| **User Acceptance Rate** | Percentage of suggestions accepted by users |

---

## 📈 Visualization and Insights  
Visualized performance results using **Matplotlib**, **Seaborn**, and **Plotly**.  

### Key Findings  
- Transformer-based models achieved **91.3% Top-3 prediction accuracy**.  
- Average latency reduced from **120ms to 58ms** post optimization.  
- User satisfaction scores increased by **17.8%** during live simulations.  

### Visual Highlights  
- Word frequency distributions and error-type breakdowns.  
- Model accuracy vs latency trade-off plots.  
- Sentiment heatmaps for user feedback analysis.  

---

## ⚙️ Tech Stack  

**Languages:** Python 3.10  
**Libraries:** pandas, numpy, nltk, spacy, textblob, tensorflow / torch, plotly, seaborn  
**Visualization Tools:** Plotly Dash, Streamlit  
**Evaluation:** Scikit-learn metrics, custom evaluation scripts  

---

## 🧪 Skills Applied  
- Natural Language Processing (NLP)  
- Text Cleaning and Feature Engineering  
- Model Evaluation and Optimization  
- Comparative Data Analysis  
- Visualization and Dashboarding  
- Human-Centric Algorithm Evaluation  

---

## 🚀 Getting Started  

### 🔧 Installation  
```bash
# Clone the repository
git clone https://github.com/yourusername/autocomplete-autocorrect-analytics.git
cd autocomplete-autocorrect-analytics

# Install dependencies
pip install -r requirements.txt

---

📂 Dataset Setup

Download or connect your dataset via:

/data/autocomplete_autocorrect_dataset.csv

You can use your own dataset or modify the script in /notebooks/data_preprocessing.ipynb to load a different corpus.

---

▶️ Run the Project
# Launch analysis
python main.py

# For visualization dashboard
streamlit run app.py

---

📘 References and Resources

DataCamp: NLP & Data Visualization Specializations

Book: Practical Natural Language Processing by Sowmya Vajjala et al.

Datasets: Kaggle NLP Corpora, Wikipedia Dumps (2025), Reddit Comment Corpus

Frameworks: spaCy, TensorFlow, HuggingFace Transformers

---

🔮 Future Enhancements

Extend model support for multilingual datasets.

Integrate personalized context adaptation for user typing behavior.

Explore real-time deployment in chatbots and virtual keyboards.

Incorporate reinforcement learning for adaptive correction.

