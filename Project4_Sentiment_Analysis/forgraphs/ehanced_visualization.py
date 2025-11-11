import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from wordcloud import WordCloud
from textblob import TextBlob
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import squarify
from collections import Counter
import re

class SentimentVisualizer:
    def __init__(self, df):
        self.df = df
        self.setup_plot_style()
    
    def setup_plot_style(self):
        """Set up consistent plotting style"""
        plt.style.use('default')
        sns.set_palette("husl")
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Red, Teal, Blue
        self.sentiment_colors = {-1: '#FF6B6B', 0: '#FFE66D', 1: '#06D6A0'}
        self.sentiment_names = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}
    
    def create_comprehensive_dashboard(self):
        """Create a comprehensive dashboard of all visualizations"""
        fig = plt.figure(figsize=(20, 16))
        
        # Define the grid
        gs = fig.add_gridspec(4, 4)
        
        # 1. Sentiment Distribution (Top Left)
        ax1 = fig.add_subplot(gs[0, 0])
        self.plot_sentiment_distribution(ax1)
        
        # 2. Text Length Analysis (Top Right)
        ax2 = fig.add_subplot(gs[0, 1])
        self.plot_text_length_distribution(ax2)
        
        # 3. Sentiment Proportions (Second Row Left)
        ax3 = fig.add_subplot(gs[1, 0])
        self.plot_sentiment_pie_chart(ax3)
        
        # 4. Word Length vs Sentiment (Second Row Right)
        ax4 = fig.add_subplot(gs[1, 1])
        self.plot_word_length_vs_sentiment(ax4)
        
        # 5. Most Frequent Words (Bottom)
        ax5 = fig.add_subplot(gs[2:, :2])
        self.plot_most_frequent_words(ax5)
        
        # 6. Sentiment Over Time (if date available)
        ax6 = fig.add_subplot(gs[2:, 2:])
        self.plot_sentiment_trends(ax6)
        
        plt.tight_layout()
        plt.show()
    
    def plot_sentiment_distribution(self, ax=None):
        """Plot sentiment distribution with enhanced styling"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        sentiment_counts = self.df['category'].value_counts().sort_index()
        bars = ax.bar([self.sentiment_names[x] for x in sentiment_counts.index], 
                     sentiment_counts.values, 
                     color=[self.sentiment_colors[x] for x in sentiment_counts.index],
                     alpha=0.8,
                     edgecolor='black',
                     linewidth=1.2)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                   f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('Sentiment Distribution', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Sentiment', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Tweets', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_axisbelow(True)
        
        if ax is None:
            plt.tight_layout()
            plt.show()
    
    def plot_sentiment_pie_chart(self, ax=None):
        """Plot pie chart of sentiment proportions"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        sentiment_counts = self.df['category'].value_counts()
        colors = [self.sentiment_colors[x] for x in sentiment_counts.index]
        labels = [self.sentiment_names[x] for x in sentiment_counts.index]
        
        wedges, texts, autotexts = ax.pie(sentiment_counts.values, 
                                         labels=labels, 
                                         colors=colors,
                                         autopct='%1.1f%%',
                                         startangle=90,
                                         shadow=True,
                                         explode=[0.05, 0.05, 0.05])
        
        # Style the text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        
        for text in texts:
            text.set_fontsize(12)
            text.set_fontweight('bold')
        
        ax.set_title('Sentiment Proportions', fontsize=16, fontweight='bold', pad=20)
        
        if ax is None:
            plt.tight_layout()
            plt.show()
    
    def plot_text_length_distribution(self, ax=None):
        """Plot distribution of text lengths by sentiment"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        
        # Calculate text length
        self.df['text_length'] = self.df['clean_text'].str.len()
        
        # Create violin plot
        sentiment_data = [self.df[self.df['category'] == sentiment]['text_length'] 
                         for sentiment in [-1, 0, 1]]
        
        violin_parts = ax.violinplot(sentiment_data, showmeans=True, showmedians=True)
        
        # Customize violin plot colors
        for i, pc in enumerate(violin_parts['bodies']):
            pc.set_facecolor(self.colors[i])
            pc.set_alpha(0.7)
        
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(['Negative', 'Neutral', 'Positive'])
        ax.set_title('Text Length Distribution by Sentiment', fontsize=16, fontweight='bold')
        ax.set_xlabel('Sentiment', fontsize=12, fontweight='bold')
        ax.set_ylabel('Text Length (characters)', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add mean and median labels
        for i, data in enumerate(sentiment_data):
            mean_val = np.mean(data)
            median_val = np.median(data)
            ax.text(i+1, mean_val + 10, f'Mean: {mean_val:.1f}', 
                   ha='center', va='bottom', fontweight='bold')
            ax.text(i+1, median_val - 15, f'Median: {median_val:.1f}', 
                   ha='center', va='top', fontweight='bold')
        
        if ax is None:
            plt.tight_layout()
            plt.show()
    
    def plot_word_length_vs_sentiment(self, ax=None):
        """Plot average word length by sentiment"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate average word length
        def avg_word_length(text):
            words = str(text).split()
            return np.mean([len(word) for word in words]) if words else 0
        
        self.df['avg_word_length'] = self.df['clean_text'].apply(avg_word_length)
        
        avg_length_by_sentiment = self.df.groupby('category')['avg_word_length'].mean()
        
        bars = ax.bar([self.sentiment_names[x] for x in avg_length_by_sentiment.index],
                     avg_length_by_sentiment.values,
                     color=[self.sentiment_colors[x] for x in avg_length_by_sentiment.index],
                     alpha=0.8,
                     edgecolor='black')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('Average Word Length by Sentiment', fontsize=16, fontweight='bold')
        ax.set_xlabel('Sentiment', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Word Length', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        if ax is None:
            plt.tight_layout()
            plt.show()
    
    def plot_most_frequent_words(self, ax=None, top_n=20):
        """Plot most frequent words by sentiment"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(15, 10))
        
        # Get stopwords
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))
        
        # Create subplots for each sentiment
        sentiments = [-1, 0, 1]
        
        for i, sentiment in enumerate(sentiments):
            # Filter data by sentiment
            sentiment_text = ' '.join(self.df[self.df['category'] == sentiment]['clean_text'])
            
            # Tokenize and count words
            words = re.findall(r'\b\w+\b', sentiment_text.lower())
            words = [word for word in words if word not in stop_words and len(word) > 2]
            
            word_freq = Counter(words)
            common_words = word_freq.most_common(top_n)
            
            # Create horizontal bar plot
            words, counts = zip(*common_words)
            y_pos = np.arange(len(words))
            
            ax.barh(y_pos + i*(len(words) + 2), counts, 
                   color=self.sentiment_colors[sentiment], 
                   alpha=0.7,
                   label=self.sentiment_names[sentiment])
            
            # Add word labels
            for j, (word, count) in enumerate(common_words):
                ax.text(count + 5, j + i*(len(words) + 2), 
                       f'{word} ({count})', va='center', fontsize=9)
        
        ax.set_yticks([])
        ax.set_title(f'Top {top_n} Most Frequent Words by Sentiment', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Frequency', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
        
        if ax is None:
            plt.tight_layout()
            plt.show()
    
    def plot_sentiment_trends(self, ax=None):
        """Plot sentiment trends (placeholder for time-based analysis)"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        
        # If we had date information, we could plot trends over time
        # For now, create a simulated trend
        sentiment_over_time = self.df['category'].value_counts().sort_index()
        
        ax.plot([self.sentiment_names[x] for x in sentiment_over_time.index],
               sentiment_over_time.values, 
               marker='o', 
               linewidth=3, 
               markersize=8,
               color='#FF6B6B')
        
        ax.fill_between([self.sentiment_names[x] for x in sentiment_over_time.index],
                       sentiment_over_time.values, 
                       alpha=0.3, 
                       color='#FF6B6B')
        
        ax.set_title('Sentiment Trends', fontsize=16, fontweight='bold')
        ax.set_xlabel('Sentiment', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        if ax is None:
            plt.tight_layout()
            plt.show()
    
    def create_wordclouds_grid(self):
        """Create a grid of word clouds for each sentiment"""
        fig, axes = plt.subplots(1, 3, figsize=(20, 8))
        
        for i, sentiment in enumerate([-1, 0, 1]):
            # Combine all text for the sentiment
            text = ' '.join(self.df[self.df['category'] == sentiment]['clean_text'])
            
            # Create word cloud
            wordcloud = WordCloud(width=400, 
                                height=300,
                                background_color='white',
                                colormap='Reds' if sentiment == -1 else 'Blues' if sentiment == 0 else 'Greens',
                                max_words=100,
                                contour_width=1,
                                contour_color='black').generate(text)
            
            axes[i].imshow(wordcloud, interpolation='bilinear')
            axes[i].set_title(f'{self.sentiment_names[sentiment]} Sentiment Word Cloud', 
                            fontsize=14, fontweight='bold')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def plot_sentiment_heatmap(self, vectorizer, model, top_features=30):
        """Plot heatmap of feature importance by sentiment"""
        if hasattr(model.named_steps['classifier'], 'coef_'):
            feature_importance = model.named_steps['classifier'].coef_
            feature_names = vectorizer.get_feature_names_out()
            
            # Get top features for each class
            fig, axes = plt.subplots(1, 3, figsize=(20, 8))
            
            for i, sentiment in enumerate([-1, 0, 1]):
                # Get top features for this sentiment
                indices = np.argsort(feature_importance[i])[-top_features:]
                top_words = [feature_names[j] for j in indices]
                top_scores = feature_importance[i][indices]
                
                # Create heatmap data
                heatmap_data = pd.DataFrame({
                    'Word': top_words,
                    'Importance': top_scores
                }).sort_values('Importance')
                
                axes[i].barh(range(len(heatmap_data)), heatmap_data['Importance'],
                           color=self.sentiment_colors[sentiment])
                axes[i].set_yticks(range(len(heatmap_data)))
                axes[i].set_yticklabels(heatmap_data['Word'])
                axes[i].set_title(f'Top Features - {self.sentiment_names[sentiment]}', 
                                fontsize=14, fontweight='bold')
                axes[i].set_xlabel('Feature Importance')
            
            plt.tight_layout()
            plt.show()