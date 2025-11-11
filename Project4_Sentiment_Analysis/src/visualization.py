import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import pandas as pd

def create_interactive_sentiment_plot(df):
    """
    Create interactive sentiment visualization
    """
    sentiment_counts = df['category'].value_counts().reset_index()
    sentiment_counts.columns = ['sentiment', 'count']
    sentiment_counts['sentiment'] = sentiment_counts['sentiment'].map({-1: 'Negative', 0: 'Neutral', 1: 'Positive'})
    
    fig = px.pie(sentiment_counts, values='count', names='sentiment', 
                 title='Sentiment Distribution',
                 color='sentiment',
                 color_discrete_map={'Negative': 'red', 'Neutral': 'blue', 'Positive': 'green'})
    
    fig.show()

def plot_sentiment_over_time(df, date_column=None):
    """
    Plot sentiment trends over time (if date information available)
    """
    if date_column and date_column in df.columns:
        df['date'] = pd.to_datetime(df[date_column])
        df['month'] = df['date'].dt.to_period('M')
        
        monthly_sentiment = df.groupby(['month', 'category']).size().unstack(fill_value=0)
        monthly_sentiment.plot(kind='line', figsize=(12, 6))
        plt.title('Sentiment Trends Over Time')
        plt.xlabel('Month')
        plt.ylabel('Number of Tweets')
        plt.legend(['Negative', 'Neutral', 'Positive'])
        plt.show()