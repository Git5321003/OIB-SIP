import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

class InteractiveVisualizations:
    def __init__(self, df):
        self.df = df
    
    def create_interactive_sentiment_dashboard(self):
        """Create an interactive dashboard with Plotly"""
        # Sentiment distribution
        sentiment_counts = self.df['category'].value_counts().reset_index()
        sentiment_counts.columns = ['sentiment', 'count']
        sentiment_counts['sentiment'] = sentiment_counts['sentiment'].map({-1: 'Negative', 0: 'Neutral', 1: 'Positive'})
        
        # Text length analysis
        self.df['text_length'] = self.df['clean_text'].str.len()
        avg_length_by_sentiment = self.df.groupby('category')['text_length'].mean().reset_index()
        avg_length_by_sentiment['sentiment'] = avg_length_by_sentiment['category'].map({-1: 'Negative', 0: 'Neutral', 1: 'Positive'})
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Sentiment Distribution', 'Text Length by Sentiment',
                          'Sentiment Proportions', 'Sentiment Analysis Dashboard'),
            specs=[[{"type": "bar"}, {"type": "box"}],
                   [{"type": "pie"}, {"type": "scatter"}]]
        )
        
        # Bar chart - Sentiment distribution
        fig.add_trace(
            go.Bar(x=sentiment_counts['sentiment'], y=sentiment_counts['count'],
                  marker_color=['#FF6B6B', '#FFE66D', '#06D6A0']),
            row=1, col=1
        )
        
        # Box plot - Text length by sentiment
        for sentiment in [-1, 0, 1]:
            sentiment_name = 'Negative' if sentiment == -1 else 'Neutral' if sentiment == 0 else 'Positive'
            color = '#FF6B6B' if sentiment == -1 else '#FFE66D' if sentiment == 0 else '#06D6A0'
            
            fig.add_trace(
                go.Box(y=self.df[self.df['category'] == sentiment]['text_length'],
                      name=sentiment_name, marker_color=color),
                row=1, col=2
            )
        
        # Pie chart - Sentiment proportions
        fig.add_trace(
            go.Pie(labels=sentiment_counts['sentiment'], values=sentiment_counts['count'],
                  marker_colors=['#FF6B6B', '#FFE66D', '#06D6A0']),
            row=2, col=1
        )
        
        # Scatter plot (placeholder for additional analysis)
        fig.add_trace(
            go.Scatter(x=avg_length_by_sentiment['sentiment'], 
                      y=avg_length_by_sentiment['text_length'],
                      mode='lines+markers', name='Avg Text Length'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=True, 
                         title_text="Twitter Sentiment Analysis Dashboard")
        fig.show()
    
    def create_interactive_word_cloud(self, max_words=100):
        """Create interactive word frequency visualization"""
        from collections import Counter
        import re
        
        # Combine all text
        all_text = ' '.join(self.df['clean_text'])
        words = re.findall(r'\b\w+\b', all_text.lower())
        
        # Remove stopwords
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Count word frequencies
        word_freq = Counter(words)
        common_words = word_freq.most_common(max_words)
        
        # Create dataframe for plotly
        word_df = pd.DataFrame(common_words, columns=['word', 'frequency'])
        
        # Create interactive bar chart
        fig = px.bar(word_df.head(30), x='frequency', y='word', 
                    orientation='h',
                    title=f'Top 30 Most Frequent Words',
                    color='frequency',
                    color_continuous_scale='viridis')
        
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        fig.show()
        
        return word_df
    
    def create_sentiment_timeline(self, date_column=None):
        """Create interactive sentiment timeline (if date data available)"""
        if date_column and date_column in self.df.columns:
            # If we have date information, create timeline
            self.df['date'] = pd.to_datetime(self.df[date_column])
            self.df['date_group'] = self.df['date'].dt.to_period('M').astype(str)
            
            timeline_data = self.df.groupby(['date_group', 'category']).size().unstack(fill_value=0)
            timeline_data = timeline_data.reset_index()
            timeline_data.columns = ['date', 'Negative', 'Neutral', 'Positive']
            
            fig = px.line(timeline_data, x='date', y=['Negative', 'Neutral', 'Positive'],
                         title='Sentiment Trends Over Time',
                         labels={'value': 'Number of Tweets', 'variable': 'Sentiment'})
            
            fig.show()
        else:
            print("No date column available for timeline analysis")