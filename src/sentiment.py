from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np

class SentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        
    def analyze_mood(self, text):
        score = self.analyzer.polarity_scores(text)['compound']
        # Adjusted thresholds: VADER is biased towards positive. 
        # Using a higher threshold for positive, and allowing a broader range for neutral.
        if score > 0.2:
            category = "positive"
        elif score < -0.1:
            category = "negative"
        else:
            category = "neutral"
        return category, score
            
    def generate_item_sentiments(self, movies_df):
        np.random.seed(42)
        movies_df['sentiment_score'] = np.random.uniform(-1, 1, size=len(movies_df))
        return movies_df
