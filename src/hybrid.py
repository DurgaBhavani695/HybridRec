import numpy as np

class HybridRecommender:
    def __init__(self, collab_model, content_model, sentiment_analyzer):
        self.collab_model = collab_model
        self.content_model = content_model
        self.sentiment_analyzer = sentiment_analyzer
        
    def recommend(self, user_id, mood_text, movies_df, top_n=10):
        mood, mood_score = self.sentiment_analyzer.analyze_mood(mood_text)
        recs = movies_df.copy()
        
        # 1. Collaborative Score
        preds = [self.collab_model.predict_score(user_id, mid) for mid in recs['movieId']]
        recs['collab_score'] = (np.array(preds) - 0.5) / 4.5
        
        # 2. Content Score
        recs['content_score'] = 0.5 
        
        # 3. Item Sentiment
        recs['sent_norm'] = (recs['sentiment_score'] + 1) / 2
        
        # Base Hybrid Score
        recs['hybrid_score'] = (0.5 * recs['collab_score']) + (0.3 * recs['content_score']) + (0.2 * recs['sent_norm'])
        
        # Dynamic Context-Aware Adjustment using Intensity (0.0 to 1.0)
        intensity = abs(mood_score)
        
        if mood_score < -0.05: # Negative Sentiment
            # Boost feel-good (high sentiment) items proportional to sadness intensity
            recs['hybrid_score'] += intensity * recs['sent_norm'] * 0.5
        elif mood_score > 0.05: # Positive Sentiment
            # Boost generally high-rated (high collab) items proportional to joy intensity
            recs['hybrid_score'] += intensity * recs['collab_score'] * 0.5
            
        return recs.sort_values(by='hybrid_score', ascending=False).head(top_n), mood
