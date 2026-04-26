import os
import sys
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

try:
    from data_loader import load_movielens_data
    from collaborative import CollaborativeRecommender
    from content_based import ContentBasedRecommender
    from sentiment import SentimentAnalyzer
    from hybrid import HybridRecommender
    print("✅ All imports successful!")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def verify():
    print("🚀 Starting verification...")
    
    # 1. Load Data
    if not os.path.exists("data/raw/movies.csv"):
        print("📥 Downloading data...")
        from download_data import download_movielens
        download_movielens()
        
    ratings, movies = load_movielens_data()
    print(f"✅ Data loaded: {len(ratings)} ratings, {len(movies)} movies")
    
    # 2. Collaborative Model
    collab = CollaborativeRecommender()
    print("🧠 Training Collaborative model (subset for speed)...")
    collab.train(ratings.head(10000))
    print("✅ Collaborative model trained.")
    
    # 3. Content Model
    content = ContentBasedRecommender()
    content.fit(movies)
    print("✅ Content model fitted.")
    
    # 4. Sentiment Analyzer
    sentiment = SentimentAnalyzer()
    movies = sentiment.generate_item_sentiments(movies)
    print("✅ Item sentiments generated.")
    
    # 5. Hybrid Engine
    hybrid = HybridRecommender(collab, content, sentiment)
    print("🛠️ Hybrid engine initialized.")
    
    # 6. Test Recommendation
    recs, mood = hybrid.recommend(user_id=1, mood_text="I feel happy", movies_df=movies.head(100))
    print(f"✅ Recommendations generated! Mood detected: {mood}")
    print(f"Top recommendation: {recs.iloc[0]['title']} with score {recs.iloc[0]['hybrid_score']:.4f}")
    
    print("✨ VERIFICATION COMPLETE!")

if __name__ == "__main__":
    verify()
