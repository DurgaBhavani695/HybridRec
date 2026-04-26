from src.collaborative import CollaborativeRecommender
import pandas as pd
import os

def test_prediction():
    # Setup test data
    df = pd.DataFrame({'userId': [1, 1, 2], 'movieId': [1, 2, 1], 'rating': [5.0, 4.0, 3.0]})
    recommender = CollaborativeRecommender()
    
    # Ensure models directory exists for test
    os.makedirs("models", exist_ok=True)
    test_model_path = "models/test_model.pkl"
    
    recommender.train(df, save_path=test_model_path)
    score = recommender.predict_score(1, 1)
    
    assert 0.0 <= score <= 5.0
    assert os.path.exists(test_model_path)
    
    # Cleanup
    if os.path.exists(test_model_path):
        os.remove(test_model_path)
