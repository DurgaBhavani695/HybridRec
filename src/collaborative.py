import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
import pickle
import os

class CollaborativeRecommender:
    def __init__(self):
        self.model = TruncatedSVD(n_components=2, random_state=42) # n_components=2 for the small test df
        self.user_features = None
        self.item_features = None
        self.user_map = {}
        self.movie_map = {}
        self.mean_rating = 0
        
    def train(self, ratings_df, save_path="models/mf_model.pkl"):
        self.mean_rating = ratings_df['rating'].mean()
        pivot_df = ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
        
        self.user_map = {id: i for i, id in enumerate(pivot_df.index)}
        self.movie_map = {id: i for i, id in enumerate(pivot_df.columns)}
        
        matrix = pivot_df.values
        # Adjust n_components if matrix is too small
        n_features = min(matrix.shape[0], matrix.shape[1]) - 1
        if n_features < self.model.n_components:
            self.model.n_components = max(1, n_features)
            
        self.user_features = self.model.fit_transform(matrix)
        self.item_features = self.model.components_.T
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)
            
    def load_model(self, load_path="models/mf_model.pkl"):
        with open(load_path, 'rb') as f:
            loaded = pickle.load(f)
            self.__dict__.update(loaded.__dict__)
            
    def predict_score(self, user_id, movie_id):
        if user_id not in self.user_map or movie_id not in self.movie_map:
            return self.mean_rating
            
        u_idx = self.user_map[user_id]
        m_idx = self.movie_map[movie_id]
        
        score = np.dot(self.user_features[u_idx], self.item_features[m_idx])
        return np.clip(score, 0.5, 5.0)
