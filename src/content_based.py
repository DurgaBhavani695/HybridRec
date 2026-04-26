from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    def __init__(self):
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.movies_df = None
        self.cosine_sim = None
        
    def fit(self, movies_df):
        self.movies_df = movies_df.copy()
        tfidf_matrix = self.tfidf.fit_transform(self.movies_df['genres'].str.replace('|', ' '))
        self.cosine_sim = cosine_similarity(tfidf_matrix)
