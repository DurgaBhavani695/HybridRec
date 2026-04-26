import pandas as pd
import os

def load_movielens_data(data_dir="data/raw"):
    ratings = pd.read_csv(os.path.join(data_dir, "ratings.csv"))
    movies = pd.read_csv(os.path.join(data_dir, "movies.csv"))
    return ratings, movies
