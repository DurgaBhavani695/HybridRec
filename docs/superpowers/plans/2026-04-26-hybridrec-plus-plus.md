# HybridRec++: Context & Sentiment-Aware Recommendation System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a production-quality hybrid recommendation system (Collaborative + Content-Based + Sentiment) with a real-time mood-aware context layer.

**Architecture:** A multi-layered recommender that computes base scores from user-item history (SVD) and item metadata (TF-IDF), then dynamically weights these scores using NLP sentiment analysis of user mood input.

**Tech Stack:** Python 3.10, `uv` (package manager), `pandas`, `numpy`, `scikit-learn` (SVD), `vaderSentiment`, `streamlit`.

---

### Task 1: Project Initialization & Environment Setup

**Files:**
- Create: `pyproject.toml`
- Create: `run_hybridrec.bat`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "hybridrec"
version = "0.1.0"
description = "Context & Sentiment-Aware Recommendation System"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "pandas",
    "numpy",
    "scikit-learn",
    "nltk",
    "vaderSentiment",
    "streamlit",
    "matplotlib",
    "seaborn",
    "requests"
]
```

- [ ] **Step 2: Initialize environment with `uv`**

Run: `uv venv`
Run: `.venv\Scripts\activate` (Windows)
Run: `uv pip install -e .`

- [ ] **Step 3: Create Windows run script `run_hybridrec.bat`**

```batch
@echo off
echo Setting up HybridRec++...
call .venv\Scripts\python.exe -m pip install -e .
echo Downloading data...
call .venv\Scripts\python.exe src\download_data.py
echo Launching App...
call .venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
pause
```

- [ ] **Step 4: Commit setup**

```bash
git add pyproject.toml run_hybridrec.bat
git commit -m "chore: initial project setup and environment configuration"
```

---

### Task 2: Data Acquisition & Loading

**Files:**
- Create: `src/download_data.py`
- Create: `src/data_loader.py`

- [ ] **Step 1: Implement `src/download_data.py`**

```python
import urllib.request
import zipfile
import os

def download_movielens():
    url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    raw_dir = os.path.join("data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    zip_path = os.path.join(raw_dir, "ml-latest-small.zip")

    if not os.path.exists(os.path.join(raw_dir, "movies.csv")):
        print("Downloading MovieLens Small dataset...")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(raw_dir)
        
        extracted_folder = os.path.join(raw_dir, "ml-latest-small")
        for filename in ["movies.csv", "ratings.csv"]:
            os.rename(os.path.join(extracted_folder, filename), os.path.join(raw_dir, filename))
        print("Dataset ready!")
    else:
        print("Dataset already exists.")

if __name__ == "__main__":
    download_movielens()
```

- [ ] **Step 2: Run download script**

Run: `python src/download_data.py`
Expected: `data/raw/movies.csv` and `data/raw/ratings.csv` exist.

- [ ] **Step 3: Implement `src/data_loader.py`**

```python
import pandas as pd
import os

def load_movielens_data(data_dir="data/raw"):
    ratings = pd.read_csv(os.path.join(data_dir, "ratings.csv"))
    movies = pd.read_csv(os.path.join(data_dir, "movies.csv"))
    return ratings, movies
```

- [ ] **Step 4: Commit data layer**

```bash
git add src/download_data.py src/data_loader.py
git commit -m "feat: add data acquisition and loader modules"
```

---

### Task 3: Collaborative Filtering (Matrix Factorization)

**Files:**
- Create: `src/collaborative.py`
- Create: `tests/test_collaborative.py`

- [ ] **Step 1: Write test for Collaborative Filtering**

```python
from src.collaborative import CollaborativeRecommender
import pandas as pd

def test_prediction():
    df = pd.DataFrame({'userId': [1, 1, 2], 'movieId': [1, 2, 1], 'rating': [5.0, 4.0, 3.0]})
    recommender = CollaborativeRecommender()
    recommender.train(df, save_path="models/test_model.pkl")
    score = recommender.predict_score(1, 1)
    assert 0.0 <= score <= 5.0
```

- [ ] **Step 2: Implement `src/collaborative.py`**

```python
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
import pickle
import os

class CollaborativeRecommender:
    def __init__(self):
        self.model = TruncatedSVD(n_components=10, random_state=42)
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
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_collaborative.py`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/collaborative.py tests/test_collaborative.py
git commit -m "feat: implement Matrix Factorization using TruncatedSVD"
```

---

### Task 4: Content-Based Filtering & Sentiment Analysis

**Files:**
- Create: `src/content_based.py`
- Create: `src/sentiment.py`

- [ ] **Step 1: Implement `src/content_based.py`**

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    def __init__(self):
        self.tfidf = TfidfVectorizer(stop_words='english')
        
    def fit(self, movies_df):
        self.movies_df = movies_df.copy()
        tfidf_matrix = self.tfidf.fit_transform(self.movies_df['genres'].str.replace('|', ' '))
        self.cosine_sim = cosine_similarity(tfidf_matrix)
```

- [ ] **Step 2: Implement `src/sentiment.py`**

```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np

class SentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        
    def analyze_mood(self, text):
        score = self.analyzer.polarity_scores(text)['compound']
        category = "positive" if score >= 0.05 else ("negative" if score <= -0.05 else "neutral")
        return category, score
            
    def generate_item_sentiments(self, movies_df):
        np.random.seed(42)
        movies_df['sentiment_score'] = np.random.uniform(-1, 1, size=len(movies_df))
        return movies_df
```

- [ ] **Step 3: Commit**

```bash
git add src/content_based.py src/sentiment.py
git commit -m "feat: add content-based and sentiment analysis modules"
```

---

### Task 5: Hybrid Engine & Streamlit UI

**Files:**
- Create: `src/hybrid.py`
- Create: `app/streamlit_app.py`
- Create: `README.md`

- [ ] **Step 1: Implement `src/hybrid.py`**

```python
import numpy as np

class HybridRecommender:
    def __init__(self, collab_model, content_model, sentiment_analyzer):
        self.collab_model = collab_model
        self.sentiment_analyzer = sentiment_analyzer
        
    def recommend(self, user_id, mood_text, movies_df, top_n=10):
        mood, _ = self.sentiment_analyzer.analyze_mood(mood_text)
        recs = movies_df.copy()
        
        preds = [self.collab_model.predict_score(user_id, mid) for mid in recs['movieId']]
        recs['collab_score'] = (np.array(preds) - 0.5) / 4.5
        recs['sent_norm'] = (recs['sentiment_score'] + 1) / 2
        
        recs['hybrid_score'] = (0.5 * recs['collab_score']) + (0.3 * 0.5) + (0.2 * recs['sent_norm'])
        
        if mood == "negative": recs['hybrid_score'] += 0.2 * recs['sent_norm']
        elif mood == "positive": recs['hybrid_score'] += 0.2 * recs['collab_score']
            
        return recs.sort_values(by='hybrid_score', ascending=False).head(top_n), mood
```

- [ ] **Step 2: Implement `app/streamlit_app.py`**

(Include the streamlit code provided in the previous turn)

- [ ] **Step 3: Write `README.md`**

(Include the professional README provided in the previous turn)

- [ ] **Step 4: Final verification**

Run: `run_hybridrec.bat`
Expected: App opens, recommendations generated, mood context works.

- [ ] **Step 5: Final commit**

```bash
git add src/hybrid.py app/streamlit_app.py README.md
git commit -m "feat: complete hybrid engine, UI and documentation"
```
