import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Add src to path to allow imports from src directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_movielens_data
from src.collaborative import CollaborativeRecommender
from src.content_based import ContentBasedRecommender
from src.sentiment import SentimentAnalyzer
from src.hybrid import HybridRecommender

# Page Configuration
st.set_page_config(
    page_title="HybridRec++ | Intelligent Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stProgress > div > div > div > div {
        background-color: #ef4444;
    }
    .movie-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🎬 HybridRec++")
st.markdown("#### *Context-Aware & Sentiment-Driven Recommendation Engine*")
st.divider()

@st.cache_resource
def initialize_system():
    """Initializes the data and models for the recommendation system."""
    # 1. Load Data
    try:
        ratings, movies = load_movielens_data()
    except Exception as e:
        st.error(f"Error loading data: {e}. Please ensure data is downloaded.")
        st.stop()
    
    # 2. Collaborative Model
    collab = CollaborativeRecommender()
    model_dir = "models"
    model_path = os.path.join(model_dir, "mf_model.pkl")
    
    if not os.path.exists(model_path):
        with st.spinner("Training Collaborative Model for the first time..."):
            collab.train(ratings, save_path=model_path)
    else:
        collab.load_model(model_path)
        
    # 3. Content Model
    content = ContentBasedRecommender()
    content.fit(movies)
    
    # 4. Sentiment Analyzer
    sentiment = SentimentAnalyzer()
    movies = sentiment.generate_item_sentiments(movies)
    
    # 5. Hybrid Engine
    hybrid = HybridRecommender(collab, content, sentiment)
    
    return hybrid, movies, ratings

# Initialize the system
hybrid_engine, movies_df, ratings_df = initialize_system()

# Sidebar Configuration
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/movie-projector.png", width=100)
    st.header("User Selection")
    
    user_ids = sorted(ratings_df['userId'].unique())
    selected_user = st.selectbox("Select User Profile", user_ids, index=0)
    
    st.divider()
    
    st.header("Mood Context")
    mood_text = st.text_area(
        "How are you feeling today?", 
        "I'm feeling a bit tired and want something relaxing and lighthearted.",
        help="The system uses NLP to analyze your mood and adjust recommendations."
    )
    
    top_n = st.slider("Number of results", 5, 20, 10)
    
    get_recs = st.button("Generate Recommendations", type="primary", use_container_width=True)

# Main Content Area
if get_recs:
    with st.spinner("Running Hybrid Intelligence..."):
        recs, mood = hybrid_engine.recommend(selected_user, mood_text, movies_df, top_n=top_n)
        
        # Display Mood Analysis
        mood_color = "green" if mood == "positive" else ("red" if mood == "negative" else "gray")
        st.subheader("Analysis Results")
        
        c1, c2 = st.columns(2)
        with c1:
            st.info(f"User ID: **{selected_user}**")
        with c2:
            st.success(f"Detected Mood: **{mood.upper()}**")

        st.markdown("---")
        st.subheader(f"Top {top_n} Recommendations for You")
        
        # Display recommendations
        # Calculate min and max for normalization
        min_score = recs['hybrid_score'].min()
        max_score = recs['hybrid_score'].max()

        for i, (idx, row) in enumerate(recs.iterrows()):
            with st.container():
                col_rank, col_details, col_score = st.columns([0.1, 0.65, 0.25])

                with col_rank:
                    st.markdown(f"### #{i+1}")

                with col_details:
                    st.markdown(f"**{row['title']}**")
                    st.caption(f"🎭 Genres: {row['genres']}")

                    # Add a small badge for high sentiment items
                    if row['sentiment_score'] > 0.5:
                        st.markdown("⭐ *Highly Positive Sentiment*")

                with col_score:
                    # Normalize: (score - min) / (max - min)
                    # This ensures the best is 100% and worst is 0% relative to this list
                    range_score = max_score - min_score
                    norm_score = (row['hybrid_score'] - min_score) / (range_score + 1e-9)

                    st.progress(float(norm_score))
                    st.write(f"Match Score: **{norm_score:.1%}**")

                st.markdown("<br>", unsafe_allow_html=True)
                st.divider()

else:
    # Landing Page
    st.info("👈 Please enter your mood and preferences in the sidebar to get started!")
    
    st.subheader("System Architecture")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🤝 Collaborative")
        st.write("Matrix Factorization (SVD) analyzes user behavior patterns to find movies liked by similar users.")
        
    with col2:
        st.markdown("#### 📂 Content-Based")
        st.write("TF-IDF Vectorization on genres ensures the recommendations align with movie characteristics.")
        
    with col3:
        st.markdown("#### 🧠 Sentiment Aware")
        st.write("VADER Sentiment Analysis processes your mood to dynamically weight the results for the current moment.")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("HybridRec++ v0.1.0 | Built with Streamlit & Scikit-Learn")
