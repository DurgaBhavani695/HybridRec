# HybridRec++ 🎬

**HybridRec++** is a production-quality movie recommendation system that leverages multiple data signals to provide highly personalized suggestions. It uniquely integrates **Collaborative Filtering**, **Content-Based Filtering**, and **Real-Time Sentiment Analysis** to understand not just what you like, but how you feel *right now*.

## 🚀 Key Features

- **Hybrid Intelligence**: Combines SVD-based Matrix Factorization with TF-IDF Content Analysis.
- **Context-Aware**: Uses VADER Sentiment Analysis to process user mood input and dynamically re-weight recommendations.
- **Interactive UI**: A polished Streamlit dashboard for real-time interaction and mood-based discovery.
- **Cold-Start Resilience**: Leverages content features when user history is thin.

## 🏗️ Architecture

1.  **Collaborative Layer**: Analyzes user-item rating patterns using Truncated SVD.
2.  **Content Layer**: Processes movie metadata (genres) using TF-IDF and Cosine Similarity.
3.  **Sentiment Layer**: Analyzes natural language input to detect user mood (Positive/Negative/Neutral).
4.  **Hybrid Engine**: Merges scores from all layers with dynamic weights based on the detected context.

## 🛠️ Installation & Setup

This project uses `uv` for dependency management.

### Prerequisites
- Python 3.10+
- `uv`

### Quick Start
Run the setup script:
```bash
python setup.py
streamlit run app/streamlit_app.py
```
*(This automatically initializes the environment, installs dependencies, and downloads the necessary data.)*


## 📊 Dataset
The system uses the **MovieLens Small Dataset** (100k ratings on 9k movies), which is automatically downloaded and processed on first run.

## 🧠 How the Hybrid Score Works
The final recommendation score is calculated as follows:
- **Base Score**: `0.5 * Collaborative + 0.3 * Content + 0.2 * Sentiment`
- **Negative Mood**: Increases weight of movies with high sentiment (item sentiment).
- **Positive Mood**: Increases weight of collaborative filtering (leveraging historical preferences).

---
*Built with ❤️ using Python, Scikit-Learn, VADER, and Streamlit.*
