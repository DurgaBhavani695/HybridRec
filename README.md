# HybridRec++ 🎬

**HybridRec++** is a production-grade recommendation system that bridges the gap between collaborative patterns and user sentiment. By integrating **Real-Time NLP** with **Matrix Factorization**, it delivers recommendations that adapt to the user's specific context.

## 🚀 Key Features

- **Dynamic Hybrid Engine**: Combines Collaborative Filtering (MF) and Content-Based (TF-IDF) scores.
- **Mood-Aware Context**: Uses **VADER NLP** to analyze user sentiment in real-time, dynamically re-weighting recommendations based on their current state of mind.
- **Production-Ready Architecture**: Modular design, TDD-validated core, and cross-platform automated setup.
- **Visual Analytics**: Built-in architecture visualization and data distribution insights.

## 🛠️ Tech Stack
| Category | Technology |
| :--- | :--- |
| **Language** | Python 3.10+ |
| **Machine Learning** | Scikit-Learn (TruncatedSVD), Scikit-Surprise |
| **NLP** | NLTK, VADER Sentiment |
| **UI/Dashboard** | Streamlit |
| **Environment** | uv (High-speed package management) |
| **Automation** | Pytest, Git, GitHub Actions/CLI |

## 🏗️ Architecture

The HybridRec++ engine integrates three distinct recommendation layers with a real-time sentiment analysis context layer.

![Architecture Diagram](docs/assets/architecture_diagram.png)

1.  **Collaborative Layer**: Analyzes user-item rating patterns using Truncated SVD.
2.  **Content Layer**: Processes movie metadata (genres) using TF-IDF and Cosine Similarity.
3.  **Sentiment Layer**: Analyzes natural language input to detect user mood (Positive/Negative/Neutral).
4.  **Hybrid Engine**: Merges scores from all layers with dynamic weights based on the detected context.

## 🛠️ Installation & Setup

This project uses `uv` for lightning-fast dependency management and environment execution.

### Prerequisites
- Python 3.10+
- `uv`

### Quick Start
1. **Initialize & Setup:**
   ```bash
   # Initializes virtual environment, installs dependencies, and downloads data
   uv run setup.py
   ```
2. **Launch Application:**
   ```bash
   uv run streamlit run app/streamlit_app.py
   ```
3. **Run Tests:**
   ```bash
   uv run pytest tests/test_collaborative.py
   ```
*(Using `uv run` ensures the application and tests always execute within the project's isolated, reproducible environment.)*


## 📊 Dataset Distribution
We analyzed the rating distribution from the MovieLens dataset to ensure the model is trained on a healthy range of data.

![Rating Distribution](docs/assets/rating_distribution.png)


## 🧠 How the Hybrid Score Works
The final recommendation score is calculated as follows:
- **Base Score**: `0.5 * Collaborative + 0.3 * Content + 0.2 * Sentiment`
- **Negative Mood**: Increases weight of movies with high sentiment (item sentiment).
- **Positive Mood**: Increases weight of collaborative filtering (leveraging historical preferences).

---
*HybridRec++ Recommendation Engine v0.1.0 | Developed for production-grade context-aware systems.*
