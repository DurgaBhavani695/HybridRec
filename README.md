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
The engine uses a sophisticated three-factor scoring algorithm:

- **Base Score**: `0.5 * Collaborative (SVD) + 0.3 * Content (TF-IDF) + 0.2 * Item Sentiment`
- **Context-Aware Adjustment**: 
    - **Intensity**: We calculate mood intensity from -1.0 to 1.0. The stronger the emotion, the larger the dynamic boost to the recommendation score.
    - **Neutral Mode**: If no strong sentiment is detected, the engine gracefully falls back to the personalized collaborative filtering score, ensuring stable and reliable results.
    - **Positive/Negative Shift**: 
        - *Negative Sentiment*: Boosts feel-good items proportional to the user's negative intensity.
        - *Positive Sentiment*: Boosts historical collaborative matches proportional to the user's positive intensity.
    - **Topic Matching**: The system scans your input for semantic keywords (e.g., 'sports', 'funny', 'space') and applies an aggressive **0.8 boost** to corresponding movie genres, ensuring your specific intent is prioritized.

## 🧪 Testing the Context Engine
You can test the system's adaptability by trying these different mood scenarios:

| Mood Category | Input Example | Why it's interesting |
| :--- | :--- | :--- |
| **Relaxation** | *"I'm exhausted, need something funny."* | Boosts 'Comedy' genre + positive sentiment items. |
| **High Intensity** | *"I AM PUMPED! GIVE ME SPACE ACTION!"* | High-intensity boost for collaborative hits + Sci-Fi/Action genre boost. |
| **Baseline** | *"Just a normal day."* | Allows the Collaborative Engine to lead with pure history. |

### Visual Demonstrations
The system dynamically adjusts its scoring algorithm based on both **Sentiment Intensity** and **Intent (Topic Extraction)**.

| Mood Input | Detected Mood | UI Result |
| :--- | :--- | :--- |
| Relaxing mood | Positive/Neutral | ![Result 1](docs/assets/ui_example_1.png) |
| Pumped/Action | Positive (High Intensity) | ![Result 2](docs/assets/ui_example_2.png) |
| Mixed Sentiment | Variable | ![Result 3](docs/assets/ui_example_3.png) |
| Detailed Context | Context-Aware | ![Result 4](docs/assets/ui_example_4.png) |
| User Profile 1 | Personalization | ![Result 5](docs/assets/ui_example_5.png) |
| User Profile 4 | Personalization | ![Result 6](docs/assets/ui_example_6.png) |

---
*HybridRec++ Recommendation Engine v0.1.0 | Developed for production-grade context-aware systems.*
