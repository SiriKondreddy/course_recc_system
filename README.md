# Personalized Learning Recommendation System

A machine learning pipeline that builds individualized course recommendations for learners by understanding their interests, skill level, and learning goals — powered by a hybrid of TF-IDF content matching and SVD-based matrix factorization across 41,000+ real-world courses.

---

## Problem Statement

Most online learning platforms show the same trending courses to every user, ignoring their background, goals, and preferences. This project tackles that gap by building a system that treats each learner as an individual and curates a personalized learning journey from Beginner through to Advanced.

---

## How It Works

The system operates in four layers:

**Layer 1 — Data Ingestion**
Raw CSV files from Coursera, edX, Skillshare, and Udemy are loaded, normalized into a unified schema (title, skills, description, level, rating, review count, platform), and cleaned of duplicates and nulls.

**Layer 2 — Feature Engineering**
A text soup is built per course by concatenating title, skills, description, and category. This soup is vectorized using TF-IDF (10,000 features, bigrams). Each course also receives a popularity score derived from its rating and log-normalized review count.

**Layer 3 — Collaborative Filtering**
Simulated user-course interactions are generated using topic-aligned profiles (e.g., a "data science" user rates data science courses highly). Truncated SVD factorizes the sparse 500×5000 interaction matrix into latent user and course factor vectors (70 components).

**Layer 4 — Hybrid Scoring**
For a given learner profile, content-based scores and collaborative scores are computed separately, normalized, and blended as:

```
final_score = 0.4 × content_score + 0.6 × collaborative_score
```

The top-N courses from this blended ranking are returned as personalized recommendations.

---

## Evaluation Results

The hybrid model was evaluated over 100 simulated learner profiles at K=10:

| Metric | Score |
| :--- | :---: |
| Precision@10 | 0.585 |
| Recall@10 | 0.612 |
| F1-Score@10 | 0.596 |
| MAP@10 | 0.587 |
| NDCG@10 | 0.695 |

**NDCG@10 of 0.695** indicates the most relevant courses consistently appear at the top of the recommendation list, not buried below rank 5.

---

## Learning Path Generator

Beyond single recommendations, the system generates a structured three-stage learning roadmap for any career goal:

```
Goal: "Data Science Machine Learning Python"

Stage 1 — Beginner
  [Udemy] Python-Introduction to Data Science and Machine Learning A-Z
  [Udemy] Python for Data Science & Machine Learning: Zero to Hero
  [Udemy] Math 0-1: Calculus for Data Science & Machine Learning

Stage 2 — Intermediate
  [Udemy] Machine Learning With Python: Predicting Customer Churn
  [Coursera] Data Science with Databricks for Data Analysts

Stage 3 — Advanced
  [Coursera] The Nuts and Bolts of Machine Learning
  [Coursera] Introduction to Machine Learning in Production
```

---

## Tech Stack

| Component | Technology |
| :--- | :--- |
| Language | Python 3.13 |
| NLP / Vectorization | Scikit-learn (TF-IDF, Cosine Similarity) |
| Matrix Factorization | Scikit-learn (TruncatedSVD) |
| Data Wrangling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Interface | Streamlit (app.py) |

---


## Dataset Coverage

| Platform | Courses |
| :--- | :---: |
| Udemy | 26,256 |
| Skillshare | 14,250 |
| Coursera | 1,139 |
| edX | 816 |
| **Total (after dedup)** | **41,911** |
