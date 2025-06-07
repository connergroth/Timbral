<p align="center">
  <img src="https://github.com/user-attachments/assets/522ad4ce-2b4a-41e4-989d-6a28f325dbf3" alt="Tensoe-Recommender" width="355"/>
</p>



# Tensoe Recommender

**Tensoe Recommender** is the machine learning engine behind [Tensoe](https:/github.com/connergroth/tensoe), an AI-powered music discovery platform. It blends collaborative filtering, content-based filtering, and user-centric data (from Last.fm and AlbumOfTheYear.org) to generate highly personalized, explainable music recommendations.

> _"A fusion of tensor and tone, using machine learning to shape resonant sound."_

---

## 🤖 Overview

This repository contains all ML-related logic, including:
- Data preprocessing pipelines
- Feature extraction
- Embedding generation
- Hybrid recommendation model (PyTorch)
- Inference API stubs
- Batch recommendation jobs

---

## 🧠 Model Design

### 🔸 Collaborative Filtering (CF)
- Based on implicit user-track interactions
- Matrix Factorization (e.g., NMF or ALS)

### 🔹 Content-Based Filtering (CBF)
- TF-IDF on tags/moods/genres
- Genre and mood embeddings (Last.fm, AOTY)
- Cosine similarity for related tracks

### 🔶 Hybrid Fusion
- Combines CF and CBF scores
- Supports tunable weights
- Outputs enriched recommendations with reasoning metadata

---

## 📂 Project Structure

```bash
tensoe-recommender/
├── data/
│   ├── user_track_matrix.csv
│   ├── track_features.json
│   ├── track_embeddings.pt
│   └── similarity_matrix.npz
├── model/
│   ├── train.py
│   ├── hybrid.py
│   ├── content_based.py
│   ├── collaborative.py
│   └── evaluation.py
├── pipeline/
│   ├── build_matrix.py
│   ├── extract_features.py
│   └── normalize_sources.py
├── infer.py
├── batch_precompute.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourname/tensoe-recommender.git
cd tensoe-recommender
```

### 2. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Environment Variables

Create a `.env` file:

```env
REDIS_URL=redis://localhost:6379
SUPABASE_URL=https://yourproject.supabase.co
SUPABASE_KEY=your-supabase-api-key
```

---

## 📊 Training

```bash
python model/train.py
```

This will:
- Build the user-item matrix
- Train CF/CBF models
- Merge outputs
- Save `track_embeddings.pt` and `recommendation_cache.json`

---

## 🔍 Inference

Use `infer.py`:

```python
from infer import get_user_recommendations

recs = get_user_recommendations(user_id="123", top_k=20)
```

Guest (seed-based):
```python
get_guest_recommendations(seed_track_id="spotify:abc123")
```

---

## 📦 Batch Precomputation

```bash
python batch_precompute.py
```

- Generates top-K recs per user
- Stores to Redis or Supabase

---

## 📊 Evaluation

```bash
python model/evaluation.py
```

Supports:
- Precision@k
- Recall@k
- Offline NDCG

---

## 🚀 Roadmap

- 🎙️ Audio/Lyric Embeddings
- 📈 User feedback loop integration
- 👩‍🔬 Personalized fine-tuning
- 🧠 GPT-powered explainability

## 📰 Credits

Built by [Conner Groth](https://www.connergroth.com) for the Tensoe ecosystem.

Model powered by real-world music intelligence from Spotify, Last.fm, and albumoftheyear.org.
