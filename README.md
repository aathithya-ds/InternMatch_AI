# 🚀 InternMatch AI

## Intelligent Internship Recommendation & Career Matching System

InternMatch AI is an AI-powered internship recommendation system that helps students discover suitable internships based on their skills, interests, career goals, and previous internship preferences.

The system combines Content-Based Filtering and Collaborative Filtering to generate personalized internship recommendations.

---

## 🎯 Key Features

- 🤖 AI-powered internship recommendations
- 🧠 Content-Based Filtering
- 👥 Collaborative Filtering
- 🔀 Hybrid Recommendation Engine
- 📊 AI Match Score
- 🛠️ Skill Gap Analysis
- 💡 Matched and Missing Skills
- 👤 Student Profile Analysis
- 📈 Recommendation Evaluation
- 🌐 Flask Web Dashboard
- 📋 Internship details including stipend, duration and work mode

---

## 🧠 Recommendation Approach

InternMatch AI uses a hybrid recommendation strategy:

### 1. Content-Based Filtering

Uses:

- Student skills
- Domain
- Interests
- Career goal

TF-IDF and Cosine Similarity are used to measure profile-to-internship similarity.

### 2. Collaborative Filtering

Analyzes internship ratings from similar students using Cosine Similarity.

### 3. Hybrid Recommendation

The final recommendation score combines:

Content-Based Score: 60%

Collaborative Filtering Score: 40%

This produces a personalized internship ranking.

---

## 🛠️ Technology Stack

### Frontend
- HTML5
- CSS3
- JavaScript

### Backend
- Python
- Flask

### Machine Learning
- Pandas
- NumPy
- Scikit-learn
- TF-IDF
- Cosine Similarity

---

## 📂 Project Structure

```text
InternMatch_AI/
│
├── app.py
├── recommender.py
├── skill_gap.py
├── evaluation.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── students.csv
│   ├── internships.csv
│   └── ratings.csv
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
└── screenshots/