# MATCH OUTCOME PREDICTION SYSTEM
### Sports Analytics using Logistic Regression and Random Forest

A production-ready sports analytics web application and machine learning inference dashboard built with **FastAPI**, **Scikit-learn**, and a modern **Vanilla HTML5/CSS/JavaScript** frontend.

---

## 📌 Project Overview

This project implements match outcome predictions between two sports teams based on historical performance indicators, composite player performance indices, goal rates, and venue home/away advantage.

The system utilizes two trained classification algorithms:
1. **Logistic Regression** (Linear classifier utilizing standardized 47-dimensional feature vectors)
2. **Random Forest Classifier** (Ensemble of 200 de-correlated decision trees)

### Target Formulation
- **Target `1`**: Team A Wins
- **Target `0`**: Team A Loses (Team B Wins)

---

## 📂 Project Directory Structure

```text
match-outcome-prediction/
│
├── backend/
│   ├── main.py                  # FastAPI server with ML inference endpoints & static file serving
│   └── requirements.txt         # Python package dependencies
│
├── frontend/
│   ├── index.html               # Full responsive dashboard interface
│   ├── style.css                # Dark navy glassmorphism sports analytics design system
│   └── script.js                # Dynamic UI, API communication, validation & Chart.js rendering
│
├── models/
│   ├── logistic_regression_model.pkl   # Serialized Logistic Regression model
│   ├── random_forest_model.pkl         # Serialized Random Forest (200 trees)
│   └── feature_scaler.pkl              # Fitted StandardScaler (47 features)
│
├── data/
│   ├── match_outcome_sample_dataset.csv  # 20 historical matches with 13 attributes
│   └── model_comparison_results.csv     # Model evaluation metrics (Accuracy, Precision, Recall, F1)
│
└── README.md                    # Project documentation
```

---

## 🧠 Machine Learning & Preprocessing Architecture

### 1. Exact Input Features & Dimension (47 Features)
The inference pipeline automatically expands user-provided match parameters into the exact 47-dimensional vector required by the trained models:

- **Numerical Statistics (8 features):**
  - `Team_A_Avg_Score`
  - `Team_B_Avg_Score`
  - `Team_A_Win_Rate`
  - `Team_B_Win_Rate`
  - `Team_A_Avg_Goals`
  - `Team_B_Avg_Goals`
  - `Team_A_Player_Performance`
  - `Team_B_Player_Performance`
- **Categorical One-Hot Encodings (39 features):**
  - `Team_A`: 18 one-hot indicators (`Team_A_Team C` through `Team_A_Team T`, dropping baseline `Team A`)
  - `Team_B`: 19 one-hot indicators (`Team_B_Team B` through `Team_B_Team T`, dropping baseline `Team A`)
  - `Venue`: 2 one-hot indicators (`Venue_Home`, `Venue_Neutral`, dropping baseline `Away`)

### 2. Model Preprocessing Steps
- **Logistic Regression**: Passes the 47-feature DataFrame through `feature_scaler.transform()`, then computes `predict()` and `predict_proba()`.
- **Random Forest**: Passes the unscaled 47-feature DataFrame with verified column names directly into `predict()` and `predict_proba()`.

### 3. Model Benchmark Evaluation (from `model_comparison_results.csv`)
| Model | Accuracy | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **100.0%** | **100.0%** | **100.0%** | **100.00%** |
| **Random Forest** | **75.0%** | **100.0%** | **50.0%** | **66.67%** |

---

## 🚀 Setup & Execution Guide

### 1. Prerequisites
- Python 3.9+
- A modern web browser

### 2. Create and Activate Virtual Environment
```bash
cd /Users/karthikreddy/Desktop/match-outcome-prediction

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r backend/requirements.txt
```

### 4. Run the Application
Start the FastAPI server using Uvicorn:
```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

### 5. Access the Web Dashboard
Open your browser and navigate to:
```
http://localhost:8000
```
Interactive API docs are available at:
```
http://localhost:8000/docs
```

---

## 🔌 API Endpoints Reference

### `GET /api/health`
Checks server status, verifies that all 3 ML model artifacts are loaded, and returns feature counts.

### `GET /api/teams`
Returns all unique participating teams present in the dataset (`Team A` through `Team T`).

### `GET /api/venues`
Returns all unique venues supported by the model (`Home`, `Away`, `Neutral`).

### `GET /api/dataset-info`
Provides dataset summary stats (total matches, attributes, venues), sample dataset rows, official evaluation comparison metrics, and Random Forest feature importances.

### `POST /api/predict`
Calculates real-time predictions and win/loss probabilities from both models.

**Request Payload:**
```json
{
  "team_a": "Team A",
  "team_b": "Team B",
  "venue": "Home",
  "team_a_avg_score": 78.0,
  "team_b_avg_score": 72.0,
  "team_a_win_rate": 65.0,
  "team_b_win_rate": 58.0,
  "team_a_avg_goals": 12.0,
  "team_b_avg_goals": 8.0,
  "team_a_player_performance": 62.0,
  "team_b_player_performance": 55.0
}
```

**Response Format:**
```json
{
  "match_details": {
    "team_a": "Team A",
    "team_b": "Team B",
    "venue": "Home"
  },
  "logistic_regression": {
    "prediction": "TEAM A WINS",
    "target": 1,
    "predicted_winner": "Team A",
    "team_a_probability": 96.62,
    "team_a_loss_probability": 3.38
  },
  "random_forest": {
    "prediction": "TEAM A WINS",
    "target": 1,
    "predicted_winner": "Team A",
    "team_a_probability": 100.0,
    "team_a_loss_probability": 0.0
  }
}
```

---

## 🧪 Testing Checklist Completed
- [x] All 5 ML and data files inspected and verified
- [x] Exact 47-feature vector reproduced and validated against sample row 0
- [x] Backend FastAPI application created with input validation
- [x] Health check endpoint (`/api/health`) verified
- [x] Teams endpoint (`/api/teams`) verified
- [x] Venues endpoint (`/api/venues`) verified
- [x] Dataset Info endpoint (`/api/dataset-info`) verified
- [x] Prediction endpoint (`/api/predict`) verified with live predictions
- [x] Full UI dashboard tested with browser interactions
# Machine-Learning-Project
