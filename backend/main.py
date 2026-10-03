import os
import sys
import warnings
from pathlib import Path
from typing import Dict, Any, List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

# Suppress harmless sklearn unpickling warnings
warnings.filterwarnings("ignore")

# Define base paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
FRONTEND_DIR = BASE_DIR / "frontend"

# Initialize FastAPI app
app = FastAPI(
    title="MATCH OUTCOME PREDICTION SYSTEM",
    description="Sports Analytics using Logistic Regression and Random Forest",
    version="1.0.0",
)

# Enable CORS for cross-origin frontend support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and data
feature_scaler = None
logistic_model = None
rf_model = None
feature_columns: List[str] = []
dataset_df: pd.DataFrame = None
model_comparison_df: pd.DataFrame = None
teams_list: List[str] = []
venues_list: List[str] = []


def load_artifacts():
    global feature_scaler, logistic_model, rf_model, feature_columns
    global dataset_df, model_comparison_df, teams_list, venues_list

    scaler_path = MODELS_DIR / "feature_scaler.pkl"
    lr_path = MODELS_DIR / "logistic_regression_model.pkl"
    rf_path = MODELS_DIR / "random_forest_model.pkl"
    dataset_path = DATA_DIR / "match_outcome_sample_dataset.csv"
    comp_path = DATA_DIR / "model_comparison_results.csv"

    # Verify all files exist
    for p in [scaler_path, lr_path, rf_path, dataset_path, comp_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required ML artifact missing: {p}")

    # Load artifacts using joblib
    feature_scaler = joblib.load(scaler_path)
    logistic_model = joblib.load(lr_path)
    rf_model = joblib.load(rf_path)
    feature_columns = list(feature_scaler.feature_names_in_)

    # Load CSV data
    dataset_df = pd.read_csv(dataset_path)
    model_comparison_df = pd.read_csv(comp_path)

    # Derive unique teams and venues from dataset
    teams_set = set(dataset_df["Team_A"].dropna().unique()).union(
        set(dataset_df["Team_B"].dropna().unique())
    )
    teams_list = sorted(list(teams_set))
    venues_list = sorted(list(dataset_df["Venue"].dropna().unique()))


# Load artifacts at startup
load_artifacts()


class PredictionRequest(BaseModel):
    team_a: str = Field(..., description="Team A name")
    team_b: str = Field(..., description="Team B name")
    venue: str = Field(..., description="Match venue (e.g. Home, Away, Neutral)")
    team_a_avg_score: float = Field(..., description="Team A Average Score", ge=0, le=200)
    team_b_avg_score: float = Field(..., description="Team B Average Score", ge=0, le=200)
    team_a_win_rate: float = Field(..., description="Team A Win Rate (0-100%)", ge=0, le=100)
    team_b_win_rate: float = Field(..., description="Team B Win Rate (0-100%)", ge=0, le=100)
    team_a_avg_goals: float = Field(..., description="Team A Average Goals", ge=0, le=50)
    team_b_avg_goals: float = Field(..., description="Team B Average Goals", ge=0, le=50)
    team_a_player_performance: float = Field(..., description="Team A Player Performance (0-100)", ge=0, le=100)
    team_b_player_performance: float = Field(..., description="Team B Player Performance (0-100)", ge=0, le=100)

    @field_validator("team_b")
    @classmethod
    def check_different_teams(cls, v, info):
        team_a = info.data.get("team_a")
        if team_a and v.strip().lower() == team_a.strip().lower():
            raise ValueError("Team A and Team B cannot be identical. Please select different teams.")
        return v


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "models_loaded": {
            "logistic_regression": logistic_model is not None,
            "random_forest": rf_model is not None,
            "feature_scaler": feature_scaler is not None,
        },
        "dataset_rows": len(dataset_df) if dataset_df is not None else 0,
        "features_count": len(feature_columns),
    }


@app.get("/api/teams")
def get_teams():
    return {
        "teams": teams_list,
        "total_teams": len(teams_list),
    }


@app.get("/api/venues")
def get_venues():
    return {
        "venues": venues_list,
        "total_venues": len(venues_list),
    }


@app.get("/api/dataset-info")
def get_dataset_info():
    if dataset_df is None or model_comparison_df is None:
        raise HTTPException(status_code=500, detail="Data files not loaded")

    # Format model comparison table
    model_comparison = model_comparison_df.to_dict(orient="records")

    # Format sample dataset records
    sample_records = dataset_df.to_dict(orient="records")

    # Feature importances from Random Forest
    rf_importances = []
    if hasattr(rf_model, "feature_importances_"):
        raw_imp = rf_model.feature_importances_
        sorted_indices = np.argsort(raw_imp)[::-1]
        for idx in sorted_indices[:10]:
            rf_importances.append({
                "feature": feature_columns[idx],
                "importance": round(float(raw_imp[idx]) * 100, 2),
            })

    return {
        "total_matches": int(len(dataset_df)),
        "total_features": int(dataset_df.shape[1]),
        "number_of_teams": int(len(teams_list)),
        "number_of_venues": int(len(venues_list)),
        "teams": teams_list,
        "venues": venues_list,
        "columns": list(dataset_df.columns),
        "sample_data": sample_records,
        "model_comparison": model_comparison,
        "top_features": rf_importances,
    }


@app.post("/api/predict")
def predict_match(req: PredictionRequest):
    if req.team_a.strip().lower() == req.team_b.strip().lower():
        raise HTTPException(
            status_code=400,
            detail="Team A and Team B cannot be the same team. Please choose distinct teams.",
        )

    # Prepare the exact 47-feature dictionary
    feature_dict: Dict[str, float] = {col: 0.0 for col in feature_columns}

    # Populate numerical statistics
    feature_dict["Team_A_Avg_Score"] = float(req.team_a_avg_score)
    feature_dict["Team_B_Avg_Score"] = float(req.team_b_avg_score)
    feature_dict["Team_A_Win_Rate"] = float(req.team_a_win_rate)
    feature_dict["Team_B_Win_Rate"] = float(req.team_b_win_rate)
    feature_dict["Team_A_Avg_Goals"] = float(req.team_a_avg_goals)
    feature_dict["Team_B_Avg_Goals"] = float(req.team_b_avg_goals)
    feature_dict["Team_A_Player_Performance"] = float(req.team_a_player_performance)
    feature_dict["Team_B_Player_Performance"] = float(req.team_b_player_performance)

    # Populate one-hot encoded categories if present in feature names
    col_team_a = f"Team_A_{req.team_a}"
    if col_team_a in feature_dict:
        feature_dict[col_team_a] = 1.0

    col_team_b = f"Team_B_{req.team_b}"
    if col_team_b in feature_dict:
        feature_dict[col_team_b] = 1.0

    col_venue = f"Venue_{req.venue}"
    if col_venue in feature_dict:
        feature_dict[col_venue] = 1.0

    # Build DataFrame with exact feature order
    df_features = pd.DataFrame([feature_dict], columns=feature_columns)

    # 1. LOGISTIC REGRESSION PREDICTION (with standard scaler)
    scaled_features = feature_scaler.transform(df_features)
    lr_pred_raw = int(logistic_model.predict(scaled_features)[0])
    lr_proba_raw = logistic_model.predict_proba(scaled_features)[0]  # [prob_loss, prob_win]
    lr_win_prob = round(float(lr_proba_raw[1]) * 100, 2)
    lr_loss_prob = round(float(lr_proba_raw[0]) * 100, 2)
    lr_label = "TEAM A WINS" if lr_pred_raw == 1 else "TEAM A LOSES"

    # 2. RANDOM FOREST PREDICTION (unscaled DataFrame)
    rf_pred_raw = int(rf_model.predict(df_features)[0])
    rf_proba_raw = rf_model.predict_proba(df_features)[0]  # [prob_loss, prob_win]
    rf_win_prob = round(float(rf_proba_raw[1]) * 100, 2)
    rf_loss_prob = round(float(rf_proba_raw[0]) * 100, 2)
    rf_label = "TEAM A WINS" if rf_pred_raw == 1 else "TEAM A LOSES"

    return {
        "match_details": {
            "team_a": req.team_a,
            "team_b": req.team_b,
            "venue": req.venue,
        },
        "logistic_regression": {
            "prediction": lr_label,
            "target": lr_pred_raw,
            "predicted_winner": req.team_a if lr_pred_raw == 1 else req.team_b,
            "team_a_probability": lr_win_prob,
            "team_a_loss_probability": lr_loss_prob,
        },
        "random_forest": {
            "prediction": rf_label,
            "target": rf_pred_raw,
            "predicted_winner": req.team_a if rf_pred_raw == 1 else req.team_b,
            "team_a_probability": rf_win_prob,
            "team_a_loss_probability": rf_loss_prob,
        },
    }


# Mount frontend directory for static serving
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
