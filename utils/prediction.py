import json
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path
from typing import Dict, Any, Tuple

# Resolve paths relative to this file's directory
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "loan_approval_pipeline.joblib"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"
DATASET_PATH = BASE_DIR / "loan_approval_data.csv"

@st.cache_resource
def load_pipeline():
    """
    Loads and caches the complete pre-trained Scikit-learn Pipeline.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please run train_model.py first.")
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metadata() -> Dict[str, Any]:
    """
    Loads and caches model metadata and configurations.
    """
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata file not found at {METADATA_PATH}. Please run train_model.py first.")
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_historical_data() -> pd.DataFrame:
    """
    Loads and caches the historical dataset for visualization.
    """
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset file not found at {DATASET_PATH}.")
    return pd.read_csv(DATASET_PATH)

def make_prediction(input_data: Dict[str, Any]) -> Tuple[int, float, float]:
    """
    Takes a dictionary of input features, formats it into a single-row DataFrame
    as expected by the Scikit-learn Pipeline, and returns the prediction and class probabilities.
    
    Returns:
        - prediction (int): 1 for Likely Approved, 0 for Likely Not Approved
        - approval_prob (float): Probability of loan approval [0.0, 1.0]
        - rejection_prob (float): Probability of loan rejection [0.0, 1.0]
    """
    pipeline = load_pipeline()
    metadata = load_metadata()
    
    # Get required features
    numerical_features = metadata["required_input_fields"]["numerical"]
    categorical_features = metadata["required_input_fields"]["categorical"]
    all_features = numerical_features + categorical_features
    
    # Build single-row DataFrame
    # Ensure correct data types (float for numerical, string/object for categorical)
    row_data = {}
    for col in all_features:
        val = input_data.get(col)
        if col in numerical_features:
            row_data[col] = [float(val) if val is not None else None]
        else:
            row_data[col] = [str(val) if val is not None else None]
            
    df_input = pd.DataFrame(row_data)
    
    # Generate prediction and probabilities
    pred = pipeline.predict(df_input)[0]
    probs = pipeline.predict_proba(df_input)[0]
    
    # Class 0: Not Approved (Rejection), Class 1: Approved
    rejection_prob = float(probs[0])
    approval_prob = float(probs[1])
    
    return int(pred), approval_prob, rejection_prob
