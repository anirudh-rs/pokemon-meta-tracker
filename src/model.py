"""
model.py — Model loading, prediction, and SHAP explanation utilities.

The Streamlit app imports from here exclusively — it never touches
XGBoost or SHAP directly.
"""

import pickle
import numpy as np
import pandas as pd
import shap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_viability.pkl"


def load_model() -> dict:
    """
    Load the trained XGBoost model and metadata.

    Returns dict with keys:
        model: XGBClassifier
        feature_cols: list of feature column names
        threshold: float decision threshold
        roc_auc: float model ROC-AUC on test set
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run the training notebook first."
        )
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)


def predict_viability(
    feature_row: pd.DataFrame,
    model_data: dict | None = None,
) -> dict:
    """
    Predict competitive viability for a single Pokémon.

    Args:
        feature_row: Single-row DataFrame with all feature columns.
                     Column names must match model_data['feature_cols'].
        model_data:  Output of load_model(). If None, loads from disk.

    Returns dict with keys:
        viable: bool
        probability: float (0-1)
        threshold: float
        shap_values: np.array of SHAP values for this prediction
        feature_names: list of feature names
    """
    if model_data is None:
        model_data = load_model()

    model = model_data['model']
    feature_cols = model_data['feature_cols']
    threshold = model_data['threshold']

    # Ensure correct column order
    X = feature_row[feature_cols]

    # Predict
    prob = model.predict_proba(X)[0, 1]
    viable = bool(prob >= threshold)

    # SHAP explanation for this prediction
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)[0]

    return {
        'viable': viable,
        'probability': float(prob),
        'threshold': threshold,
        'shap_values': shap_vals,
        'feature_names': feature_cols,
    }


def get_top_shap_features(
    shap_values: np.ndarray,
    feature_names: list,
    n: int = 10,
) -> pd.DataFrame:
    """
    Return top N features by absolute SHAP value for a single prediction.

    Args:
        shap_values: 1D array of SHAP values from predict_viability()
        feature_names: list of feature names
        n: number of top features to return

    Returns DataFrame with columns: feature, shap_value, direction
    """
    df = pd.DataFrame({
        'feature': feature_names,
        'shap_value': shap_values,
    })
    df['abs_shap'] = df['shap_value'].abs()
    df['direction'] = df['shap_value'].apply(
        lambda x: 'Positive' if x > 0 else 'Negative'
    )
    return df.nlargest(n, 'abs_shap').reset_index(drop=True)


def build_feature_row(
    hp: int, attack: int, defense: int,
    sp_attack: int, sp_defense: int, speed: int,
    type_1: str, type_2: str,
    is_legendary: bool,
    height: float, weight: float,
) -> pd.DataFrame:
    """
    Build a feature row for prediction from raw user inputs.

    This is what the Streamlit dashboard calls when a user
    enters custom stats. Mirrors the feature engineering pipeline.

    Args: Individual stat and type values from the dashboard UI.
    Returns: Single-row DataFrame ready for predict_viability().
    """
    from src.feature_engineering import (
        build_feature_frame, POKEMON_TYPES
    )
    from src.data_loader import load_featured

    # Build a minimal raw row matching load_pokemon() output schema
    raw = pd.DataFrame([{
        'pokedex_id': 0,
        'name': 'Custom',
        'bst': hp + attack + defense + sp_attack + sp_defense + speed,
        'hp': hp,
        'attack': attack,
        'defense': defense,
        'sp_attack': sp_attack,
        'sp_defense': sp_defense,
        'speed': speed,
        'type_1': type_1,
        'type_2': type_2 if type_2 else 'None',
        'height': height,
        'weight': weight,
        'generation': 0,
        'form_type': 'Base',
        'is_legendary': is_legendary,
        'legendary_category': 'Traditional' if is_legendary else 'None',
    }])

    # Run feature engineering on the single row
    featured = build_feature_frame(raw)

    # Align columns to match training data exactly
    # (get_dummies may produce different columns for rare types)
    model_data = load_model()
    feature_cols = model_data['feature_cols']

    # Add any missing columns as 0
    for col in feature_cols:
        if col not in featured.columns:
            featured[col] = 0

    return featured[feature_cols]

TIER_MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_tier_classifier.pkl"


def load_tier_model() -> dict:
    """
    Load the multi-class tier classifier and metadata.

    Returns dict with keys:
        model: XGBClassifier
        feature_cols: list
        label_encoder: LabelEncoder
        tier_order: list
        exact_accuracy: float
    """
    if not TIER_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Tier model not found at {TIER_MODEL_PATH}. "
            "Run the training notebook first."
        )
    with open(TIER_MODEL_PATH, 'rb') as f:
        return pickle.load(f)


def predict_tier(feature_row: pd.DataFrame,
                 tier_model_data: dict | None = None) -> dict:
    """
    Predict competitive tier (Low/Mid/Top) for a single Pokemon.

    Args:
        feature_row: Single-row DataFrame with all feature columns.
        tier_model_data: Output of load_tier_model(). Loads from disk if None.

    Returns dict with keys:
        tier: str ('Low Tier', 'Mid Tier', 'Top Tier')
        probabilities: dict of {tier_name: probability}
        accuracy_note: str disclaimer about model accuracy
    """
    if tier_model_data is None:
        tier_model_data = load_tier_model()

    model        = tier_model_data['model']
    feature_cols = tier_model_data['feature_cols']
    le           = tier_model_data['label_encoder']
    tier_order   = tier_model_data['tier_order']

    X = feature_row[feature_cols]

    pred_class = model.predict(X)[0]
    pred_probs = model.predict_proba(X)[0]

    tier = le.inverse_transform([pred_class])[0]
    probs = {tier_order[i]: round(float(pred_probs[i]) * 100, 1)
             for i in range(len(tier_order))}

    return {
        'tier': tier,
        'probabilities': probs,
        'accuracy_note': 'Tier prediction accuracy: 58.3% on Gen 8 test set.',
    }