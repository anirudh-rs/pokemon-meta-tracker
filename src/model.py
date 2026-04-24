"""
model.py - Model loading, prediction, and SHAP explanation utilities.
"""

import pickle
import numpy as np
import pandas as pd
import shap
from pathlib import Path

PROJECT_ROOT    = Path(__file__).resolve().parent.parent
MODEL_PATH      = PROJECT_ROOT / "models" / "xgboost_viability.pkl"
TIER_MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_tier_classifier.pkl"


def load_model() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run the training notebook first.")
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)


def load_tier_model() -> dict:
    if not TIER_MODEL_PATH.exists():
        raise FileNotFoundError(f"Tier model not found at {TIER_MODEL_PATH}. Run the training notebook first.")
    with open(TIER_MODEL_PATH, 'rb') as f:
        return pickle.load(f)


def predict_viability(feature_row, model_data=None):
    if model_data is None:
        model_data = load_model()
    model        = model_data['model']
    feature_cols = model_data['feature_cols']
    threshold    = model_data['threshold']
    X            = feature_row[feature_cols]
    prob         = model.predict_proba(X)[0, 1]
    explainer    = shap.TreeExplainer(model)
    shap_vals    = explainer.shap_values(X)[0]
    return {
        'viable':        bool(prob >= threshold),
        'probability':   float(prob),
        'threshold':     threshold,
        'shap_values':   shap_vals,
        'feature_names': feature_cols,
    }


def predict_tier(feature_row, tier_model_data=None):
    if tier_model_data is None:
        tier_model_data = load_tier_model()
    model        = tier_model_data['model']
    feature_cols = tier_model_data['feature_cols']
    le           = tier_model_data['label_encoder']
    tier_order   = tier_model_data['tier_order']
    X            = feature_row[feature_cols]
    pred_class   = model.predict(X)[0]
    pred_probs   = model.predict_proba(X)[0]
    tier         = le.inverse_transform([pred_class])[0]
    probs        = {tier_order[i]: round(float(pred_probs[i]) * 100, 1) for i in range(len(tier_order))}
    return {
        'tier':          tier,
        'probabilities': probs,
        'accuracy_note': 'Tier prediction accuracy: 58.3% on Gen 8 test set.',
    }


def get_top_shap_features(shap_values, feature_names, n=10):
    df = pd.DataFrame({'feature': feature_names, 'shap_value': shap_values})
    df['abs_shap']  = df['shap_value'].abs()
    df['direction'] = df['shap_value'].apply(lambda x: 'Positive' if x > 0 else 'Negative')
    return df.nlargest(n, 'abs_shap').reset_index(drop=True)


def build_feature_row(
    hp,
    attack,
    defense,
    sp_attack,
    sp_defense,
    speed,
    type_1,
    type_2,
    is_legendary,
    height,
    weight,
    hidden_ability='Unknown',
    best_ability_score=1,
    has_crippling_ability=0,
):
    from src.feature_engineering import build_feature_frame

    raw = pd.DataFrame([{
        'pokedex_id':            0,
        'name':                  'Custom',
        'bst':                   hp + attack + defense + sp_attack + sp_defense + speed,
        'hp':                    hp,
        'attack':                attack,
        'defense':               defense,
        'sp_attack':             sp_attack,
        'sp_defense':            sp_defense,
        'speed':                 speed,
        'type_1':                type_1,
        'type_2':                type_2 if type_2 else 'None',
        'height':                height,
        'weight':                weight,
        'generation':            0,
        'form_type':             'Base',
        'is_legendary':          int(is_legendary),
        'legendary_category':    'Traditional' if is_legendary else 'None',
        'ability_1':             hidden_ability,
        'ability_2':             'Unknown',
        'hidden_ability':        hidden_ability,
        'ability_1_score':       float(best_ability_score),
        'ability_2_score':       0.0,
        'hidden_ability_score':  float(best_ability_score),
        'best_ability_score':    float(best_ability_score),
        'has_crippling_ability': float(has_crippling_ability),
    }])

    featured     = build_feature_frame(raw)
    model_data   = load_model()
    feature_cols = model_data['feature_cols']

    for col in feature_cols:
        if col not in featured.columns:
            featured[col] = 0

    return featured[feature_cols]
