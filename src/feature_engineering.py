"""
feature_engineering.py — Derived features for EDA and modelling.

Takes the cleaned DataFrame from data_loader.py and adds derived columns:
- Stat ratios and splits (offensive vs. defensive totals)
- Stat profile classification (e.g., 'Physical Attacker', 'Special Wall')
- Type one-hot encoding (both slot-specific and "has-any-type")
"""

import pandas as pd
import numpy as np

# The 18 canonical Pokémon types
POKEMON_TYPES = [
    "Normal", "Fire", "Water", "Electric", "Grass", "Ice",
    "Fighting", "Poison", "Ground", "Flying", "Psychic", "Bug",
    "Rock", "Ghost", "Dragon", "Dark", "Steel", "Fairy",
]

# Core stat columns — used in several feature calculations below
STAT_COLS = ["hp", "attack", "defense", "sp_attack", "sp_defense", "speed"]


def add_stat_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived stat columns:

    - offensive_total: attack + sp_attack + speed (offensive capability)
    - defensive_total: hp + defense + sp_defense (defensive capability)
    - physical_bias: attack - sp_attack (positive = physical attacker)
    - defensive_bias: defense - sp_defense (positive = physically bulky)
    - speed_tier: categorical speed bucket (Slow/Medium/Fast/Very Fast)
    - stat_profile: dominant stat role (e.g., 'Physical Sweeper')
    """
    df = df.copy()

    # Offensive / defensive totals
    df["offensive_total"] = df["attack"] + df["sp_attack"] + df["speed"]
    df["defensive_total"] = df["hp"] + df["defense"] + df["sp_defense"]

    # Physical vs. special bias — positive means physical-leaning
    df["physical_bias"] = df["attack"] - df["sp_attack"]
    df["defensive_bias"] = df["defense"] - df["sp_defense"]

    # Speed tier — based on competitive Pokémon conventions
    # <60: Slow, 60-89: Medium, 90-109: Fast, 110+: Very Fast
    df["speed_tier"] = pd.cut(
        df["speed"],
        bins=[-1, 59, 89, 109, 255],
        labels=["Slow", "Medium", "Fast", "Very Fast"],
    )

    # Stat profile — competitive role based on top two stats
    df["stat_profile"] = df.apply(_classify_stat_profile, axis=1)

    return df


def _classify_stat_profile(row: pd.Series) -> str:
    """
    Classify a Pokémon's competitive role based on its two highest stats.

    Logic:
    - Attack + Speed → Physical Sweeper
    - Sp.Atk + Speed → Special Sweeper
    - Any two offensive → Mixed Attacker
    - HP + Defense → Physical Wall
    - HP + Sp.Def → Special Wall
    - Any two defensive → Defensive Wall
    - Otherwise → Balanced (one offensive, one defensive in top two)
    """
    stats = {stat: row[stat] for stat in STAT_COLS}
    top_two = sorted(stats.items(), key=lambda x: x[1], reverse=True)[:2]
    top_stats = {s[0] for s in top_two}

    offensive = {"attack", "sp_attack", "speed"}
    defensive = {"hp", "defense", "sp_defense"}

    if top_stats == {"attack", "speed"}:
        return "Physical Sweeper"
    if top_stats == {"sp_attack", "speed"}:
        return "Special Sweeper"
    if top_stats.issubset(offensive):
        return "Mixed Attacker"
    if top_stats == {"hp", "defense"}:
        return "Physical Wall"
    if top_stats == {"hp", "sp_defense"}:
        return "Special Wall"
    if top_stats.issubset(defensive):
        return "Defensive Wall"
    return "Balanced"


def add_type_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add type-related features:

    - num_types: 1 for mono-type, 2 for dual-type
    - has_type_<TypeName>: 18 boolean columns, True if Pokémon has that type
      in either slot 1 or slot 2 (useful for "how many Fire types" queries)
    - type_1_<TypeName> and type_2_<TypeName>: slot-specific one-hot
      (for the model, since slot-1 vs slot-2 matters for some mechanics)
    """
    df = df.copy()

    # num_types — 'None' in type_2 means mono-type
    df["num_types"] = np.where(df["type_2"] == "None", 1, 2)

    # "Has this type" columns — True if either slot matches
    for ptype in POKEMON_TYPES:
        df[f"has_type_{ptype}"] = (
            (df["type_1"] == ptype) | (df["type_2"] == ptype)
        )

    # Slot-specific one-hot — for the model
    type_1_dummies = pd.get_dummies(df["type_1"], prefix="type_1")
    type_2_dummies = pd.get_dummies(df["type_2"], prefix="type_2")

    df = pd.concat([df, type_1_dummies, type_2_dummies], axis=1)

    return df


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps in sequence.

    Input: output of data_loader.load_pokemon() or load_processed()
    Output: DataFrame with all derived features ready for EDA and modelling.
    """
    df = add_stat_features(df)
    df = add_type_features(df)
    return df


if __name__ == "__main__":
    # Quick sanity check when run directly
    from src.data_loader import load_processed

    df = load_processed()
    df = build_feature_frame(df)

    print(f"Final shape: {df.shape}")
    print(f"\nStat profile distribution:\n{df['stat_profile'].value_counts()}")
    print(f"\nSpeed tier distribution:\n{df['speed_tier'].value_counts()}")
    print(f"\nDual-type %: {(df['num_types'] == 2).mean():.1%}")