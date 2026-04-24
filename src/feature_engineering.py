"""
feature_engineering.py — Derived features for EDA and modelling.

Takes the cleaned DataFrame from data_loader.py and adds derived columns:
- Stat ratios and splits (offensive vs. defensive totals)
- Stat profile classification (e.g., 'Physical Attacker', 'Special Wall')
- Type one-hot encoding (both slot-specific and "has-any-type")
"""

import pandas as pd
import numpy as np

# Competitive ability tier scores
# Used by both feature engineering and the Streamlit predictor
ABILITY_SCORES = {
    'Speed Boost':5,'Regenerator':5,'Protean':5,'Huge Power':5,'Pure Power':5,
    'Drizzle':5,'Drought':5,'Sand Stream':5,'Snow Warning':5,'Electric Surge':5,
    'Psychic Surge':5,'Misty Surge':5,'Grassy Surge':5,'Intimidate':5,
    'Levitate':5,'Magic Bounce':5,'Multiscale':5,'Wonder Guard':5,'Shadow Tag':5,
    'Arena Trap':5,'Imposter':5,'Hadron Engine':5,'Orichalcum Pulse':5,
    'Adaptability':4,'Download':4,'Magic Guard':4,'Tinted Lens':4,'Contrary':4,
    'Moody':4,'Swift Swim':4,'Chlorophyll':4,'Sand Rush':4,'Slush Rush':4,
    'Gale Wings':4,'Tough Claws':4,'Pixilate':4,'Refrigerate':4,'Galvanize':4,
    'Aerilate':4,'Unburden':4,'Prankster':4,'Beast Boost':4,'Soul Heart':4,
    'Intrepid Sword':4,'Dauntless Shield':4,'Water Bubble':4,'Transistor':4,
    "Dragon's Maw":4,'Disguise':4,'Shadow Shield':4,'Neuroforce':4,
    'Mold Breaker':4,'Sand Force':4,'Tablets Of Ruin':4,'Sword Of Ruin':4,
    'Beads Of Ruin':4,'Vessel Of Ruin':4,
    'Iron Fist':3,'Sheer Force':3,'Guts':3,'Technician':3,'Serene Grace':3,
    'Skill Link':3,'Hustle':3,'Rock Head':3,'Reckless':3,'Strong Jaw':3,
    'Mega Launcher':3,'Fur Coat':3,'Thick Fat':3,'Flash Fire':3,
    'Water Absorb':3,'Volt Absorb':3,'Sap Sipper':3,'Storm Drain':3,
    'Lightning Rod':3,'Motor Drive':3,'Defiant':3,'Competitive':3,'Moxie':3,
    'Gooey':3,'Stamina':3,'Steelworker':3,'Natural Cure':3,'Shed Skin':3,
    'Unaware':3,'Punk Rock':3,'Ice Scales':3,'Ice Face':3,'Power Construct':3,
    'Queenly Majesty':3,'Dazzling':3,'Corrosion':3,'Stakeout':3,
    'Full Metal Body':3,'Prism Armor':3,'Shields Down':3,'Comatose':3,
    'Trace':3,'Neutralizing Gas':3,'Gorilla Tactics':3,
    'Static':2,'Flame Body':2,'Poison Point':2,'Effect Spore':2,
    'Synchronize':2,'Marvel Scale':2,'Own Tempo':2,'Cloud Nine':2,
    'Air Lock':2,'Pressure':2,'Filter':2,'Solid Rock':2,'Analytic':2,
    'Rough Skin':2,'Iron Barbs':2,'Weak Armor':2,'Overcoat':2,'Harvest':2,
    'Hydration':2,'Rain Dish':2,'Ice Body':2,'Dry Skin':2,'Solar Power':2,
    'Scrappy':2,'Heatproof':2,'Simple':2,'Flare Boost':2,'Toxic Boost':2,
    'Quick Feet':2,'Anger Point':2,'Unnerve':2,'Frisk':2,'Justified':2,
    'Rattled':2,'Wonder Skin':2,'Cursed Body':2,'Mirror Armor':2,
    'Cute Charm':1,'Oblivious':1,'Illuminate':1,'Trace':3,'Forewarn':1,
    'Anticipation':1,'Leaf Guard':1,'Sand Veil':1,'Snow Cloak':1,
    'Sticky Hold':1,'Suction Cups':1,'Tangled Feet':1,'Normalize':1,
    'Super Luck':2,'Sniper':2,'Pickup':1,'Run Away':1,'Keen Eye':1,
    'Insomnia':1,'Vital Spirit':1,'Early Bird':1,'Shed Skin':3,
    'Defeatist':-3,'Slow Start':-3,'Truant':-3,'Klutz':-2,'Stall':-2,
    'Heavy Metal':-1,'Zen Mode':-1,
}

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

def add_ability_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ability-based features.
    If the row already has ability score columns populated (e.g. from
    build_feature_row for a custom Pokemon), skips the CSV merge entirely.
    """
    from pathlib import Path

    ABILITY_COLS = ['ability_1', 'ability_2', 'hidden_ability',
                    'ability_1_score', 'ability_2_score', 'hidden_ability_score',
                    'best_ability_score', 'has_crippling_ability']

    # If all ability score columns are already present and populated, skip merge
    score_cols_present = all(c in df.columns for c in
                             ['ability_1_score', 'best_ability_score',
                              'has_crippling_ability'])
    if score_cols_present and df['best_ability_score'].notna().all():
        # Ensure string ability name columns exist even if empty
        for col in ['ability_1', 'ability_2', 'hidden_ability']:
            if col not in df.columns:
                df[col] = 'Unknown'
        return df

    # Otherwise load from CSV and merge
    abilities_path = (Path(__file__).resolve().parent.parent
                      / "data" / "raw" / "pokemon_abilities_scored.csv")

    if not abilities_path.exists():
        for col in ABILITY_COLS:
            if col not in df.columns:
                df[col] = 0 if 'score' in col or col == 'has_crippling_ability' else 'Unknown'
        df['best_ability_score']    = df.get('best_ability_score', 1)
        df['has_crippling_ability'] = df.get('has_crippling_ability', 0)
        return df

    ab = pd.read_csv(abilities_path)
    ab_cols = ['pokedex_id', 'ability_1', 'ability_2', 'hidden_ability',
               'ability_1_score', 'ability_2_score', 'hidden_ability_score',
               'best_ability_score', 'has_crippling_ability']

    df = df.merge(ab[ab_cols], on='pokedex_id', how='left')

    df['ability_1']             = df['ability_1'].fillna('Unknown')
    df['ability_2']             = df['ability_2'].fillna('Unknown')
    df['hidden_ability']        = df['hidden_ability'].fillna('Unknown')
    df['ability_1_score']       = df['ability_1_score'].fillna(1)
    df['ability_2_score']       = df['ability_2_score'].fillna(0)
    df['hidden_ability_score']  = df['hidden_ability_score'].fillna(0)
    df['best_ability_score']    = df['best_ability_score'].fillna(1)
    df['has_crippling_ability'] = df['has_crippling_ability'].fillna(0)

    return df

def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps in sequence.

    Input: output of data_loader.load_pokemon() or load_processed()
    Output: DataFrame with all derived features ready for EDA and modelling.
    """
    df = add_stat_features(df)
    df = add_type_features(df)
    df = add_ability_features(df)
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