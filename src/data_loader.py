"""
data_loader.py — Loads and normalises the raw Pokémon dataset.

This module is the single source of truth for raw data access.
Notebooks and the Streamlit app should import from here, never
read the CSV directly.
"""

from pathlib import Path
import pandas as pd

# Project root detection — works from notebooks/, app/, or src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "pokemon.csv"

# Generation boundaries by National Pokédex ID
# Source: official Pokédex ranges, Gen 1-9
GENERATION_RANGES = {
    1: (1, 151),
    2: (152, 251),
    3: (252, 386),
    4: (387, 493),
    5: (494, 649),
    6: (650, 721),
    7: (722, 809),
    8: (810, 905),
    9: (906, 1025),
}

# Hand-curated list of Legendary and Mythical Pokémon by name
# Includes Sub-Legendaries, Legendaries, Mythicals, Ultra Beasts, Paradox Pokémon
LEGENDARY_NAMES = {
    # Gen 1
    "Articuno", "Zapdos", "Moltres", "Mewtwo", "Mew",
    # Gen 2
    "Raikou", "Entei", "Suicune", "Lugia", "Ho-Oh", "Celebi",
    # Gen 3
    "Regirock", "Regice", "Registeel", "Latias", "Latios",
    "Kyogre", "Groudon", "Rayquaza", "Jirachi", "Deoxys",
    # Gen 4
    "Uxie", "Mesprit", "Azelf", "Dialga", "Palkia", "Heatran",
    "Regigigas", "Giratina", "Cresselia", "Phione", "Manaphy",
    "Darkrai", "Shaymin", "Arceus",
    # Gen 5
    "Victini", "Cobalion", "Terrakion", "Virizion", "Tornadus",
    "Thundurus", "Reshiram", "Zekrom", "Landorus", "Kyurem",
    "Keldeo", "Meloetta", "Genesect",
    # Gen 6
    "Xerneas", "Yveltal", "Zygarde", "Diancie", "Hoopa", "Volcanion",
    # Gen 7 (includes Ultra Beasts)
    "Type: Null", "Silvally", "Tapu Koko", "Tapu Lele", "Tapu Bulu",
    "Tapu Fini", "Cosmog", "Cosmoem", "Solgaleo", "Lunala",
    "Nihilego", "Buzzwole", "Pheromosa", "Xurkitree", "Celesteela",
    "Kartana", "Guzzlord", "Necrozma", "Magearna", "Marshadow",
    "Poipole", "Naganadel", "Stakataka", "Blacephalon", "Zeraora",
    "Meltan", "Melmetal",
    # Gen 8 (includes DLC)
    "Zacian", "Zamazenta", "Eternatus", "Kubfu", "Urshifu",
    "Zarude", "Regieleki", "Regidrago", "Glastrier", "Spectrier",
    "Calyrex", "Enamorus",
    # Gen 9 (includes Paradox Pokémon and DLC)
    "Wo-Chien", "Chien-Pao", "Ting-Lu", "Chi-Yu",
    "Koraidon", "Miraidon",
    "Great Tusk", "Scream Tail", "Brute Bonnet", "Flutter Mane",
    "Slither Wing", "Sandy Shocks", "Roaring Moon",
    "Iron Treads", "Iron Bundle", "Iron Hands", "Iron Jugulis",
    "Iron Moth", "Iron Thorns", "Iron Valiant",
    "Walking Wake", "Iron Leaves", "Gouging Fire", "Raging Bolt",
    "Iron Boulder", "Iron Crown",
    "Okidogi", "Munkidori", "Fezandipiti", "Ogerpon",
    "Terapagos", "Pecharunt",
}

# Mythical Pokémon — event-only, narrative-special, BST typically 600
MYTHICAL_NAMES = {
    "Mew", "Celebi", "Jirachi", "Deoxys",
    "Phione", "Manaphy", "Darkrai", "Shaymin", "Arceus",
    "Victini", "Keldeo", "Meloetta", "Genesect",
    "Diancie", "Hoopa", "Volcanion",
    "Magearna", "Marshadow", "Zeraora", "Meltan", "Melmetal",
    "Zarude",
    "Pecharunt",
}

# Ultra Beasts — Gen 7, BST typically 570, mechanically distinct from legendaries
ULTRA_BEASTS = {
    "Nihilego", "Buzzwole", "Pheromosa", "Xurkitree",
    "Celesteela", "Kartana", "Guzzlord",
    "Poipole", "Naganadel", "Stakataka", "Blacephalon",
}

# Paradox Pokémon — Gen 9 past/future variants
PARADOX_POKEMON = {
    # Past Paradox
    "Great Tusk", "Scream Tail", "Brute Bonnet", "Flutter Mane",
    "Slither Wing", "Sandy Shocks", "Roaring Moon",
    "Walking Wake", "Gouging Fire", "Raging Bolt",
    # Future Paradox
    "Iron Treads", "Iron Bundle", "Iron Hands", "Iron Jugulis",
    "Iron Moth", "Iron Thorns", "Iron Valiant",
    "Iron Leaves", "Iron Boulder", "Iron Crown",
}


def _get_generation(pokemon_id: int) -> int:
    """Return generation number from National Pokédex ID."""
    for gen, (start, end) in GENERATION_RANGES.items():
        if start <= pokemon_id <= end:
            return gen
    return 0  # Out of known range — flag for investigation


def _get_form_type(name: str) -> str:
    """
    Classify each row as Base, Mega, Regional, Gigantamax, or Alternate form.

    Detection order matters — we check for more specific form types first
    (Mega, Regional) before falling through to the general Alternate check.
    """
    name_lower = name.lower()

    # Mega Evolutions — "Charizard Mega Charizard X"
    if "mega " in name_lower or name_lower.startswith("mega"):
        return "Mega"

    # Regional variants — Alolan, Galarian, Hisuian, Paldean
    if any(region in name_lower for region in ["alola", "galar", "hisui", "paldea"]):
        return "Regional"

    # Gigantamax — included for safety even though this dataset doesn't have them
    if "gmax" in name_lower or "gigantamax" in name_lower:
        return "Gigantamax"

    # Alternate formes — Primal, Origin, Therian, Crowned, Eternamax, Ultra, etc.
    # We match "Name Variant Name" pattern: a word appearing after the species name
    alternate_keywords = [
        "primal", "origin forme", "therian forme", "incarnate forme",
        "attack forme", "defense forme", "speed forme", "normal forme",
        "blade forme",  # Aegislash — Shield is base, Blade is alternate
        "black kyurem", "white kyurem",
        "crowned sword", "crowned shield",
        "eternamax", "ultra necrozma",
        "dawn wings", "dusk mane",
        "sunny form", "rainy form", "snowy form",
        "complete forme", "10% forme", "50% forme",
        "white-striped", "blue-striped",
        "blue plumage", "yellow plumage", "white plumage",
        "large size", "small size", "super size",
        "zen mode", "galar zen",
        "pirouette forme", "aria forme",
        "resolute forme", "ordinary forme",
        "school form", "solo form",
        "dusk form", "midnight form", "midday form",
        "noice face", "ice face",
        "hangry mode", "full belly",
        "gorging form", "gulping form",
        "low key", "amped form",
        "sandy cloak", "trash cloak", "plant cloak",
        "sky forme", "land forme",
    ]

    if any(kw in name_lower for kw in alternate_keywords):
        return "Alternate"

    return "Base"

def _is_legendary(name: str) -> bool:
    """Check if Pokémon is in the curated Legendary/Mythical list."""
    # Strip form suffixes before matching — e.g. "Mewtwo Mega Mewtwo X" -> "Mewtwo"
    base_name = name.split()[0] if " " in name else name
    # Handle hyphenated names like "Ho-Oh", "Tapu Koko", "Type: Null"
    return any(legendary in name for legendary in LEGENDARY_NAMES)

def _get_legendary_category(name: str) -> str:
    """
    Classify legendary sub-category for richer analysis.

    Returns one of: 'None', 'Traditional', 'Mythical', 'Ultra Beast', 'Paradox'.
    Order matters — we check more specific categories first.
    """
    # Paradox — match by exact containment since names are distinctive
    if any(p in name for p in PARADOX_POKEMON):
        return "Paradox"

    # Ultra Beasts
    if any(ub in name for ub in ULTRA_BEASTS):
        return "Ultra Beast"

    # Mythical
    if any(m in name for m in MYTHICAL_NAMES):
        return "Mythical"

    # Traditional legendary — falls through to existing logic
    if _is_legendary(name):
        return "Traditional"

    return "None"


def load_raw_pokemon() -> pd.DataFrame:
    """Load the raw Pokémon CSV with standardised column names."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Expected Pokémon CSV at {RAW_DATA_PATH}. "
            "Download from Kaggle and place in data/raw/pokemon.csv."
        )

    df = pd.read_csv(RAW_DATA_PATH)

    # Standardise column names — lowercase snake_case
    df = df.rename(columns={
        "ID": "pokedex_id",
        "Name": "name",
        "Total": "bst",
        "HP": "hp",
        "Attack": "attack",
        "Defense": "defense",
        "SpAtk": "sp_attack",
        "SpDef": "sp_defense",
        "Speed": "speed",
        "Type1": "type_1",
        "Type2": "type_2",
        "Height": "height",
        "Weight": "weight",
    })

    return df


def load_pokemon() -> pd.DataFrame:
    """
    Load Pokémon data with derived columns added.

    Adds:
    - generation: int (1-9), derived from pokedex_id
    - form_type: str ('Base', 'Mega', 'Regional', 'Gigantamax')
    - is_legendary: bool
    - type_2: filled with 'None' where missing (for single-type Pokémon)
    """
    df = load_raw_pokemon()

    # Derive generation from ID
    df["generation"] = df["pokedex_id"].apply(_get_generation)

    # Classify form type
    df["form_type"] = df["name"].apply(_get_form_type)

    # Mark Legendaries
    df["is_legendary"] = df["name"].apply(_is_legendary)

    # Sub-category for legendary analysis (Traditional / Mythical / Ultra Beast / Paradox / None)
    df["legendary_category"] = df["name"].apply(_get_legendary_category)

    # Handle single-type Pokémon — 'None' is cleaner than NaN for categorical analysis
    df["type_2"] = df["type_2"].fillna("None")

    # Patch the one null weight (we saw this in exploration)
    df["weight"] = df["weight"].fillna(df["weight"].median())

    return df

PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "pokemon_clean.parquet"


def save_processed() -> Path:
    """Run full load pipeline and save to processed/. Returns output path."""
    df = load_pokemon()
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_DATA_PATH, index=False)
    return PROCESSED_DATA_PATH


def load_processed() -> pd.DataFrame:
    """Load pre-computed processed data. Falls back to live load if missing."""
    if PROCESSED_DATA_PATH.exists():
        return pd.read_parquet(PROCESSED_DATA_PATH)
    return load_pokemon()

FEATURED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "pokemon_featured.parquet"


def save_featured() -> Path:
    """Run full pipeline (clean + features) and save to processed/."""
    from src.feature_engineering import build_feature_frame
    df = build_feature_frame(load_pokemon())
    FEATURED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FEATURED_DATA_PATH, index=False)
    return FEATURED_DATA_PATH


def load_featured() -> pd.DataFrame:
    """Load feature-engineered data for EDA/modelling."""
    if FEATURED_DATA_PATH.exists():
        return pd.read_parquet(FEATURED_DATA_PATH)
    from src.feature_engineering import build_feature_frame
    return build_feature_frame(load_pokemon())

TIERS_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "smogon.csv"

# Viability threshold — tiers considered "competitively viable"
VIABLE_TIERS = {"AG", "Uber", "OU", "UU", "BL"}

# Manual name mapping for Pokémon whose names don't normalise cleanly
# Maps main-dataset name -> Smogon name_key
# Used for Rotom forms, gender forms, and "canonical forme" Pokémon
MANUAL_NAME_MAP = {
    # Rotom forms — Smogon uses appliance abbreviations
    "Rotom Wash Rotom":   "rotom-w",
    "Rotom Heat Rotom":   "rotom-h",
    "Rotom Frost Rotom":  "rotom-f",
    "Rotom Fan Rotom":    "rotom-s",   # Smogon: rotom-s (spin)
    "Rotom Mow Rotom":    "rotom-c",   # Smogon: rotom-c (cut)

    # Hoopa — Smogon splits confined/unbound
    "Hoopa Hoopa Confined": "hoopa",
    "Hoopa Hoopa Unbound":  "hoopa-u",

    # Keldeo — both forms map to base keldeo tier
    "Keldeo Ordinary Form":  "keldeo",
    "Keldeo Resolute Form":  "keldeo",

    # Darmanitan — Standard Mode is the base form
    "Darmanitan Standard Mode": "darmanitan",

    # Basculin — Red-Striped is the base form in Smogon
    "Basculin Red-Striped Form": "basculin",

    # Gourgeist — Average Size is the base form
    "Gourgeist Average Size": "gourgeist",

    # Meowstic — gender forms tiered separately in Smogon
    "Meowstic Male":   "meowstic-m",
    "Meowstic Female": "meowstic-f",

    # Shaymin — Land Forme is the base
    "Shaymin Land Forme": "shaymin",
    "Shaymin Sky Forme":  "shaymin-s",

    # Zygarde — 50% is the canonical base form
    "Zygarde 50% Forme": "zygarde",

    # Meloetta — Aria is the base form
    "Meloetta Aria Forme": "meloetta",

    # Deoxys formes — each has its own Smogon tier
    "Deoxys Normal Forme":  "deoxys",
    "Deoxys Attack Forme":  "deoxys-a",
    "Deoxys Defense Forme": "deoxys-d",
    "Deoxys Speed Forme":   "deoxys-s",

    # Landorus/Tornadus/Thundurus — Incarnate and Therian tiered separately
    "Landorus Incarnate Forme":  "landorus",
    "Landorus Therian Forme":    "landorus-t",
    "Tornadus Incarnate Forme":  "tornadus",
    "Tornadus Therian Forme":    "tornadus-t",
    "Thundurus Incarnate Forme": "thundurus",
    "Thundurus Therian Forme":   "thundurus-t",

    # Kyurem fusions
    "Kyurem White Kyurem": "kyurem-w",
    "Kyurem Black Kyurem": "kyurem-b",

    # Giratina Origin — Uber, distinct from Altered Forme
    "Giratina Origin Forme": "giratina-o",

    # Meloetta Pirouette — tiered separately from Aria
    "Meloetta Pirouette Forme": "meloetta-p",

    # Gourgeist size variants — all PU but Smogon tracks separately
    "Gourgeist Small Size":   "gourgeist-small",
    "Gourgeist Large Size":   "gourgeist-large",
    "Gourgeist Super Size":   "gourgeist-super",

    # Wormadam cloaks — all PU but Smogon tracks separately
    "Wormadam Plant Cloak": "wormadam-plant",
    "Wormadam Sandy Cloak": "wormadam-sandy",
    "Wormadam Trash Cloak": "wormadam-trash",
}

def _normalise_name_for_join(name: str) -> str:
    """
    Normalise Pokémon names to a simple lowercase hyphenated key for joining.

    Checks MANUAL_NAME_MAP first for edge cases, then falls back to
    automated normalisation for standard names.
    """
    # Check manual map first — handles Rotom forms, gender forms, etc.
    if name in MANUAL_NAME_MAP:
        return MANUAL_NAME_MAP[name]

    name = name.strip()

    # Smogon dataset prefixes forms at the start: "Mega X", "Primal X"
    smogon_prefixes = [
        "Mega ", "Primal ", "Galarian ", "Alolan ", "Hisuian ",
        "Paldean ", "Origin ", "Black ", "White ",
    ]
    for prefix in smogon_prefixes:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break

    # Main dataset suffixes forms — take only what's before the form keyword
    form_keywords = [
        " Mega ", " Primal ", " Galarian ", " Alolan ", " Hisuian ",
        " Paldean ", " Origin ", " Therian ", " Incarnate ",
        " Black ", " White ", " Normal ", " Attack ", " Defense ",
        " Speed ", " Sandy ", " Trash ", " Plant ", " Altered ",
        " Shield ", " Blade ", " Crowned ", " Eternamax ",
        " Dawn ", " Dusk ", " Ultra ", " Sunny ", " Rainy ", " Snowy ",
    ]
    for kw in form_keywords:
        if kw in name:
            name = name.split(kw)[0]
            break

    # Lowercase and clean punctuation
    name = name.lower()
    name = name.replace("♀", "-f").replace("♂", "-m")
    name = name.replace("'", "").replace(".", "").replace(":", "")
    name = name.replace(" ", "-")

    return name.strip("-")

def load_tiers() -> pd.DataFrame:
    """
    Load and clean the Smogon tiers dataset.

    Returns a DataFrame with normalised name keys and binary viability labels.
    Excludes Mega and Primal forms — we join on base form names only.
    """
    if not TIERS_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Expected Smogon tiers CSV at {TIERS_DATA_PATH}. "
            "Download from Kaggle and place in data/raw/smogon_tiers.csv."
        )

    df = pd.read_csv(TIERS_DATA_PATH)

    # Rename columns to snake_case
    df = df.rename(columns={
        "X.": "pokedex_id",
        "Name": "name",
        "Type.1": "type_1",
        "Type.2": "type_2",
        "Total": "bst",
        "HP": "hp",
        "Attack": "attack",
        "Defense": "defense",
        "Sp..Atk": "sp_attack",
        "Sp..Def": "sp_defense",
        "Speed": "speed",
        "Generation": "generation",
        "Legendary": "is_legendary",
        "Mega": "is_mega",
        "Tier": "tier",
    })

    # Exclude Mega and Primal forms from the tiers dataset
    # We want base-form tier labels only — Mega tiers don't apply to base forms
    df = df[df["is_mega"] == False].copy()

    # Add normalised name key
    df["name_key"] = df["name"].apply(_normalise_name_for_join)

    # Drop any rows where name_key is empty (shouldn't happen now, safety net)
    df = df[df["name_key"].str.len() > 0]

    # Drop duplicate name_keys — keep the first occurrence
    # (some Pokémon appear twice due to form variants we didn't fully exclude)
    df = df.drop_duplicates(subset="name_key", keep="first")

    # Add binary viability label
    df["is_viable"] = df["tier"].isin(VIABLE_TIERS)

    return df[["name", "name_key", "tier", "is_viable"]]


def load_pokemon_with_tiers() -> pd.DataFrame:
    """
    Join featured Pokémon data with Smogon tier labels.

    Base forms receive tier labels via name_key matching.
    Alternate forms that have explicit entries in MANUAL_NAME_MAP also
    receive tier labels — these are competitively distinct formes
    (Deoxys-A, Landorus-T, Kyurem-B etc.) that Smogon tiers separately.
    All other non-base forms (Mega, Regional, Gigantamax) remain Untiered.
    """
    df_pokemon = load_featured()
    df_tiers = load_tiers()

    # Add normalised name key to main dataset
    df_pokemon["name_key"] = df_pokemon["name"].apply(_normalise_name_for_join)

    # Join tiers onto Base forms AND Alternate forms that are in MANUAL_NAME_MAP
    # (these are competitively distinct formes with their own Smogon tiers)
    manually_mapped = set(MANUAL_NAME_MAP.keys())
    join_mask = (
        (df_pokemon["form_type"] == "Base") |
        (df_pokemon["name"].isin(manually_mapped))
    )

    df_join = df_pokemon[join_mask].merge(
        df_tiers[["name_key", "tier", "is_viable"]],
        on="name_key",
        how="left",
    )

    df_no_join = df_pokemon[~join_mask].copy()
    df_no_join["tier"] = "Untiered"
    df_no_join["is_viable"] = float("nan")

    # Recombine and restore original order
    df_merged = pd.concat([df_join, df_no_join], ignore_index=True)
    df_merged = df_merged.sort_values("pokedex_id").reset_index(drop=True)

    # Fill remaining untiered
    df_merged["tier"] = df_merged["tier"].fillna("Untiered")

    return df_merged

TIERED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "pokemon_with_tiers.parquet"


def load_tiered() -> pd.DataFrame:
    """Load the joined Pokémon + Smogon tiers dataset."""
    if TIERED_DATA_PATH.exists():
        return pd.read_parquet(TIERED_DATA_PATH)
    return load_pokemon_with_tiers()

EXTENDED_TIERS_PATH = PROJECT_ROOT / "data" / "raw" / "pokemon_data.csv"

# Game version codes to generation numbers
GAME_GEN_MAP = {
    'RB': 1, 'GS': 2, 'RS': 3, 'DP': 4,
    'BW': 5, 'XY': 6, 'SM': 7, 'SS': 8
}


def load_extended_tiers() -> pd.DataFrame:
    """
    Load the extended tiers dataset (covers Gen 8).
    Uses 'oldformats' column as the tier source — 'formats' reflects
    permissibility not placement, so it overcounts Uber significantly.
    Returns name, gen_number, tier, is_viable columns only.
    """
    if not EXTENDED_TIERS_PATH.exists():
        raise FileNotFoundError(
            f"Expected extended tiers CSV at {EXTENDED_TIERS_PATH}."
        )

    df = pd.read_csv(EXTENDED_TIERS_PATH)

    # Map game codes to generation numbers
    df['gen_number'] = df['generation'].map(GAME_GEN_MAP)

    # Use oldformats as tier — more accurate than formats
    df = df.rename(columns={'oldformats': 'tier'})

    # Normalise name for joining
    df['name_key'] = df['name'].apply(_normalise_name_for_join)

    # Add viability label — no BL tiers in this dataset so simpler mapping
    df['is_viable'] = df['tier'].isin(VIABLE_TIERS)

    return df[['name', 'name_key', 'gen_number', 'tier', 'is_viable']]


def load_pokemon_with_tiers() -> pd.DataFrame:
    """
    Join featured Pokémon data with Smogon tier labels.

    Priority:
    1. Original Smogon dataset (Gens 1-7, has BL granularity)
    2. Extended dataset (fills Gen 8 gaps)
    3. Untiered for everything else

    Only Base forms and manually-mapped Alternate forms receive tier labels.
    """
    df_pokemon = load_featured()
    df_tiers = load_tiers()
    df_extended = load_extended_tiers()

    # Add normalised name key to main dataset
    df_pokemon['name_key'] = df_pokemon['name'].apply(_normalise_name_for_join)

    # Only join tiers onto Base forms and manually-mapped Alternate forms
    manually_mapped = set(MANUAL_NAME_MAP.keys())
    join_mask = (
        (df_pokemon['form_type'] == 'Base') |
        (df_pokemon['name'].isin(manually_mapped))
    )

    df_join = df_pokemon[join_mask].copy()
    df_no_join = df_pokemon[~join_mask].copy()

    # --- Pass 1: original Smogon dataset (Gens 1-7, higher priority) ---
    df_join = df_join.merge(
        df_tiers[['name_key', 'tier', 'is_viable']],
        on='name_key',
        how='left',
    )

    # --- Pass 2: fill remaining untiered rows from extended dataset ---
    # Only use extended dataset for rows that didn't match in Pass 1
    untiered_mask = df_join['tier'].isna()

    # Get Gen 8 entries from extended dataset only
    # We don't want it overwriting Gens 1-7 where our original is more accurate
    extended_fill = df_extended[df_extended['gen_number'] >= 7][
        ['name_key', 'tier', 'is_viable']
    ].rename(columns={
        'tier': 'tier_ext',
        'is_viable': 'is_viable_ext'
    })

    df_join = df_join.merge(
        extended_fill,
        on='name_key',
        how='left',
    )

    # Fill from extended where original was missing
    df_join.loc[untiered_mask & df_join['tier_ext'].notna(), 'tier'] = \
        df_join.loc[untiered_mask & df_join['tier_ext'].notna(), 'tier_ext']
    df_join.loc[untiered_mask & df_join['is_viable_ext'].notna(), 'is_viable'] = \
        df_join.loc[untiered_mask & df_join['is_viable_ext'].notna(), 'is_viable_ext']

    # Drop helper columns
    df_join = df_join.drop(columns=['tier_ext', 'is_viable_ext'])

    # --- Combine base/mapped forms with non-join forms ---
    df_no_join['tier'] = 'Untiered'
    df_no_join['is_viable'] = float('nan')

    df_merged = pd.concat([df_join, df_no_join], ignore_index=True)
    df_merged = df_merged.sort_values('pokedex_id').reset_index(drop=True)

    # Fill remaining untiered
    df_merged['tier'] = df_merged['tier'].fillna('Untiered')

    return df_merged

if __name__ == "__main__":
    # Quick sanity check when run directly: python src/data_loader.py
    df = load_pokemon()
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"\nGenerations present: {sorted(df['generation'].unique())}")
    print(f"Form types: {df['form_type'].value_counts().to_dict()}")
    print(f"Legendaries: {df['is_legendary'].sum()}")
    print(f"\nSample:\n{df.head()}")