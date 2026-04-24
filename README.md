# PokeMeta Analyser

A machine learning and data analysis project tracking how the Pokemon competitive meta has shifted across all nine generations. Identifies type dominance shifts, legendary disruptions, and predicts competitive viability based on stats, typing and ability scores.

Built with Python, XGBoost, SHAP, Plotly, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.9452-green)

---

## Dashboard

Nine interactive tabs with a Pokemon-themed dark UI, sprite decorations, type badge icons, and Pokeball dividers.

| Tab | Description |
|---|---|
| Home | Project overview, key findings, dashboard navigation guide |
| Generations | Power creep, BST trends, legendary gap across all 9 generations |
| Type Dominance | Type frequency shifts, Fairy disruption heatmap, type reference |
| Legendary Impact | Traditional vs Mythical vs Ultra Beast vs Paradox category analysis |
| Stat Profiles | Competitive role distribution, dual-type prevalence, offensive coverage |
| Viability Predictor | Custom Pokemon input with ability dropdown, SHAP explainability, tier prediction, and type matchup defender |
| Model Insights | SHAP importance, confusion matrix, outlier spotlight |
| Gen 9 Predictions | Model predictions for all 112 Scarlet and Violet Pokemon |
| Power Rankings | Generations ranked by competitive contribution |

---

## Key Findings

| Finding | Detail |
|---|---|
| Power creep is real but subtle | Non-legendary BST median rose 48 points across 9 generations |
| The legendary gap is closing | Peaked at 253 BST points in Gen 6, now just 116 in Gen 9 |
| Fairy disrupted everything | Went from 1.3% of Gen 5 to 17.6% of Gen 6 in one generation |
| Speed matters most after BST | Speed is the 2nd most important SHAP feature for viability |
| Gen 6 produced the best competitive Pokemon | 37.2% of its tiered Pokemon reached viable tiers |
| Abilities matter more than expected | hidden_ability_score is the 11th most important SHAP feature |
| The model is blind to crippling abilities | 27 overrated Pokemon have great stats but abilities like Truant and Slow Start |
| Gen 9 looks strong | 43 of 112 Gen 9 Pokemon predicted viable, with Iron Moth and Iron Valiant at 99%+ |

---

## Models

### Binary Viability Classifier

Predicts whether a Pokemon would be competitively viable (Uber/OU/UU or above).

| Property | Value |
|---|---|
| Algorithm | XGBoost Classifier |
| Training data | Gens 1-7, 565 tiered Pokemon |
| Test data | Gen 8, 84 Pokemon (held out entirely during training) |
| Viability definition | Smogon tier Uber, OU, UU, or BL |
| Decision threshold | 0.35 (optimised for viable class F1) |
| ROC-AUC | 0.9452 |
| Accuracy | 89.3% |
| Viable class F1 | 0.69 |
| Viable class recall | 91% |

### 3-Class Tier Classifier

Predicts whether a Pokemon would land in Top Tier (Uber/OU), Mid Tier (UU/RU), or Low Tier (NU/PU).

| Property | Value |
|---|---|
| Algorithm | XGBoost Multi-class Classifier |
| Classes | Top Tier, Mid Tier, Low Tier |
| Training data | Gens 1-7, 464 tiered Pokemon (ZU excluded) |
| Test data | Gen 8, 24 Pokemon |
| Exact accuracy | 58.3% |
| Objective | multi:softmax |

### Top SHAP Features (Binary Model)

1. BST - by far the strongest predictor
2. Speed - more important than any individual offensive stat
3. Offensive total (Attack + Sp.Atk + Speed)
4. Defensive total (HP + Defense + Sp.Def)
5. HP
6. Number of types (dual-typing helps viability)
7. Attack
8. Defensive bias
9. Has type Rock (negative effect)
10. Defense
11. Hidden ability score (competitive ability quality)
12. Height
13. Has type Steel (positive effect)
14. Has type Water
15. Has type Ice

---

## Project Structure

```
pokemon-meta-tracker/
├── data/
│   ├── raw/
│   │   ├── pokemon.csv                  # Main Pokemon dataset
│   │   ├── smogon.csv                   # Smogon tiers Gens 1-6
│   │   ├── pokemon_data.csv             # Extended tiers Gens 7-8
│   │   ├── pokemon_abilities.csv        # Raw ability data from PokeAPI
│   │   └── pokemon_abilities_scored.csv # Abilities with competitive scores
│   └── processed/
│       ├── pokemon_clean.parquet
│       ├── pokemon_featured.parquet
│       └── pokemon_with_tiers.parquet
├── notebooks/
│   ├── 02_eda_generations.ipynb         # Full EDA and model training
│   └── 03_ability_features.ipynb        # Ability scoring and retraining
├── src/
│   ├── __init__.py
│   ├── data_loader.py                   # Loads, cleans, classifies Pokemon data
│   ├── feature_engineering.py           # Stat ratios, type encoding, ability scores
│   ├── model.py                         # XGBoost training, prediction, SHAP utilities
│   └── visualisations.py                # Eight reusable Plotly chart functions
├── models/
│   ├── xgboost_viability.pkl            # Binary viability model
│   ├── xgboost_tier_classifier.pkl      # 3-class tier model
│   └── shap_summary.png                 # Global SHAP feature importance plot
├── app/
│   └── streamlit_app.py                 # Nine-tab Streamlit dashboard
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Setup

### 1. Clone and create environment

```bash
git clone https://github.com/anirudh-rs/pokemon-meta-tracker.git
cd pokemon-meta-tracker

conda create -n pokemon-meta python=3.11 -y
conda activate pokemon-meta
```

### 2. Install dependencies

```bash
conda install -c conda-forge pandas numpy scikit-learn jupyter matplotlib seaborn -y
pip install xgboost shap plotly streamlit pyarrow
pip install -e .
```

### 3. Verify installs

```bash
python -c "import pandas; print('pandas', pandas.__version__)"
python -c "import xgboost; print('xgboost', xgboost.__version__)"
python -c "import shap; print('shap', shap.__version__)"
python -c "import streamlit; print('streamlit', streamlit.__version__)"
```

---

## Data Sources

Download these CSV files and place them in `data/raw/`:

| File | Source | Description |
|---|---|---|
| `pokemon.csv` | Kaggle - Pokemon Dataset | 1,194 Pokemon, all 9 generations, base stats and typing |
| `smogon.csv` | Kaggle - Smogon Tiers | Smogon competitive tiers, Gens 1-6 |
| `pokemon_data.csv` | Kaggle - Competitive Pokemon | Extended tier data covering Gens 7-8 |

Ability data is fetched automatically from PokeAPI during the notebook run — no manual download needed.

---

## Running the Project

### Generate processed data and train both models

Open `notebooks/02_eda_generations.ipynb` in Jupyter and run all cells. Then open `notebooks/03_ability_features.ipynb` and run all cells. This will:

- Clean and feature-engineer the raw data into parquet files
- Fetch ability data from PokeAPI for all 1,025 base form Pokemon
- Score abilities by competitive impact (-3 to +5)
- Run the full exploratory analysis
- Train the binary XGBoost viability classifier with ability features
- Train the 3-class tier classifier
- Save both models to `models/`
- Save the SHAP summary plot

```bash
jupyter notebook
```

### Launch the dashboard

```bash
streamlit run app/streamlit_app.py
```

Opens at `http://localhost:8501`.

---

## Feature Details

### Ability Scoring System

Abilities are scored from -3 to +5 based on competitive impact:

| Score | Tier | Examples |
|---|---|---|
| 5 | S - Game defining | Speed Boost, Regenerator, Intimidate, Levitate, Drizzle |
| 4 | A - Excellent | Adaptability, Magic Guard, Swift Swim, Prankster, Disguise |
| 3 | B - Good | Technician, Guts, Sheer Force, Moxie, Natural Cure |
| 2 | C - Situational | Static, Flame Body, Rough Skin, Pressure, Filter |
| 1 | D - Weak/Filler | Keen Eye, Run Away, Sand Veil, Illuminate |
| -2 to -3 | Crippling | Truant, Slow Start, Defeatist, Klutz, Stall |

### Viability Predictor
Input any combination of stats, typing, and ability. The model returns:
- Binary verdict (Competitively Viable / Not Viable) with probability
- SHAP waterfall chart explaining which stats drove the prediction
- Predicted tier (Top/Mid/Low) with class probabilities
- 10 most similar real Pokemon by BST
- Type Matchup Defender popup showing weaknesses, resistances, and ability-granted immunities
- Top 5 counters per weakness type

### Gen 9 Inference
The binary model was trained on Gens 1-7 and validated on Gen 8. It has never seen Gen 9 data. The Gen 9 tab runs predictions on all 112 Scarlet and Violet base form Pokemon purely from their stats, typing and ability scores.

### Outlier Spotlight
Surfaces Pokemon where the model strongly disagrees with Smogon:
- Overrated (27 Pokemon) - good stats but crippling or mediocre abilities
- Underrated (1 Pokemon, Dracozolt) - weak stats but broken ability-move combination

### Power Rankings
Ranks all 8 generations by the percentage of their tiered Pokemon that reached viable tiers (UU or above). Gen 6 leads at 37.2% despite being the smallest generation.

---

## Limitations

- Smogon tier data does not cover Gen 9
- The model uses base stats, typing and ability scores - movepool and EV spreads are not captured
- Ability scores are manually curated - some niche abilities may be under or over-valued
- Viability is defined using Smogon OU singles - VGC doubles would require different data
- Gen 8 Power Ranking may be understated due to thinner tier dataset coverage
- 3-class tier classifier accuracy (58.3%) reflects the genuine fuzziness of tier boundaries

---

## Planned Extensions

- Nature recommender - suggest optimal nature for a given stat spread
- Speed tier calculator - show which Pokemon a given Speed stat outspeeds
- Smogon set display - show recommended competitive set for any real Pokemon
- Moveset coverage analyser - given 4 moves, calculate total type coverage
- Pokemon Comparison Tool - side-by-side stat radar charts with SHAP comparison
- BST Budget Optimiser - find optimal stat distribution for a given BST total
- Head-to-Head Matchup - compare two Pokemon with SHAP side by side
- VGC vs Smogon comparison tab

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.11 | Core language |
| pandas, numpy | Data manipulation |
| scikit-learn | Preprocessing and evaluation |
| XGBoost | Classification models |
| SHAP | Model explainability |
| Plotly | Interactive visualisations |
| Streamlit | Dashboard framework |
| pyarrow | Parquet file I/O |
| PokeAPI | Ability data source |
