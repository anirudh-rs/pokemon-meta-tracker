# -*- coding: utf-8 -*-
"""
streamlit_app.py - PokeMeta Analyser Dashboard
Runs with: streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.data_loader import load_featured, load_tiered
from src.visualisations import (
    TYPE_COLOURS,
    plot_bst_by_generation, plot_type_composition_absolute,
    plot_type_frequency_percentage, plot_legendary_gap,
    plot_legendary_gap_median, plot_stat_profiles,
    plot_type_coverage, plot_dual_type_prevalence,
)
from src.model import load_model, load_tier_model, predict_viability, predict_tier, build_feature_row, get_top_shap_features

# -- Page config ---------------------------------------------------------------
st.set_page_config(
    page_title="PokeMeta Analyser",
    page_icon="https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/items/master-ball.png",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -- Sprite helpers ------------------------------------------------------------
SPRITE_BASE = "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon"

def sprite_url(pid, official=False):
    if official:
        return f"{SPRITE_BASE}/other/official-artwork/{pid}.png"
    return f"{SPRITE_BASE}/{pid}.png"

def sprite_img(pid, size=96, official=False):
    url = sprite_url(pid, official)
    return (
        '<img src="' + url + '" width="' + str(size) + '" height="' + str(size) + '" '
        'style="image-rendering:pixelated;object-fit:contain;" '
        'onerror="this.style.display=\'none\'">'
    )

def type_badge_html(type_name, size=20):
    colour = TYPE_COLOURS.get(type_name, "#888888")
    pid    = TYPE_POKEMON.get(type_name, 25)
    surl   = sprite_url(pid)
    return (
        '<span style="background:' + colour + ';border-radius:12px;'
        'padding:2px 8px 2px 4px;display:inline-flex;align-items:center;'
        'gap:4px;font-family:Nunito;font-size:0.75rem;color:white;font-weight:700;">'
        '<img src="' + surl + '" width="' + str(size) + '" height="' + str(size) + '" '
        'style="image-rendering:pixelated;">' + type_name + '</span>'
    )

# -- Reference data ------------------------------------------------------------
GEN_POKEMON = {1:6, 2:157, 3:257, 4:445, 5:635, 6:658, 7:778, 8:887, 9:995}
TYPE_POKEMON = {
    "Normal":143, "Fire":6,   "Water":130, "Electric":26,
    "Grass":3,    "Ice":131,  "Fighting":68, "Poison":94,
    "Ground":445, "Flying":18, "Psychic":65, "Bug":212,
    "Rock":248,   "Ghost":94,  "Dragon":149, "Dark":197,
    "Steel":376,  "Fairy":700,
}

POKEBALL_SVG = (
    '<div style="text-align:center;margin:1rem 0;opacity:0.5;">'
    '<svg width="36" height="36" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">'
    '<path d="M10,50 A40,40 0 0 1 90,50 Z" fill="#CC0000"/>'
    '<path d="M10,50 A40,40 0 0 0 90,50 Z" fill="#EAEAEA"/>'
    '<rect x="10" y="46" width="80" height="8" fill="#333"/>'
    '<circle cx="50" cy="50" r="12" fill="#333"/>'
    '<circle cx="50" cy="50" r="8" fill="#EAEAEA"/>'
    '<circle cx="50" cy="50" r="40" fill="none" stroke="#333" stroke-width="4"/>'
    '</svg></div>'
)

def divider():
    st.markdown(POKEBALL_SVG, unsafe_allow_html=True)

# -- CSS -----------------------------------------------------------------------
def inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=Nunito:wght@400;600;700&display=swap');

.stApp { background-color:#1a1a2e; color:#EAEAEA; }
.block-container { padding-top:0.5rem !important; padding-bottom:2rem; max-width:1200px; }

header[data-testid="stHeader"]  { background:transparent !important; height:0 !important; }
div[data-testid="stToolbar"]    { display:none !important; }
.viewerBadge_container__1QSob   { display:none !important; }
footer                          { display:none !important; }
#MainMenu                       { display:none !important; }

h1,h2,h3 {
    font-family:'Press Start 2P',monospace !important;
    color:#FFCB05 !important;
    text-shadow:2px 2px 0px #CC0000;
    line-height:1.6 !important;
}
h1 { font-size:1.1rem !important; }
h2 { font-size:0.85rem !important; }
h3 { font-size:0.7rem  !important; }
p,li,label,.stMarkdown { font-family:'Nunito',sans-serif !important; color:#EAEAEA !important; }

.stTabs [data-baseweb="tab-list"] {
    background-color:#16213e; border-radius:8px 8px 0 0; gap:3px; padding:5px 5px 0;
}
.stTabs [data-baseweb="tab"] {
    font-family:'Press Start 2P',monospace !important;
    font-size:0.5rem !important;
    color:#A0A0A0 !important;
    background-color:#0f3460 !important;
    border-radius:6px 6px 0 0 !important;
    padding:7px 10px !important;
    border:1px solid #CC0000 !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"],
.stTabs [data-baseweb="tab"][aria-selected="true"] p,
.stTabs [data-baseweb="tab"][aria-selected="true"] span,
.stTabs [data-baseweb="tab"][aria-selected="true"] div {
    color:#FFCB05 !important;
    background-color:#CC0000 !important;
    border-bottom:2px solid #FFCB05 !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background-color:#16213e; border-radius:0 8px 8px 8px;
    padding:1.2rem; border:1px solid #CC0000;
}

[data-testid="stMetric"] {
    background-color:#0f3460; border:1px solid #FFCB05; border-radius:8px; padding:0.8rem;
}
[data-testid="stMetricLabel"] {
    font-family:'Press Start 2P',monospace !important; font-size:0.4rem !important; color:#FFCB05 !important;
}
[data-testid="stMetricValue"] {
    font-family:'Press Start 2P',monospace !important; font-size:0.85rem !important; color:#EAEAEA !important;
}

.stSelectbox>div>div {
    background-color:#0f3460 !important; border:1px solid #CC0000 !important;
    color:#EAEAEA !important; font-family:'Nunito',sans-serif !important;
}

[data-testid="stButton"] button {
    font-family:'Press Start 2P',monospace !important;
    font-size:0.26rem !important;
    padding:0 6px !important;
    width:100% !important;
    height:64px !important; min-height:64px !important; max-height:64px !important;
    line-height:1.4 !important;
    white-space:normal !important;
    word-break:break-word !important;
    overflow:hidden !important;
    display:flex !important; align-items:center !important; justify-content:center !important;
    text-align:center !important;
    background-color:#CC0000 !important;
    color:#FFFFFF !important;
    border:2px solid #FFCB05 !important;
    border-radius:6px !important;
    transition:all 0.15s ease;
}
[data-testid="stButton"] button:hover {
    background-color:#FFCB05 !important;
    color:#CC0000 !important;
    transform:scale(1.04);
}

.stAlert { font-family:'Nunito',sans-serif !important; border-radius:8px !important; }
.stDataFrame { font-family:'Nunito',sans-serif !important; }
.stSpinner>div { border-top-color:#CC0000 !important; }
.stMultiSelect span[data-baseweb="tag"] {
    background-color:#CC0000 !important; font-family:'Nunito',sans-serif !important;
}
::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-track { background:#1a1a2e; }
::-webkit-scrollbar-thumb { background:#CC0000; border-radius:3px; }
::-webkit-scrollbar-thumb:hover { background:#FFCB05; }

@keyframes float {
    0%,100% { transform:translateY(0); }
    50%      { transform:translateY(-10px); }
}
</style>
""", unsafe_allow_html=True)

# -- Plotly theme --------------------------------------------------------------
def tp(fig, h=450):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0f3460",
        font=dict(family="Nunito,sans-serif", color="#EAEAEA"),
        title_font=dict(family="Nunito,sans-serif", color="#FFCB05", size=14),
        legend=dict(bgcolor="rgba(15,52,96,0.8)", bordercolor="#CC0000",
                    borderwidth=1, font=dict(color="#EAEAEA")),
        xaxis=dict(gridcolor="#1a1a2e", zerolinecolor="#CC0000",
                   tickfont=dict(color="#EAEAEA"), title_font=dict(color="#EAEAEA")),
        yaxis=dict(gridcolor="#1a1a2e", zerolinecolor="#CC0000",
                   tickfont=dict(color="#EAEAEA"), title_font=dict(color="#EAEAEA")),
        height=h,
    )
    return fig

# -- Section header ------------------------------------------------------------
def sh(title, sub="", pid=None, pid2=None):
    if pid:
        c1, c2 = st.columns([5, 1])
        with c1:
            st.markdown("### " + title)
            if sub:
                st.markdown(
                    "<p style='color:#A0A0A0;font-size:0.85rem;margin-top:-0.4rem;'>" + sub + "</p>",
                    unsafe_allow_html=True)
        with c2:
            if pid2:
                st.markdown(
                    '<div style="text-align:right;margin-top:-0.4rem;display:flex;gap:4px;justify-content:flex-end;">'
                    + sprite_img(pid, 48) + sprite_img(pid2, 48) +
                    '</div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div style="text-align:right;margin-top:-0.4rem;">'
                    + sprite_img(pid, 64) +
                    '</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown("### " + title)
        if sub:
            st.markdown(
                "<p style='color:#A0A0A0;font-size:0.85rem;margin-top:-0.4rem;'>" + sub + "</p>",
                unsafe_allow_html=True)
    divider()

# -- TAB 1: HOME ---------------------------------------------------------------
def tab_home(df, df_t):
    sh("POKEMETA ANALYSER", "Competitive meta shifts across all 9 generations", pid=25)

    # Key stats
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Total Pokemon",        "1,025")
    with c2: st.metric("Generations",           "9")
    with c3: st.metric("Tiered Pokemon",        "606")
    with c4: st.metric("Competitively Viable",  "168")

    divider()

    cl, cr = st.columns([3, 2])
    with cl:
        st.markdown("## ABOUT THIS PROJECT")
        lines = (
            "<div style='font-family:Nunito,sans-serif;font-size:0.95rem;line-height:1.9;'>"
            "<p>This dashboard tracks how the Pokemon competitive meta has evolved across all nine "
            "generations, from the original 151 in Red and Blue to the 1,025+ of Scarlet and Violet.</p>"
            "<p>Using machine learning and data analysis, it covers:</p>"
            "<ul>"
            "<li><strong style='color:#FFCB05'>Data Pipeline</strong> - 1,194 Pokemon classified "
            "across Base, Mega, Regional and Alternate forms with derived features</li>"
            "<li><strong style='color:#FFCB05'>Exploratory Analysis</strong> - Power creep, type "
            "dominance shifts, legendary impact, stat profiles across all 9 generations</li>"
            "<li><strong style='color:#FFCB05'>Viability Classifier</strong> - XGBoost model trained "
            "on Smogon tiers from Gens 1-7, validated on Gen 8, ROC-AUC 0.9452</li>"
            "<li><strong style='color:#FFCB05'>Tier Classifier</strong> - 3-class model predicting "
            "Top, Mid or Low tier with 58.3% accuracy</li>"
            "<li><strong style='color:#FFCB05'>Gen 9 Inference</strong> - Model predictions for 112 "
            "Scarlet and Violet Pokemon with no official tier data yet</li>"
            "<li><strong style='color:#FFCB05'>Power Rankings</strong> - Generations ranked by "
            "competitive contribution. Gen 6 leads at 37.2% viable</li>"
            "</ul>"
            "</div>"
        )
        st.markdown(lines, unsafe_allow_html=True)

    with cr:
        hero_url = sprite_url(150, official=True)
        st.markdown(
            '<div style="text-align:center;padding:0.5rem;">'
            '<img src="' + hero_url + '" '
            'style="width:200px;height:200px;object-fit:contain;'
            'filter:drop-shadow(0 0 18px rgba(204,0,0,0.6));'
            'animation:float 3s ease-in-out infinite;" '
            'onerror="this.style.display=\'none\'">'
            '</div>',
            unsafe_allow_html=True)

        st.markdown("## KEY FINDINGS")
        for emoji, title, desc in [
            ("📈", "Power Creep",        "Non-legendary BST median rose 48 pts across 9 gens"),
            ("⚔️", "Gap Closing",        "Legendary gap peaked Gen 6 at 253 BST, now 116"),
            ("🧚", "Fairy Disruption",   "Went from 1.3% of Gen 5 to 17.6% of Gen 6"),
            ("⚡", "Speed Dominates",    "2nd most important SHAP feature after BST"),
            ("🏆", "Gen 6 Leads",        "37.2% of Gen 6 Pokemon are competitively viable"),
            ("🤖", "Gen 9 Predicted",    "43 of 112 Gen 9 Pokemon predicted viable by the model"),
            ("🔍", "Model Blind Spot",   "27 overrated Pokemon - stats look strong, abilities let them down"),
        ]:
            st.markdown(
                '<div style="background:#0f3460;border-left:3px solid #CC0000;'
                'padding:0.4rem 0.7rem;margin-bottom:0.4rem;border-radius:0 6px 6px 0;">'
                '<span style="font-size:0.85rem">' + emoji + '</span>'
                '<strong style="color:#FFCB05;font-family:Nunito;font-size:0.85rem;"> '
                + title + '</strong><br>'
                '<span style="color:#A0A0A0;font-size:0.78rem;font-family:Nunito;">'
                + desc + '</span>'
                '</div>',
                unsafe_allow_html=True)

    divider()

    # Dashboard guide
    st.markdown("## WHAT'S INSIDE")
    tabs_info = [
        ("📊", "GENERATIONS",          "Power creep, BST trends and legendary gap across all 9 gens"),
        ("🌊", "TYPE DOMINANCE",        "How type frequencies shifted, including Fairy's Gen 6 disruption"),
        ("🏆", "LEGENDARY IMPACT",      "How legendaries disrupted competitive balance per generation"),
        ("⚔️", "STAT PROFILES",         "Physical Sweepers, Special Walls and more across generations"),
        ("🤖", "VIABILITY PREDICTOR",   "Input any stats and typing, get a viability prediction with SHAP"),
        ("🔬", "MODEL INSIGHTS",        "SHAP importance, confusion matrix, and outlier spotlight"),
        ("🌟", "GEN 9 PREDICTIONS",     "Model predictions for all 112 Scarlet and Violet Pokemon"),
        ("🎖️", "POWER RANKINGS",        "Generations ranked by competitive contribution"),
    ]

    col1, col2 = st.columns(2)
    for i, (icon, title, desc) in enumerate(tabs_info):
        with col1 if i % 2 == 0 else col2:
            st.markdown(
                '<div style="background:#0f3460;border:1px solid #CC0000;'
                'padding:0.7rem;border-radius:8px;margin-bottom:0.5rem;'
                'display:flex;gap:0.7rem;align-items:flex-start;">'
                '<span style="font-size:1.1rem;">' + icon + '</span>'
                '<div>'
                '<p style="font-family:Press Start 2P;font-size:0.5rem;'
                'color:#FFCB05;margin:0 0 4px;">' + title + '</p>'
                '<p style="font-family:Nunito;font-size:0.8rem;'
                'color:#A0A0A0;margin:0;">' + desc + '</p>'
                '</div></div>',
                unsafe_allow_html=True)

    divider()

    # Data sources
    st.markdown("## DATA SOURCES")
    for col, (t, b) in zip(st.columns(3), [
        ("Pokemon Dataset",   "1,194 Pokemon across all 9 gens<br>Base stats, types, forms"),
        ("Smogon Tiers",      "606 tiered Pokemon across Gens 1-8<br>OU, UU, Uber, RU, NU, PU"),
        ("XGBoost Models",    "Binary: ROC-AUC 0.9452<br>Tier classifier: 58.3% accuracy"),
    ]):
        with col:
            st.markdown(
                '<div style="background:#0f3460;border:1px solid #FFCB05;'
                'padding:0.9rem;border-radius:8px;text-align:center;">'
                '<p style="color:#FFCB05;font-family:Nunito;font-weight:700;margin-bottom:4px;">'
                + t + '</p>'
                '<p style="color:#A0A0A0;font-size:0.82rem;font-family:Nunito;margin:0;">'
                + b + '</p>'
                '</div>',
                unsafe_allow_html=True)
# -- TAB 2: GENERATION OVERVIEW ------------------------------------------------
def tab_generations(df):
    sh("GENERATION OVERVIEW", "How has Pokemon power evolved across generations?", pid=172)

    c1, c2 = st.columns(2)
    with c1:
        gen_range = st.select_slider("Generation Range", options=list(range(1, 10)), value=(1, 9))
    with c2:
        form_filter = st.selectbox("Form Filter",
                                   ["Base forms only", "Include Megas", "Include all forms"])

    df_f = df.copy()
    if form_filter == "Base forms only":
        df_f = df_f[df_f['form_type'] == 'Base']
    elif form_filter == "Include Megas":
        df_f = df_f[df_f['form_type'].isin(['Base', 'Mega'])]
    df_f = df_f[df_f['generation'].between(*gen_range)]

    cl, cr = st.columns(2)
    with cl:
        st.plotly_chart(tp(plot_bst_by_generation(df_f, base_forms_only=False)),
                        use_container_width=True, key="gen_bst")
    with cr:
        st.plotly_chart(tp(plot_legendary_gap_median(df_f, base_forms_only=False)),
                        use_container_width=True, key="gen_gap")

    divider()
    st.markdown("### GENERATION SUMMARY")

    s = (df_f[df_f['form_type'] == 'Base']
         .groupby('generation')
         .agg(total=('name','count'), avg_bst=('bst','mean'),
              median_bst=('bst','median'), legendaries=('is_legendary','sum'),
              dual=('num_types', lambda x: (x==2).mean()*100))
         .round(1).reset_index())
    s.columns = ['Gen','Total','Avg BST','Med BST','Legs','Dual%']

    rows = ""
    for _, r in s.iterrows():
        g  = int(r['Gen'])
        bc = "#4CAF50" if r['Avg BST'] > 420 else "#EAEAEA"
        rows += (
            "<tr style='border-bottom:1px solid #0f3460;'>"
            "<td style='padding:5px;text-align:center;'>" + sprite_img(GEN_POKEMON.get(g, 25), 36) + "</td>"
            "<td style='padding:5px;text-align:center;font-family:Press Start 2P;"
            "font-size:0.5rem;color:#FFCB05;'>" + str(g) + "</td>"
            "<td style='padding:5px;text-align:center;'>" + str(int(r['Total'])) + "</td>"
            "<td style='padding:5px;text-align:center;color:" + bc + ";'>" + str(r['Avg BST']) + "</td>"
            "<td style='padding:5px;text-align:center;'>" + str(r['Med BST']) + "</td>"
            "<td style='padding:5px;text-align:center;color:#F95587;'>" + str(int(r['Legs'])) + "</td>"
            "<td style='padding:5px;text-align:center;'>" + str(r['Dual%']) + "%</td>"
            "</tr>"
        )

    headers = ['', 'GEN', 'COUNT', 'AVG BST', 'MED BST', 'LEGS', 'DUAL%']
    ths = "".join(
        '<th style="padding:7px;font-family:Press Start 2P;font-size:0.5rem;color:#FFCB05;">'
        + h + '</th>' for h in headers)
    st.markdown(
        '<table style="width:100%;border-collapse:collapse;background:#0f3460;'
        'border-radius:8px;overflow:hidden;font-family:Nunito,sans-serif;color:#EAEAEA;">'
        '<thead><tr style="background:#CC0000;">' + ths + '</tr></thead>'
        '<tbody>' + rows + '</tbody></table>',
        unsafe_allow_html=True)

# -- TAB 3: TYPE DOMINANCE -----------------------------------------------------
def tab_types(df):
    sh("TYPE DOMINANCE", "How has type composition shifted across generations?", pid=311, pid2=312)

    c1, c2 = st.columns(2)
    with c1:
        view = st.radio("View Mode", ["Absolute Counts", "% of Generation"], horizontal=True)
    with c2:
        sel = st.multiselect("Highlight Types",
                             options=list(TYPE_COLOURS.keys())[:-1], default=[])

    db = df[df['form_type'] == 'Base'].copy()
    fig = (plot_type_composition_absolute if view == "Absolute Counts"
           else plot_type_frequency_percentage)(db, base_forms_only=False)
    if sel:
        fig.for_each_trace(
            lambda t: t.update(visible=True) if t.name in sel else t.update(visible='legendonly'))
    st.plotly_chart(tp(fig, 550), use_container_width=True, key="type_main")

    divider()
    st.markdown("### TYPE REFERENCE")
    badges = '<div style="display:flex;flex-wrap:wrap;gap:5px;margin-bottom:0.8rem;">'
    for t in list(TYPE_COLOURS.keys())[:-1]:
        badges += type_badge_html(t, 18)
    st.markdown(badges + '</div>', unsafe_allow_html=True)

    st.markdown("### TYPE COUNT HEATMAP")
    tc = [c for c in db.columns if c.startswith('has_type_')]
    gt = db.groupby('generation')[tc].sum().rename(
        columns=lambda c: c.replace('has_type_', ''))
    fig_h = px.imshow(gt.T, color_continuous_scale='YlOrRd',
                      title="Type Count by Generation",
                      labels=dict(x="Generation", y="Type", color="Count"),
                      aspect="auto")
    st.plotly_chart(tp(fig_h, 500), use_container_width=True, key="type_heat")

# -- TAB 4: LEGENDARY IMPACT ---------------------------------------------------
def tab_legendary(df):
    sh("LEGENDARY IMPACT",
       "How have legendary Pokemon disrupted competitive balance?", pid=417)

    db = df[df['form_type'] == 'Base'].copy()
    c1, c2 = st.columns(2)
    with c1:
        f1 = plot_legendary_gap(db, base_forms_only=False)
        f1.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.2,
                                     xanchor="center", x=0.5, font=dict(size=10)))
        st.plotly_chart(tp(f1), use_container_width=True, key="leg_gap")
    with c2:
        st.plotly_chart(tp(plot_legendary_gap_median(db, base_forms_only=False)),
                        use_container_width=True, key="leg_med")

    divider()
    st.markdown("### CATEGORY DEEP DIVE")
    cat = st.selectbox("Legendary Category",
                       ["Traditional", "Mythical", "Ultra Beast", "Paradox"])
    cdf = db[db['legendary_category'] == cat]
    if len(cdf):
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Count",    str(len(cdf)))
        with c2: st.metric("Mean BST", str(round(cdf['bst'].mean())))
        with c3: st.metric("Max BST",  str(int(cdf['bst'].max())))
        st.dataframe(
            cdf.nlargest(10, 'bst')[
                ['name','generation','type_1','type_2','bst',
                 'hp','attack','defense','sp_attack','sp_defense','speed']
            ].reset_index(drop=True),
            use_container_width=True)
    else:
        st.info("No " + cat + " Pokemon in current dataset.")

# -- TAB 5: STAT PROFILES ------------------------------------------------------
def tab_profiles(df):
    sh("STAT PROFILES",
       "Are modern Pokemon more specialised than earlier generations?", pid=587)

    c1, c2 = st.columns(2)
    with c1:
        inc = st.checkbox("Include Legendaries", value=False)
    with c2:
        gr = st.select_slider("Generation Range", options=list(range(1, 10)),
                              value=(1, 9), key="pg")
    df_f = df[(df['form_type'] == 'Base') & (df['generation'].between(*gr))].copy()

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            tp(plot_stat_profiles(df_f, base_forms_only=False, legendaries=inc)),
            use_container_width=True, key="pp")
    with c2:
        st.plotly_chart(
            tp(plot_dual_type_prevalence(df_f, base_forms_only=False, legendaries=inc)),
            use_container_width=True, key="pd")
    st.plotly_chart(
        tp(plot_type_coverage(df_f, base_forms_only=False, legendaries=inc), 400),
        use_container_width=True, key="pc")

# Full defensive type chart
# DEFENSIVE_CHART[defending_type] = {attacking_type: multiplier}
DEFENSIVE_CHART = {
    "Normal":   {"Fighting":2, "Ghost":0},
    "Fire":     {"Water":2,"Ground":2,"Rock":2,"Fire":0.5,"Grass":0.5,"Ice":0.5,"Bug":0.5,"Steel":0.5,"Fairy":0.5},
    "Water":    {"Electric":2,"Grass":2,"Water":0.5,"Ice":0.5,"Steel":0.5,"Fire":0.5},
    "Electric": {"Ground":2,"Electric":0.5,"Flying":0.5,"Steel":0.5},
    "Grass":    {"Fire":2,"Ice":2,"Poison":2,"Flying":2,"Bug":2,"Water":0.5,"Electric":0.5,"Grass":0.5,"Ground":0.5},
    "Ice":      {"Fire":2,"Fighting":2,"Rock":2,"Steel":2,"Ice":0.5},
    "Fighting": {"Flying":2,"Psychic":2,"Fairy":2,"Bug":0.5,"Rock":0.5,"Dark":0.5},
    "Poison":   {"Ground":2,"Psychic":2,"Fighting":0.5,"Poison":0.5,"Bug":0.5,"Grass":0.5,"Fairy":0.5},
    "Ground":   {"Water":2,"Grass":2,"Ice":2,"Electric":0,"Poison":0.5,"Rock":0.5},
    "Flying":   {"Electric":2,"Ice":2,"Rock":2,"Ground":0,"Fighting":0.5,"Bug":0.5,"Grass":0.5},
    "Psychic":  {"Bug":2,"Ghost":2,"Dark":2,"Fighting":0.5,"Psychic":0.5},
    "Bug":      {"Fire":2,"Flying":2,"Rock":2,"Fighting":0.5,"Ground":0.5,"Grass":0.5},
    "Rock":     {"Water":2,"Grass":2,"Fighting":2,"Ground":2,"Steel":2,"Normal":0.5,"Fire":0.5,"Poison":0.5,"Flying":0.5},
    "Ghost":    {"Ghost":2,"Dark":2,"Normal":0,"Fighting":0,"Poison":0.5,"Bug":0.5},
    "Dragon":   {"Ice":2,"Dragon":2,"Fairy":2,"Fire":0.5,"Water":0.5,"Electric":0.5,"Grass":0.5},
    "Dark":     {"Fighting":2,"Bug":2,"Fairy":2,"Ghost":0.5,"Dark":0.5,"Psychic":0},
    "Steel":    {"Fire":2,"Fighting":2,"Ground":2,"Normal":0.5,"Grass":0.5,"Ice":0.5,"Flying":0.5,"Psychic":0.5,"Bug":0.5,"Rock":0.5,"Dragon":0.5,"Steel":0.5,"Fairy":0.5,"Poison":0},
    "Fairy":    {"Poison":2,"Steel":2,"Fighting":0.5,"Bug":0.5,"Dark":0.5,"Dragon":0},
}

def compute_defensive_matchups(type_1, type_2="None"):
    """
    Returns a dict of {attacking_type: final_multiplier} for a given typing.
    Combines both types by multiplying their individual multipliers.
    """
    all_types = list(DEFENSIVE_CHART.keys())
    result = {}
    for atk in all_types:
        m1 = DEFENSIVE_CHART.get(type_1, {}).get(atk, 1)
        m2 = DEFENSIVE_CHART.get(type_2, {}).get(atk, 1) if type_2 != "None" else 1
        final = m1 * m2
        if final != 1:
            result[atk] = final
    return result

# Abilities that grant immunity to a specific type
ABILITY_IMMUNITIES = {
    'Levitate':     'Ground',
    'Flash Fire':   'Fire',
    'Water Absorb': 'Water',
    'Volt Absorb':  'Electric',
    'Storm Drain':  'Water',
    'Lightning Rod':'Electric',
    'Sap Sipper':   'Grass',
    'Motor Drive':  'Electric',
    'Dry Skin':     'Water',
    'Earth Eater':    'Ground',
    'Well-Baked Body':'Fire',
    'Wonder Guard': None,  # special case - handled separately
}

@st.dialog("WHAT BEATS IT!!", width="large")
def show_type_matchup_dialog(type_1, type_2, df, ability='Unknown'):
    t2_display = " / " + type_2 if type_2 != "None" else ""
    ab_display = (" + " + ability) if ability and ability != 'Unknown' else ""

    st.markdown(
        '<p style="font-family:Press Start 2P;font-size:0.5rem;color:#FFCB05;">'
        'DEFENSIVE PROFILE: ' + type_1.upper() + t2_display.upper() + ab_display.upper() + '</p>',
        unsafe_allow_html=True)

    matchups = compute_defensive_matchups(type_1, type_2)

    # Apply ability immunity override
    immune_type = ABILITY_IMMUNITIES.get(ability)
    if immune_type and immune_type in matchups:
        del matchups[immune_type]

    # Show ability immunity note if applicable
    if immune_type:
        tc = TYPE_COLOURS.get(immune_type, "#888")
        st.markdown(
            '<div style="background:#1b5e20;border-left:3px solid #4CAF50;'
            'padding:0.5rem 0.8rem;border-radius:0 6px 6px 0;margin-bottom:0.8rem;">'
            '<p style="font-family:Press Start 2P;font-size:0.5rem;'
            'color:#4CAF50;margin:0 0 3px;">ABILITY IMMUNITY</p>'
            '<p style="font-family:Nunito;font-size:0.82rem;color:#EAEAEA;margin:0;">'
            + ability + ' grants immunity to '
            '<span style="background:' + tc + ';color:white;padding:1px 8px;'
            'border-radius:8px;font-weight:700;">' + immune_type + '</span>'
            ' — removing it from weaknesses.</p>'
            '</div>',
            unsafe_allow_html=True)

    # Sort into categories
    weak4x   = {t: m for t, m in matchups.items() if m == 4}
    weak2x   = {t: m for t, m in matchups.items() if m == 2}
    resist2x = {t: m for t, m in matchups.items() if m == 0.5}
    resist4x = {t: m for t, m in matchups.items() if m == 0.25}
    immune   = {t: m for t, m in matchups.items() if m == 0}

    def badge_row(label, types_dict, colour):
        if not types_dict:
            return
        badges = '<div style="margin-bottom:0.7rem;">'
        badges += '<span style="font-family:Press Start 2P;font-size:0.45rem;color:' + colour + ';">' + label + '</span><br>'
        badges += '<div style="display:flex;flex-wrap:wrap;gap:4px;margin-top:4px;">'
        for t in types_dict:
            tc = TYPE_COLOURS.get(t, "#888")
            badges += (
                '<span style="background:' + tc + ';color:white;'
                'font-family:Nunito;font-size:0.75rem;font-weight:700;'
                'padding:2px 10px;border-radius:10px;">' + t + '</span>'
            )
        badges += '</div></div>'
        st.markdown(badges, unsafe_allow_html=True)

    badge_row("4x WEAK",       weak4x,   "#FF4444")
    badge_row("2x WEAK",       weak2x,   "#F44336")
    badge_row("RESISTS 0.5x",  resist2x, "#4CAF50")
    badge_row("RESISTS 0.25x", resist4x, "#00C853")
    badge_row("IMMUNE",        immune,   "#A0A0A0")

    # Add ability immunity to the immune row display
    if immune_type:
        tc = TYPE_COLOURS.get(immune_type, "#888")
        st.markdown(
            '<div style="margin-bottom:0.7rem;">'
            '<span style="font-family:Press Start 2P;font-size:0.45rem;color:#A0A0A0;">'
            'IMMUNE (via ability)</span><br>'
            '<div style="display:flex;flex-wrap:wrap;gap:4px;margin-top:4px;">'
            '<span style="background:' + tc + ';color:white;font-family:Nunito;'
            'font-size:0.75rem;font-weight:700;padding:2px 10px;border-radius:10px;">'
            + immune_type + '</span>'
            '</div></div>',
            unsafe_allow_html=True)

    if not weak4x and not weak2x:
        st.markdown(
            '<p style="font-family:Nunito;color:#4CAF50;font-size:0.9rem;">'
            'No weaknesses! Exceptional defensive typing.</p>',
            unsafe_allow_html=True)

    # Top counters per weakness type
    all_weaknesses = list(weak4x.keys()) + list(weak2x.keys())
    if all_weaknesses:
        st.markdown(
            '<p style="font-family:Press Start 2P;font-size:0.45rem;'
            'color:#FFCB05;margin-top:0.8rem;">TOP COUNTERS BY WEAKNESS TYPE</p>',
            unsafe_allow_html=True)

        base_df = df[
            (df['form_type'] == 'Base') &
            (df['legendary_category'] == 'None')
        ].copy()

        for weak_type in all_weaknesses:
            multiplier = matchups[weak_type]
            label = "4x" if multiplier == 4 else "2x"
            tc    = TYPE_COLOURS.get(weak_type, "#888")

            top5 = base_df[
                (base_df['type_1'] == weak_type) |
                (base_df['type_2'] == weak_type)
            ].nlargest(5, 'bst')

            if len(top5) == 0:
                continue

            st.markdown(
                '<div style="margin-bottom:0.8rem;">'
                '<span style="background:' + tc + ';color:white;'
                'font-family:Nunito;font-size:0.8rem;font-weight:700;'
                'padding:2px 12px;border-radius:10px;">'
                + weak_type + ' (' + label + ')</span>'
                '</div>',
                unsafe_allow_html=True)

            cols = st.columns(5)
            for i, (_, row) in enumerate(top5.iterrows()):
                with cols[i]:
                    t2d  = " / " + row['type_2'] if row['type_2'] != "None" else ""
                    surl = sprite_url(int(row['pokedex_id']), official=False)
                    st.markdown(
                        '<div style="background:#0f3460;border:1px solid ' + tc + ';'
                        'border-radius:8px;padding:0.4rem;text-align:center;">'
                        '<img src="' + surl + '" width="56" height="56" '
                        'style="image-rendering:pixelated;" '
                        'onerror="this.style.display=\'none\'">'
                        '<p style="font-family:Press Start 2P;font-size:0.35rem;'
                        'color:#FFCB05;margin:3px 0 1px;">' + row['name'].upper() + '</p>'
                        '<p style="font-family:Nunito;font-size:0.65rem;'
                        'color:#A0A0A0;margin:0;">' + row['type_1'] + t2d + '</p>'
                        '<p style="font-family:Nunito;font-size:0.7rem;'
                        'color:#EAEAEA;margin:0;">BST ' + str(int(row['bst'])) + '</p>'
                        '</div>',
                        unsafe_allow_html=True)

# -- TAB 6: VIABILITY PREDICTOR ------------------------------------------------

# -- Nature data ---------------------------------------------------------------
NATURES = {
    # (boost_stat, drop_stat, competitive_note)
    "Adamant": ("attack",    "sp_attack",  "Best physical attacker nature. More raw power."),
    "Brave":   ("attack",    "speed",      "Physical attacker for Trick Room. Slowest = fastest in TR."),
    "Lonely":  ("attack",    "defense",    "Physical attacker, sacrifices physical bulk."),
    "Naughty": ("attack",    "sp_defense", "Physical attacker, sacrifices special bulk."),
    "Modest":  ("sp_attack", "attack",     "Best special attacker nature. More raw power."),
    "Quiet":   ("sp_attack", "speed",      "Special attacker for Trick Room."),
    "Mild":    ("sp_attack", "defense",    "Special attacker, sacrifices physical bulk."),
    "Rash":    ("sp_attack", "sp_defense", "Special attacker, sacrifices special bulk."),
    "Jolly":   ("speed",     "sp_attack",  "Physical attacker that needs Speed. Most common on fast sweepers."),
    "Timid":   ("speed",     "attack",     "Special attacker that needs Speed. Most common on fast special sweepers."),
    "Hasty":   ("speed",     "defense",    "Mixed or physical attacker needing Speed. Sacrifices physical bulk."),
    "Naive":   ("speed",     "sp_defense", "Mixed attacker needing Speed. Sacrifices special bulk."),
    "Bold":    ("defense",   "attack",     "Physical wall. Takes hits from physical attackers."),
    "Relaxed": ("defense",   "speed",      "Physical wall for Trick Room. Maximises bulk over Speed."),
    "Impish":  ("defense",   "sp_attack",  "Physical wall that keeps Attack. Most common on defensive Pokemon."),
    "Lax":     ("defense",   "sp_defense", "Physical wall, sacrifices special bulk. Rarely used."),
    "Calm":    ("sp_defense","attack",     "Special wall. Takes hits from special attackers."),
    "Sassy":   ("sp_defense","speed",      "Special wall for Trick Room."),
    "Careful": ("sp_defense","sp_attack",  "Special wall that keeps Attack. Common on defensive Pokemon."),
    "Gentle":  ("sp_defense","defense",    "Special wall, sacrifices physical bulk. Rarely used."),
    "Hardy":   (None, None,               "Neutral - no stat change. Never use competitively."),
    "Docile":  (None, None,               "Neutral - no stat change. Never use competitively."),
    "Serious": (None, None,               "Neutral - no stat change. Never use competitively."),
    "Bashful": (None, None,               "Neutral - no stat change. Never use competitively."),
    "Quirky":  (None, None,               "Neutral - no stat change. Never use competitively."),
}

# -- Speed tier reference data -------------------------------------------------
def build_speed_tiers(df_t):
    """Build speed tier reference from tiered dataset."""
    return (
        df_t[
            (df_t['form_type'] == 'Base') &
            (df_t['is_viable'] == True)
        ][['name','speed','tier','type_1','type_2','pokedex_id']]
        .drop_duplicates(subset='name')
        .sort_values('speed', ascending=False)
        .reset_index(drop=True)
    )

STAT_LABELS = {
    "attack":    "Attack",
    "sp_attack": "Sp.Atk",
    "defense":   "Defense",
    "sp_defense":"Sp.Def",
    "speed":     "Speed",
}

NATURE_GROUPS = {
    "Boosts Attack":    ["Adamant","Brave","Lonely","Naughty"],
    "Boosts Sp.Atk":   ["Modest","Quiet","Mild","Rash"],
    "Boosts Speed":     ["Jolly","Timid","Hasty","Naive"],
    "Boosts Defense":   ["Bold","Relaxed","Impish","Lax"],
    "Boosts Sp.Def":    ["Calm","Sassy","Careful","Gentle"],
    "Neutral (no effect)": ["Hardy","Docile","Serious","Bashful","Quirky"],
}


# Abilities that make Speed-dropping natures viable (speed is boosted by ability)
SPEED_BOOSTING_ABILITIES = {
    'Speed Boost', 'Swift Swim', 'Chlorophyll', 'Sand Rush',
    'Slush Rush', 'Surge Surfer', 'Unburden',
}

# Abilities that amplify Attack (making Attack-boosting nature even more important)
ATTACK_AMPLIFYING_ABILITIES = {
    'Huge Power', 'Pure Power', 'Gorilla Tactics',
}

# Abilities that amplify Sp.Atk
SPATK_AMPLIFYING_ABILITIES = {
    'Solar Power', 'Hadron Engine',
}

def recommend_nature(hp, attack, defense, sp_attack, sp_defense, speed,
                     ability='Unknown', is_crippling=False):
    """
    Recommend the best competitive nature based on stat spread and ability.
    Returns (nature_name, reason_string)
    """
    atk_bias   = attack    - sp_attack
    spatk_bias = sp_attack - attack
    def_bias   = defense   - sp_defense
    spdef_bias = sp_defense - defense
    fast       = speed >= 80

    # Crippling ability — note it in the reason but still give best nature
    crippling_note = " Note: crippling ability limits competitive viability regardless of nature." if is_crippling else ""

    # Speed Boost / weather speed abilities — speed-dropping natures become viable
    if ability in SPEED_BOOSTING_ABILITIES:
        if atk_bias >= 10:
            return "Adamant", (
                ability + " boosts Speed automatically — no need to invest in Speed via nature. "
                "Adamant maximises Attack for more immediate power." + crippling_note
            )
        elif spatk_bias >= 10:
            return "Modest", (
                ability + " boosts Speed automatically — Modest maximises Sp.Atk "
                "instead of wasting the nature on Speed." + crippling_note
            )
        else:
            return "Adamant", (
                ability + " provides Speed — Adamant is the safe default "
                "to maximise physical output." + crippling_note
            )

    # Huge Power / Pure Power — Attack already doubled, maximise further
    if ability in ATTACK_AMPLIFYING_ABILITIES:
        if fast:
            return "Jolly", (
                ability + " doubles Attack — the stat is already massive. "
                "Jolly adds Speed to ensure you move first." + crippling_note
            )
        else:
            return "Adamant", (
                ability + " doubles Attack — Adamant pushes it even higher. "
                "Overwhelming physical power is the strategy." + crippling_note
            )

    # Solar Power / Hadron Engine — Sp.Atk boosted by ability
    if ability in SPATK_AMPLIFYING_ABILITIES:
        if fast:
            return "Timid", (
                ability + " already boosts Sp.Atk — Timid adds Speed "
                "so you move before opponents can KO you." + crippling_note
            )
        else:
            return "Modest", (
                ability + " boosts Sp.Atk — Modest amplifies it further "
                "for maximum special damage." + crippling_note
            )

    # Standard logic below — no special ability interaction
    # Physical sweeper
    if atk_bias >= 20 and fast:
        return "Jolly", (
            "High Attack and decent Speed — Jolly maximises Speed to outpace threats "
            "while dropping unused Sp.Atk." + crippling_note
        )
    if atk_bias >= 20 and speed <= 50:
        return "Brave", (
            "High Attack and very low Speed — Brave suits a Trick Room role "
            "where being slower is an advantage." + crippling_note
        )
    if atk_bias >= 20:
        return "Adamant", (
            "High Attack and low Speed — Adamant squeezes maximum power "
            "from physical moves." + crippling_note
        )

    # Special sweeper
    if spatk_bias >= 20 and fast:
        return "Timid", (
            "High Sp.Atk and decent Speed — Timid maximises Speed "
            "while dropping unused Attack." + crippling_note
        )
    if spatk_bias >= 20 and speed <= 50:
        return "Quiet", (
            "High Sp.Atk and very low Speed — Quiet suits a Trick Room role." + crippling_note
        )
    if spatk_bias >= 20:
        return "Modest", (
            "High Sp.Atk and low Speed — Modest squeezes maximum power "
            "from special moves." + crippling_note
        )

    # Mixed attacker
    if abs(atk_bias) < 20 and fast:
        return "Naive", (
            "Balanced offensive stats with good Speed — Naive boosts Speed "
            "while dropping the less important Sp.Def." + crippling_note
        )
    if abs(atk_bias) < 20:
        return "Hasty", (
            "Balanced offensive stats — Hasty boosts Speed "
            "while dropping Defense." + crippling_note
        )

    # Physical wall
    if def_bias >= 30 and attack < 80:
        return "Bold", (
            "High Defense and low Attack — Bold maximises physical bulk." + crippling_note
        )
    if def_bias >= 20:
        return "Impish", (
            "High Defense — Impish boosts Defense while keeping Attack intact." + crippling_note
        )

    # Special wall
    if spdef_bias >= 30 and attack < 80:
        return "Calm", (
            "High Sp.Def and low Attack — Calm maximises special bulk." + crippling_note
        )
    if spdef_bias >= 20:
        return "Careful", (
            "High Sp.Def — Careful boosts Sp.Def while keeping Attack intact." + crippling_note
        )

    # Default fallback
    if attack >= sp_attack:
        return "Jolly", (
            "Balanced stats leaning physical — Jolly is a safe all-round choice." + crippling_note
        )
    else:
        return "Timid", (
            "Balanced stats leaning special — Timid is a safe all-round choice." + crippling_note
        )


def tab_predictor(df, df_t, model_data, tier_model_data):
    sh("VIABILITY PREDICTOR",
       "Design a Pokemon and see if it would be competitively viable", pid=702)

    TYPES = [
        "Normal","Fire","Water","Electric","Grass","Ice","Fighting","Poison",
        "Ground","Flying","Psychic","Bug","Rock","Ghost","Dragon","Dark","Steel","Fairy",
    ]

    PRESETS = {
        "Pseudo-Leg":  dict(hp=108,attack=130,defense=95, sp_atk=80, sp_def=85, speed=102,type1="Dragon",  type2="None"),
        "Phys Wall":   dict(hp=250,attack=10, defense=230,sp_atk=10, sp_def=230,speed=5,  type1="Normal",  type2="None"),
        "Spec Wall":   dict(hp=255,attack=10, defense=75, sp_atk=10, sp_def=230,speed=5,  type1="Psychic", type2="None"),
        "Cannon":      dict(hp=45, attack=30, defense=35, sp_atk=150,sp_def=65, speed=135,type1="Electric",type2="None"),
        "Sweeper":     dict(hp=75, attack=145,defense=80, sp_atk=30, sp_def=80, speed=110,type1="Fire",    type2="None"),
        "Pivot":       dict(hp=95, attack=90, defense=80, sp_atk=90, sp_def=80, speed=70, type1="Water",   type2="Ground"),
        "Ghost Trick": dict(hp=60, attack=50, defense=60, sp_atk=130,sp_def=90, speed=125,type1="Ghost",   type2="None"),
        "Utility":     dict(hp=110,attack=60, defense=110,sp_atk=60, sp_def=110,speed=50, type1="Ice",     type2="Steel"),
        "Uber":        dict(hp=120,attack=150,defense=100,sp_atk=150,sp_def=100,speed=90, type1="Dragon",  type2="Psychic"),
    }

    # Session state init
    defaults = dict(hp=80, attack=80, defense=80, sp_atk=80, sp_def=80, speed=80,
                    type1="Normal", type2="None", is_legendary=False,
                    height=1.5, weight=50.0)
    for k, v in defaults.items():
        if "pred_" + k not in st.session_state:
            st.session_state["pred_" + k] = v

    # Preset buttons - columns unpacked once outside the loop
    st.markdown(
        "<p style='font-family:Press Start 2P;font-size:0.5rem;color:#FFCB05;"
        "margin-bottom:0.5rem;'>QUICK PRESETS</p>",
        unsafe_allow_html=True)

    p0,p1,p2,p3,p4,p5,p6,p7,p8 = st.columns(9)
    preset_cols = [p0,p1,p2,p3,p4,p5,p6,p7,p8]
    for i, (label, vals) in enumerate(PRESETS.items()):
        with preset_cols[i]:
            if st.button(label, key="preset_" + str(i), use_container_width=True):
                for k, v in vals.items():
                    st.session_state["pred_" + k] = v
                st.session_state["show_results"] = False
                st.rerun()

    divider()

    # Stats inputs
    st.markdown(
        "<p style='font-family:Press Start 2P;font-size:0.5rem;color:#FFCB05;"
        "margin-bottom:0.5rem;'>STATS AND TYPING</p>",
        unsafe_allow_html=True)

    s1,s2,s3,s4,s5,s6 = st.columns(6)
    with s1: st.slider("HP",      1, 255, key="pred_hp")
    with s2: st.slider("Attack",  1, 255, key="pred_attack")
    with s3: st.slider("Defense", 1, 255, key="pred_defense")
    with s4: st.slider("Sp. Atk", 1, 255, key="pred_sp_atk")
    with s5: st.slider("Sp. Def", 1, 255, key="pred_sp_def")
    with s6: st.slider("Speed",   1, 255, key="pred_speed")

    t1,t2,lg,ht,wt,bx = st.columns([2,2,1.2,1.5,1.5,2])
    with t1:
        st.selectbox("Primary Type", TYPES,
                     index=TYPES.index(st.session_state["pred_type1"]),
                     key="pred_type1")
    with t2:
        opts = ["None"] + TYPES
        st.selectbox("Secondary Type", opts,
                     index=opts.index(st.session_state["pred_type2"]),
                     key="pred_type2")
    with lg:
        st.checkbox("Legendary", key="pred_is_legendary")
    with ht:
        st.number_input("Height (m)",  min_value=0.1, max_value=20.0,
                        step=0.1, format="%.1f", key="pred_height")
    with wt:
        st.number_input("Weight (kg)", min_value=0.1, max_value=1000.0,
                        step=0.5, format="%.1f", key="pred_weight")
    with bx:
        bst = sum(st.session_state["pred_" + x]
                  for x in ["hp","attack","defense","sp_atk","sp_def","speed"])
        st.markdown(
            '<div style="background:#0f3460;border:1px solid #FFCB05;padding:0.5rem;'
            'border-radius:6px;text-align:center;margin-top:0.15rem;">'
            '<span style="font-family:Press Start 2P;font-size:0.45rem;color:#A0A0A0;">BST</span><br>'
            '<span style="font-family:Press Start 2P;font-size:0.95rem;color:#FFCB05;">'
            + str(bst) + '</span></div>',
            unsafe_allow_html=True)
        st.markdown("<div style='margin-top:0.5rem;'></div>", unsafe_allow_html=True)
        go_btn = st.button("ANALYSE", use_container_width=True, key="analyse_btn")

    # ── ABILITY / NATURE / SPEED BLOCK ────────────────────────────────────────
    st.markdown(
        "<p style='font-family:Press Start 2P;font-size:0.5rem;color:#FFCB05;"
        "margin-bottom:0.3rem;margin-top:0.5rem;'>ABILITY</p>",
        unsafe_allow_html=True)

    # Grouped ability options with descriptions
    ABILITY_OPTIONS = {
        "-- S Tier (Score 5): Game-defining --": {
            "Speed Boost":      "Raises Speed by 1 each turn. Makes slow Pokemon fast enough to sweep.",
            "Regenerator":      "Heals 1/3 HP on switching out. Enables aggressive pivoting.",
            "Protean":          "Changes type to match the move used. Every hit gets STAB.",
            "Huge Power":       "Doubles Attack stat. Makes otherwise weak Pokemon hit like legendaries.",
            "Pure Power":       "Doubles Attack stat. Same as Huge Power.",
            "Intimidate":       "Lowers opponent's Attack on switch-in. Excellent for bulky teams.",
            "Levitate":         "Immune to Ground-type moves. Removes one of the best coverage types.",
            "Magic Bounce":     "Reflects status moves back at the user. Shuts down hazard setters.",
            "Multiscale":       "Halves damage taken at full HP. Makes tanky Pokemon nearly unkillable early.",
            "Shadow Tag":       "Prevents opponent from switching out. Enables guaranteed KOs.",
            "Arena Trap":       "Prevents grounded opponents from switching. Trapping ability.",
            "Imposter":         "Transforms into the opponent on switch-in. Best on Ditto.",
            "Drizzle":          "Summons permanent Rain on switch-in. Boosts Water moves 50%.",
            "Drought":          "Summons permanent Sun on switch-in. Boosts Fire moves 50%.",
            "Sand Stream":      "Summons permanent Sandstorm. Boosts Rock Sp.Def 50%.",
            "Electric Surge":   "Sets Electric Terrain. Boosts Electric moves and blocks sleep.",
            "Psychic Surge":    "Sets Psychic Terrain. Boosts Psychic moves and blocks priority.",
            "Misty Surge":      "Sets Misty Terrain. Halves Dragon damage and blocks status.",
            "Grassy Surge":     "Sets Grassy Terrain. Boosts Grass moves and heals each turn.",
            "Hadron Engine":    "Sets Electric Terrain and boosts Sp.Atk. Gen 9 Miraidon ability.",
            "Orichalcum Pulse": "Sets Sun and boosts Attack in Sun. Gen 9 Koraidon ability.",
            "Wonder Guard":     "Only super-effective moves can hit. Only on Shedinja.",
        },
        "-- A Tier (Score 4): Excellent --": {
            "Adaptability":     "STAB bonus raised from 1.5x to 2x. Every same-type move hits harder.",
            "Magic Guard":      "Only takes damage from direct attacks. No chip damage ever.",
            "Contrary":         "Stat changes are reversed. Turns debuffs into buffs.",
            "Swift Swim":       "Doubles Speed in Rain. Essential for Rain teams.",
            "Chlorophyll":      "Doubles Speed in Sun. Essential for Sun teams.",
            "Sand Rush":        "Doubles Speed in Sand. Core Sand team ability.",
            "Unburden":         "Doubles Speed after consuming held item. Strong with berries.",
            "Prankster":        "Status moves get +1 priority. Thunder Wave and Tailwind go first.",
            "Beast Boost":      "Raises highest stat after KO. Snowballs quickly.",
            "Tough Claws":      "Boosts contact moves by 30%. Strong on physical attackers.",
            "Pixilate":         "Normal moves become Fairy-type with 20% boost.",
            "Galvanize":        "Normal moves become Electric-type with 20% boost.",
            "Aerilate":         "Normal moves become Flying-type with 20% boost.",
            "Disguise":         "Absorbs one hit with no damage. Free turn for Mimikyu.",
            "Shadow Shield":    "Halves damage at full HP. Legendary-exclusive Multiscale.",
            "Water Bubble":     "Doubles Water move power and halves Fire damage taken.",
            "Transistor":       "Boosts Electric moves by 50%. Gen 8 Regielectki ability.",
            "Dragon's Maw":     "Boosts Dragon moves by 50%. Gen 8 Regidrago ability.",
            "Intrepid Sword":   "Raises Attack by 1 on entry. Zacian ability.",
            "Dauntless Shield": "Raises Defense by 1 on entry. Zamazenta ability.",
        },
        "-- B Tier (Score 3): Good --": {
            "Iron Fist":        "Boosts punching moves by 20%. Strong on Conkeldurr and Hitmonchan.",
            "Sheer Force":      "Removes secondary effects but boosts those moves by 30%.",
            "Guts":             "Boosts Attack 50% when statused. Turns status into a benefit.",
            "Technician":       "Boosts moves with 60 BP or less by 50%. Makes weak moves strong.",
            "Serene Grace":     "Doubles secondary effect chances. 60% flinch chance with Air Slash.",
            "Reckless":         "Boosts recoil moves by 20%. Strong on Entei and Staraptor.",
            "Strong Jaw":       "Boosts biting moves by 50%. Used by Dracovish and Garchomp.",
            "Moxie":            "Raises Attack after KO. Snowballs in offensive play.",
            "Defiant":          "Raises Attack by 2 when stats are lowered. Punishes Intimidate.",
            "Competitive":      "Raises Sp.Atk by 2 when stats are lowered. Special Defiant.",
            "Natural Cure":     "Cures status on switching out. Great on defensive pivots.",
            "Shed Skin":        "33% chance to cure status each turn. Reliable on bulky Pokemon.",
            "Unaware":          "Ignores opponent's stat boosts when attacking or defending.",
            "Thick Fat":        "Halves damage from Fire and Ice moves. Great bulk improvement.",
            "Flash Fire":       "Immune to Fire, boosts own Fire moves when hit by Fire.",
            "Water Absorb":     "Immune to Water, heals 25% HP when hit by Water.",
            "Volt Absorb":      "Immune to Electric, heals 25% HP when hit by Electric.",
            "Sap Sipper":       "Immune to Grass, raises Attack when hit by Grass.",
            "Storm Drain":      "Immune to Water, raises Sp.Atk when hit by Water.",
            "Lightning Rod":    "Immune to Electric, raises Sp.Atk when hit by Electric.",
            "Gooey":            "Lowers opponent's Speed when they make contact.",
            "Stamina":          "Raises Defense by 1 when hit. Mudsdale's ability.",
            "Ice Scales":       "Halves Special damage taken. Frosmoth ability.",
            "Punk Rock":        "Boosts sound moves 30% and halves sound damage taken.",
        },
        "-- C Tier (Score 2): Situational --": {
            "Static":           "30% chance to paralyse on contact. Mild deterrent.",
            "Flame Body":       "30% chance to burn on contact. Useful on walls.",
            "Rough Skin":       "Damages attacker 1/8 HP on contact. Passive chip damage.",
            "Iron Barbs":       "Damages attacker 1/8 HP on contact. Same as Rough Skin.",
            "Synchronize":      "Copies status conditions back to the opponent.",
            "Pressure":         "Opponent's moves use 2 PP instead of 1. Stalling ability.",
            "Filter":           "Reduces super-effective damage by 25%.",
            "Solid Rock":       "Reduces super-effective damage by 25%. Same as Filter.",
            "Analytic":         "Boosts moves by 30% if moving last. Good on slow attackers.",
            "Hustle":           "Raises Attack 50% but lowers physical accuracy 20%.",
            "Simple":           "All stat changes are doubled. Can be good or bad.",
            "Marvel Scale":     "Raises Defense 50% when statused. Milotic ability.",
            "Overcoat":         "Immune to weather damage and powder moves.",
            "Harvest":          "50% chance to restore a berry each turn in Sun.",
            "Scrappy":          "Normal and Fighting moves hit Ghost types.",
            "Cursed Body":      "30% chance to disable the move that hits this Pokemon.",
        },
        "-- D Tier (Score 1): Weak / Filler --": {
            "Keen Eye":         "Prevents accuracy reduction. Rarely matters competitively.",
            "Inner Focus":      "Prevents flinching. Minor benefit.",
            "Own Tempo":        "Prevents confusion. Confusion is rare.",
            "Oblivious":        "Prevents infatuation and Taunt. Very niche.",
            "Early Bird":       "Wakes up from sleep faster. Sleep is uncommon.",
            "Pickup":           "May pick up an item after battle. No in-battle effect.",
            "Shed Skin":        "33% chance to cure status each turn.",
            "Run Away":         "Can always flee from wild battles. No competitive use.",
            "Illuminate":       "Raises encounter rate. No competitive use.",
            "Insomnia":         "Immune to sleep. Niche but occasionally useful.",
            "Vital Spirit":     "Immune to sleep. Same as Insomnia.",
            "Sand Veil":        "Raises evasion in Sand 20%. Banned in many formats.",
            "Snow Cloak":       "Raises evasion in Hail 20%. Banned in many formats.",
            "Leaf Guard":       "Prevents status in Sun. Very situational.",
            "Frisk":            "Reveals opponent's held item on switch-in. Minor info.",
            "Anticipation":     "Warns of super-effective moves. Minor info.",
        },
    }

    CRIPPLING_ABILITY_INFO = {
        "Truant":      "Can only act every other turn. Slaking's curse — 670 BST wasted.",
        "Slow Start":  "Attack and Speed halved for the first 5 turns. Regigigas crippled.",
        "Defeatist":   "Attack and Sp.Atk halved when HP drops below 50%. Archeops ruined.",
        "Klutz":       "Cannot use held items. Removes all item-based strategies.",
        "Stall":       "Always moves last regardless of Speed. Nearly always bad.",
        "Heavy Metal":  "Doubles weight. Makes the Pokemon weaker to Low Kick and Grass Knot.",
    }

    ab1, ab2 = st.columns([2, 2])

    with ab1:
        # Build flat option list with group headers
        flat_options = ["None / Unknown"]
        group_map    = {}
        for group, abilities in ABILITY_OPTIONS.items():
            flat_options.append(group)  # non-selectable header
            for ab_name in abilities:
                flat_options.append("  " + ab_name)
                group_map["  " + ab_name] = (ab_name, abilities[ab_name])

        selected_raw = st.selectbox(
            "Select Ability",
            options=flat_options,
            index=0,
            key="pred_ability",
            help="Sorted by competitive impact. Group headers are not selectable.",
        )

        # Show description below the selectbox
        if selected_raw in group_map:
            ab_name, ab_desc = group_map[selected_raw]
            st.markdown(
                '<div style="background:#0f3460;border-left:3px solid #FFCB05;'
                'padding:0.4rem 0.6rem;border-radius:0 6px 6px 0;margin-top:0.3rem;">'
                '<p style="font-family:Press Start 2P;font-size:0.42rem;'
                'color:#FFCB05;margin:0 0 3px;">' + ab_name.upper() + '</p>'
                '<p style="font-family:Nunito;font-size:0.78rem;'
                'color:#A0A0A0;margin:0;">' + ab_desc + '</p>'
                '</div>',
                unsafe_allow_html=True)
        elif selected_raw.startswith("--"):
            st.markdown(
                '<div style="background:#1a1a2e;border-left:3px solid #555;'
                'padding:0.4rem 0.6rem;border-radius:0 6px 6px 0;margin-top:0.3rem;">'
                '<p style="font-family:Nunito;font-size:0.78rem;color:#555;margin:0;">'
                'Please select an ability from this tier group.</p>'
                '</div>',
                unsafe_allow_html=True)
        elif selected_raw == "None / Unknown":
            st.markdown(
                '<div style="background:#0f3460;border-left:3px solid #555;'
                'padding:0.4rem 0.6rem;border-radius:0 6px 6px 0;margin-top:0.3rem;">'
                '<p style="font-family:Nunito;font-size:0.78rem;color:#555;margin:0;">'
                'No ability selected - model uses average ability score (1).</p>'
                '</div>',
                unsafe_allow_html=True)

    with ab2:
        has_crippling = st.checkbox(
            "Has crippling ability",
            key="pred_crippling",
            help="Truant, Slow Start, Defeatist etc. — abilities that actively hurt the Pokemon")

        if has_crippling:
            crippling_options = list(CRIPPLING_ABILITY_INFO.keys())
            selected_crippling = st.selectbox(
                "Select crippling ability",
                options=crippling_options,
                key="pred_crippling_name",
            )
            # Show crippling ability description
            st.markdown(
                '<div style="background:#b71c1c;border-left:3px solid #F44336;'
                'padding:0.4rem 0.7rem;border-radius:0 6px 6px 0;margin-top:0.3rem;">'
                '<p style="font-family:Nunito;font-size:0.78rem;color:#EAEAEA;margin:0;">'
                + CRIPPLING_ABILITY_INFO.get(selected_crippling, "") + '</p>'
                '</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                '<div style="background:#1a1a2e;border:1px solid #333;'
                'border-radius:6px;padding:0.5rem 0.8rem;margin-top:0.2rem;opacity:0.4;">'
                '<p style="font-family:Nunito;font-size:0.8rem;color:#555;margin:0;">'
                'Select crippling ability</p>'
                '</div>',
                unsafe_allow_html=True)
    
    # Nature recommender
    st.markdown(
        "<p style='font-family:Press Start 2P;font-size:0.5rem;color:#FFCB05;"
        "margin-bottom:0.3rem;margin-top:0.5rem;'>RECOMMENDED NATURE</p>",
        unsafe_allow_html=True)

    selected_raw_ab = st.session_state.get("pred_ability", "None / Unknown")
    ability_for_nature = selected_raw_ab.strip() if selected_raw_ab not in ("None / Unknown", "") and not selected_raw_ab.startswith("--") else "Unknown"
    is_crippling_for_nature = st.session_state.get("pred_crippling", False)

    rec_nature, rec_reason = recommend_nature(
        st.session_state["pred_hp"],
        st.session_state["pred_attack"],
        st.session_state["pred_defense"],
        st.session_state["pred_sp_atk"],
        st.session_state["pred_sp_def"],
        st.session_state["pred_speed"],
        ability=ability_for_nature,
        is_crippling=is_crippling_for_nature,
    )
    boost_stat, drop_stat, _ = NATURES[rec_nature]
    boost_label = "+" + STAT_LABELS[boost_stat] if boost_stat else "No change"
    drop_label  = "-" + STAT_LABELS[drop_stat]  if drop_stat  else "No change"

    st.markdown(
        '<div style="background:#0f3460;border-left:4px solid #FFCB05;'
        'padding:0.8rem 1rem;border-radius:0 8px 8px 0;margin-bottom:0.5rem;">'
        '<div style="display:flex;align-items:center;gap:1rem;margin-bottom:6px;">'
        '<p style="font-family:Press Start 2P;font-size:0.7rem;'
        'color:#FFCB05;margin:0;">' + rec_nature.upper() + '</p>'
        '<span style="background:#4CAF50;color:white;font-family:Nunito;'
        'font-size:0.75rem;font-weight:700;padding:2px 10px;border-radius:8px;">'
        + boost_label + '</span>'
        '<span style="background:#F44336;color:white;font-family:Nunito;'
        'font-size:0.75rem;font-weight:700;padding:2px 10px;border-radius:8px;">'
        + drop_label + '</span>'
        '</div>'
        '<p style="font-family:Nunito;font-size:0.82rem;color:#A0A0A0;margin:0;">'
        + rec_reason + '</p>'
        '</div>',
        unsafe_allow_html=True)

    # All natures expander
    with st.expander("SEE ALL 25 NATURES"):
        for group_name, nature_list in NATURE_GROUPS.items():
            st.markdown(
                '<p style="font-family:Press Start 2P;font-size:0.5rem;'
                'color:#FFCB05;margin:0.7rem 0 0.3rem;">' + group_name.upper() + '</p>',
                unsafe_allow_html=True)
            for nat in nature_list:
                b, d, note = NATURES[nat]
                b_html = (
                    '<span style="background:#4CAF50;color:white;font-family:Nunito;'
                    'font-size:0.7rem;font-weight:700;padding:1px 8px;border-radius:6px;">'
                    '+' + STAT_LABELS[b] + '</span> '
                ) if b else ''
                d_html = (
                    '<span style="background:#F44336;color:white;font-family:Nunito;'
                    'font-size:0.7rem;font-weight:700;padding:1px 8px;border-radius:6px;">'
                    '-' + STAT_LABELS[d] + '</span>'
                ) if d else ''
                is_rec = nat == rec_nature
                border = 'border-left:3px solid #FFCB05;' if is_rec else 'border-left:3px solid #1a1a2e;'
                st.markdown(
                    '<div style="display:flex;align-items:flex-start;gap:0.8rem;'
                    'background:#1a1a2e;' + border +
                    'padding:0.4rem 0.7rem;margin-bottom:3px;border-radius:0 4px 4px 0;">'
                    '<p style="font-family:Press Start 2P;font-size:0.5rem;'
                    'color:' + ('#FFCB05' if is_rec else '#EAEAEA') + ';'
                    'margin:0;min-width:70px;">' + nat.upper() + '</p>'
                    '<div style="display:flex;gap:4px;align-items:center;min-width:140px;">'
                    + b_html + d_html +
                    '</div>'
                    '<p style="font-family:Nunito;font-size:0.75rem;'
                    'color:#A0A0A0;margin:0;">' + note + '</p>'
                    '</div>',
                    unsafe_allow_html=True)

    # Speed tier calculator
    user_speed = st.session_state["pred_speed"]

    # Nature speed modifier — uses rec_nature already calculated above
    SPEED_PLUS  = {"Jolly","Timid","Hasty","Naive"}
    SPEED_MINUS = {"Brave","Quiet","Relaxed","Sassy"}
    if rec_nature in SPEED_PLUS:
        nature_mod  = 1.1
        nature_note = rec_nature + " (+10% Speed)"
    elif rec_nature in SPEED_MINUS:
        nature_mod  = 0.9
        nature_note = rec_nature + " (-10% Speed)"
    else:
        nature_mod  = 1.0
        nature_note = ""

    # Ability speed modifier — uses ability_for_nature already calculated above
    WEATHER_SPEED = {'Swift Swim','Chlorophyll','Sand Rush','Slush Rush','Surge Surfer'}
    if ability_for_nature == 'Speed Boost':
        ability_speed_note = "Speed Boost raises Speed each turn - showing base only"
        ability_mod        = 1.0
    elif ability_for_nature in WEATHER_SPEED:
        ability_speed_note = ability_for_nature + " doubles Speed in weather"
        ability_mod        = 2.0
    elif ability_for_nature == 'Unburden':
        ability_speed_note = "Unburden doubles Speed after item consumed"
        ability_mod        = 2.0
    else:
        ability_speed_note = ""
        ability_mod        = 1.0

    effective_speed = int(user_speed * nature_mod * ability_mod)
    speed_changed   = effective_speed != user_speed

    expander_title = "SEE SPEED TIERS FOR " + str(user_speed) + " BASE SPEED"
    if speed_changed:
        expander_title += " (" + str(effective_speed) + " EFFECTIVE)"

    with st.expander(expander_title):
        speed_df = build_speed_tiers(df_t)

        outspeeds = speed_df[speed_df['speed'] <  effective_speed].copy()
        ties      = speed_df[speed_df['speed'] == effective_speed].copy()
        outsped   = speed_df[speed_df['speed'] >  effective_speed].copy()

        # Show modifier note if applicable
        notes = [n for n in [nature_note, ability_speed_note] if n]
        if notes:
            st.markdown(
                '<div style="background:#0f3460;border-left:3px solid #FFCB05;'
                'padding:0.4rem 0.8rem;border-radius:0 6px 6px 0;margin-bottom:0.8rem;">'
                '<p style="font-family:Nunito;font-size:0.82rem;color:#FFCB05;margin:0;">'
                + " · ".join(notes) + '</p>'
                '</div>',
                unsafe_allow_html=True)

        # Full list popup
        @st.dialog("FULL SPEED TIER LIST", width="large")
        def show_speed_dialog(user_spd, speed_df):
            outspeeds_all = speed_df[speed_df['speed'] <  user_spd].sort_values('speed', ascending=False)
            ties_all      = speed_df[speed_df['speed'] == user_spd]
            outsped_all   = speed_df[speed_df['speed'] >  user_spd].sort_values('speed')

            def full_band(label, colour, df_band):
                if len(df_band) == 0:
                    return
                st.markdown(
                    '<p style="font-family:Press Start 2P;font-size:0.45rem;color:'
                    + colour + ';margin:0.7rem 0 0.3rem;">'
                    + label + ' (' + str(len(df_band)) + ')</p>',
                    unsafe_allow_html=True)
                for _, row in df_band.iterrows():
                    surl = sprite_url(int(row['pokedex_id']))
                    t2   = " / " + row['type_2'] if row['type_2'] != "None" else ""
                    tier_col = {
                        'Uber':'#F95587','OU':'#FFCB05','BL':'#FF9800',
                        'UU':'#4CAF50','BL2':'#66BB6A','RU':'#29B6F6',
                    }.get(row['tier'], '#A0A0A0')
                    st.markdown(
                        '<div style="display:flex;align-items:center;gap:0.7rem;'
                        'background:#0f3460;border-left:3px solid ' + colour + ';'
                        'padding:0.35rem 0.7rem;margin-bottom:3px;border-radius:0 4px 4px 0;">'
                        '<img src="' + surl + '" width="40" height="40" '
                        'style="image-rendering:pixelated;" '
                        'onerror="this.style.display=\'none\'">'
                        '<div style="flex:1;">'
                        '<p style="font-family:Press Start 2P;font-size:0.5rem;'
                        'color:#EAEAEA;margin:0;">' + row['name'].upper() + '</p>'
                        '<p style="font-family:Nunito;font-size:0.7rem;'
                        'color:#A0A0A0;margin:0;">' + row['type_1'] + t2 + '</p>'
                        '</div>'
                        '<span style="font-family:Press Start 2P;font-size:0.5rem;'
                        'color:' + tier_col + ';margin-right:0.5rem;">' + str(row['tier']) + '</span>'
                        '<span style="font-family:Press Start 2P;font-size:0.5rem;'
                        'color:' + colour + ';min-width:28px;text-align:right;">'
                        + str(int(row['speed'])) + '</span>'
                        '</div>',
                        unsafe_allow_html=True)

            full_band("OUTSPED BY", "#F44336", outsped_all)
            full_band("SPEED TIES", "#FFCB05", ties_all)
            full_band("OUTSPEEDS",  "#4CAF50", outspeeds_all)

        def speed_row(row, colour):
            surl = sprite_url(int(row['pokedex_id']))
            t2   = " / " + row['type_2'] if row['type_2'] != "None" else ""
            tier_col = {
                'Uber':'#F95587','OU':'#FFCB05','BL':'#FF9800',
                'UU':'#4CAF50','BL2':'#66BB6A','RU':'#29B6F6',
            }.get(row['tier'], '#A0A0A0')
            st.markdown(
                '<div style="display:flex;align-items:center;gap:0.7rem;'
                'background:#0f3460;border-left:3px solid ' + colour + ';'
                'padding:0.35rem 0.7rem;margin-bottom:3px;border-radius:0 4px 4px 0;">'
                '<img src="' + surl + '" width="40" height="40" '
                'style="image-rendering:pixelated;" '
                'onerror="this.style.display=\'none\'">'
                '<div style="flex:1;">'
                '<p style="font-family:Press Start 2P;font-size:0.5rem;'
                'color:#EAEAEA;margin:0;">' + row['name'].upper() + '</p>'
                '<p style="font-family:Nunito;font-size:0.7rem;'
                'color:#A0A0A0;margin:0;">' + row['type_1'] + t2 + '</p>'
                '</div>'
                '<span style="font-family:Press Start 2P;font-size:0.5rem;'
                'color:' + tier_col + ';margin-right:0.5rem;">' + str(row['tier']) + '</span>'
                '<span style="font-family:Press Start 2P;font-size:0.5rem;'
                'color:' + colour + ';min-width:28px;text-align:right;">'
                + str(int(row['speed'])) + '</span>'
                '</div>',
                unsafe_allow_html=True)

        # Summary counts
        st.markdown(
            '<div style="display:flex;gap:1rem;margin-bottom:0.8rem;">'
            '<div style="background:#1b5e20;border-radius:6px;padding:0.4rem 0.8rem;text-align:center;">'
            '<p style="font-family:Press Start 2P;font-size:0.5rem;color:#4CAF50;margin:0;">OUTSPEEDS</p>'
            '<p style="font-family:Press Start 2P;font-size:0.7rem;color:#4CAF50;margin:0;">'
            + str(len(outspeeds)) + '</p></div>'
            '<div style="background:#1a237e;border-radius:6px;padding:0.4rem 0.8rem;text-align:center;">'
            '<p style="font-family:Press Start 2P;font-size:0.5rem;color:#FFCB05;margin:0;">TIES</p>'
            '<p style="font-family:Press Start 2P;font-size:0.7rem;color:#FFCB05;margin:0;">'
            + str(len(ties)) + '</p></div>'
            '<div style="background:#b71c1c;border-radius:6px;padding:0.4rem 0.8rem;text-align:center;">'
            '<p style="font-family:Press Start 2P;font-size:0.5rem;color:#F44336;margin:0;">OUTSPED BY</p>'
            '<p style="font-family:Press Start 2P;font-size:0.7rem;color:#F44336;margin:0;">'
            + str(len(outsped)) + '</p></div>'
            '<div style="background:#0f3460;border-radius:6px;padding:0.4rem 0.8rem;'
            'text-align:center;border:1px solid #FFCB05;">'
            '<p style="font-family:Press Start 2P;font-size:0.5rem;color:#A0A0A0;margin:0;">BASE / EFF.</p>'
            '<p style="font-family:Press Start 2P;font-size:0.7rem;color:#FFCB05;margin:0;">'
            + str(user_speed) + ' / ' + str(effective_speed) + '</p></div>'
            '</div>',
            unsafe_allow_html=True)

        # Top 10 per band
        for label, colour, df_band, ascending in [
            ("OUTSPED BY (top 10 closest)", "#F44336", outsped.nsmallest(10,'speed'),  True),
            ("SPEED TIES",                  "#FFCB05", ties,                           False),
            ("OUTSPEEDS (top 10 closest)",  "#4CAF50", outspeeds.nlargest(10,'speed'), False),
        ]:
            if len(df_band) == 0:
                continue
            st.markdown(
                '<p style="font-family:Press Start 2P;font-size:0.5rem;color:'
                + colour + ';margin:0.5rem 0 0.3rem;">' + label + '</p>',
                unsafe_allow_html=True)
            for _, row in df_band.iterrows():
                speed_row(row, colour)

        # Full list button
        st.markdown("<div style='margin-top:0.8rem;'></div>", unsafe_allow_html=True)
        if st.button("SEE FULL SPEED TIER LIST", key="speed_tier_full_btn"):
            show_speed_dialog(effective_speed, speed_df)

    divider()

    if go_btn:
        st.session_state["show_results"] = True

    if st.session_state.get("show_results"):
        with st.spinner("Consulting the Pokedex..."):
            try:
                # Look up ability score from scored CSV
                ability_name = st.session_state.get("pred_ability", "").strip()
                is_crippling = st.session_state.get("pred_crippling", False)

                if is_crippling:
                    ab_score = -3
                elif ability_name:
                    from src.feature_engineering import ABILITY_SCORES
                    is_crippling  = st.session_state.get("pred_crippling", False)
                    selected_raw  = st.session_state.get("pred_ability", "None / Unknown")

                    if is_crippling:
                        ab_score     = -3
                        ability_name = st.session_state.get("pred_crippling_name", "Truant")
                    elif selected_raw and selected_raw not in ("None / Unknown",) and not selected_raw.startswith("--"):
                        ability_name = selected_raw.strip()
                        ab_score     = ABILITY_SCORES.get(ability_name, 1)
                    else:
                        ab_score     = 1
                        ability_name = "Unknown"

                row = build_feature_row(
                    hp=st.session_state["pred_hp"],
                    attack=st.session_state["pred_attack"],
                    defense=st.session_state["pred_defense"],
                    sp_attack=st.session_state["pred_sp_atk"],
                    sp_defense=st.session_state["pred_sp_def"],
                    speed=st.session_state["pred_speed"],
                    type_1=st.session_state["pred_type1"],
                    type_2=(st.session_state["pred_type2"]
                            if st.session_state["pred_type2"] != "None" else None),
                    is_legendary=st.session_state["pred_is_legendary"],
                    height=st.session_state["pred_height"],
                    weight=st.session_state["pred_weight"],
                    hidden_ability=ability_name if ability_name else 'Unknown',
                    best_ability_score=max(0, ab_score),
                    has_crippling_ability=1 if is_crippling else 0,
                )
                res  = predict_viability(row, model_data)
                prob = res['probability'] * 100
                vcol = "#4CAF50" if res['viable'] else "#F44336"
                vbg  = "#1b5e20" if res['viable'] else "#b71c1c"
                vrd  = "COMPETITIVELY VIABLE" if res['viable'] else "NOT VIABLE"
                vico = "✅" if res['viable'] else "❌"

                v1, v2 = st.columns([1, 2])
                with v1:
                    st.markdown(
                        '<div style="background:' + vbg + ';border:2px solid ' + vcol + ';'
                        'padding:1.3rem;border-radius:8px;text-align:center;">'
                        '<p style="font-family:Press Start 2P;font-size:0.65rem;color:' + vcol + ';'
                        'margin:0;line-height:2;">' + vico + '<br>' + vrd + '</p>'
                        '</div>',
                        unsafe_allow_html=True)
                    st.markdown(
                        '<div style="background:#0f3460;border:1px solid #FFCB05;'
                        'padding:0.7rem;border-radius:6px;margin-top:0.7rem;text-align:center;">'
                        '<p style="font-family:Nunito;color:#A0A0A0;margin:0 0 5px;font-size:0.8rem;">'
                        'Viability Probability</p>'
                        '<div style="background:#1a1a2e;border-radius:4px;height:14px;">'
                        '<div style="background:' + vcol + ';width:' + str(round(prob)) + '%;'
                        'height:100%;border-radius:4px;"></div></div>'
                        '<p style="font-family:Press Start 2P;font-size:0.85rem;color:#FFCB05;margin:5px 0 0;">'
                        + str(round(prob, 1)) + '%</p></div>',
                        unsafe_allow_html=True)

                with v2:
                    st.markdown(
                        "<p style='font-family:Press Start 2P;font-size:0.45rem;"
                        "color:#FFCB05;'>WHY THIS VERDICT?</p>",
                        unsafe_allow_html=True)
                    tf = get_top_shap_features(
                        res['shap_values'], res['feature_names'], n=10)
                    fig_sh = go.Figure(go.Bar(
                        x=tf['shap_value'],
                        y=tf['feature'].str.replace('_', ' ').str.title(),
                        orientation='h', name="",
                        marker_color=["#4CAF50" if v > 0 else "#F44336"
                                      for v in tf['shap_value']],
                        text=tf['shap_value'].round(3).astype(str),
                        textposition='outside',
                    ))
                    fig_sh.update_layout(
                        title=dict(text=""), xaxis_title="SHAP Value",
                        yaxis_title="", height=340, showlegend=False,
                        xaxis=dict(zeroline=True, zerolinecolor='#FFCB05',
                                   zerolinewidth=2))
                    st.plotly_chart(tp(fig_sh, 340),
                                    use_container_width=True, key="shap_bar")
                    # Tier prediction
                    tier_result = predict_tier(row, tier_model_data)
                    tier        = tier_result['tier']
                    tier_probs  = tier_result['probabilities']

                    tier_colours_map = {
                        'Top Tier': '#F95587',
                        'Mid Tier': '#FFCB05',
                        'Low Tier': '#6390F0',
                    }
                    tier_definitions = {
                        'Top Tier': 'Uber / OU — used in the highest levels of competitive play',
                        'Mid Tier': 'UU / RU — solid but outclassed in top formats',
                        'Low Tier': 'NU / PU — rarely used competitively',
                    }
                    tier_col = tier_colours_map.get(tier, '#EAEAEA')
                    tier_def = tier_definitions.get(tier, '')

                    # Three probability boxes
                    boxes_html = '<div style="display:flex;gap:8px;margin-bottom:10px;">'
                    for t in ['Top Tier', 'Mid Tier', 'Low Tier']:
                        p      = tier_probs[t]
                        tc     = tier_colours_map[t]
                        weight = 'font-weight:700;' if t == tier else ''
                        boxes_html += (
                            '<div style="flex:1;text-align:center;background:#1a1a2e;'
                            + ('border:2px solid ' + tc if t == tier else 'border:1px solid #333') + ';'
                            'border-radius:8px;padding:0.5rem 0.3rem;">'
                            '<p style="font-family:Press Start 2P;font-size:0.5rem;'
                            'color:' + tc + ';margin:0 0 4px;' + weight + '">' + t.upper() + '</p>'
                            '<p style="font-family:Nunito;font-size:0.85rem;'
                            'color:#EAEAEA;margin:0;' + weight + '">' + str(p) + '%</p>'
                            '</div>'
                        )
                    boxes_html += '</div>'

                    st.markdown(
                        '<div style="margin-top:1.5rem;'
                        'border-left:4px solid ' + tier_col + ';'
                        'background:#0f3460;'
                        'padding:0.9rem 1rem;border-radius:0 8px 8px 0;">'

                        '<p style="font-family:Press Start 2P;font-size:0.5rem;'
                        'color:' + tier_col + ';margin:0 0 4px;">PREDICTED TIER</p>'

                        '<p style="font-family:Press Start 2P;font-size:0.75rem;'
                        'color:' + tier_col + ';margin:0 0 6px;">' + tier.upper() + '</p>'

                        '<p style="font-family:Nunito;font-size:0.82rem;'
                        'color:#A0A0A0;margin:0 0 12px;">' + tier_def + '</p>'

                        '<p style="font-family:Nunito;font-size:0.72rem;'
                        'color:#555;margin:0;">' + tier_result['accuracy_note'] + '</p>'
                        '</div>',
                        unsafe_allow_html=True)

                st.markdown(boxes_html, unsafe_allow_html=True)

                divider()
                st.markdown(
                    "<p style='font-family:Press Start 2P;font-size:0.45rem;color:#FFCB05;'>"
                    "10 MOST SIMILAR POKEMON BY BST</p>",
                    unsafe_allow_html=True)

                sim = df[df['form_type'] == 'Base'].copy()
                sim['bst_diff'] = abs(sim['bst'] - bst)
                similar = sim.nsmallest(10, 'bst_diff').reset_index(drop=True)
                top = similar.iloc[0]
                t2d = "/ " + top['type_2'] if top['type_2'] != "None" else ""
                top_sprite = sprite_url(int(top['pokedex_id']), official=True)
                st.markdown(
                    '<div style="display:flex;align-items:center;gap:1.2rem;background:#0f3460;'
                    'border:1px solid #FFCB05;padding:0.8rem;border-radius:8px;margin-bottom:0.8rem;">'
                    '<img src="' + top_sprite + '" '
                    'style="width:90px;height:90px;object-fit:contain;'
                    'filter:drop-shadow(0 0 7px rgba(255,203,5,0.4));" '
                    'onerror="this.style.display=\'none\'">'
                    '<div>'
                    '<p style="font-family:Press Start 2P;font-size:0.5rem;color:#A0A0A0;margin:0;">'
                    'CLOSEST MATCH</p>'
                    '<p style="font-family:Press Start 2P;font-size:0.7rem;color:#FFCB05;margin:4px 0;">'
                    + str(top['name']).upper() + '</p>'
                    '<p style="font-family:Nunito;color:#EAEAEA;margin:0;">'
                    + str(top['type_1']) + ' ' + t2d +
                    ' - BST ' + str(int(top['bst'])) +
                    ' - Gen ' + str(int(top['generation'])) + '</p>'
                    '</div></div>',
                    unsafe_allow_html=True)

                disp = similar[['name','bst','type_1','type_2',
                                'generation','stat_profile']].copy()
                disp.index = range(1, len(disp) + 1)
                st.dataframe(disp, use_container_width=True)
            
                if st.button("SEE WHAT BEATS IT!!", key="beats_me_btn"):
                    selected_raw = st.session_state.get("pred_ability", "None / Unknown")
                    ab_for_dialog = selected_raw.strip() if selected_raw not in ("None / Unknown", "") and not selected_raw.startswith("--") else "Unknown"
                    show_type_matchup_dialog(
                        st.session_state["pred_type1"],
                        st.session_state["pred_type2"],
                        df,
                        ability=ab_for_dialog
                    )

            except Exception as e:
                st.error("Prediction error: " + str(e))
    else:
        st.markdown(
            '<div style="background:#0f3460;border:1px dashed #A0A0A0;'
            'padding:1.8rem;border-radius:8px;text-align:center;">'
            '<p style="font-family:Press Start 2P;font-size:0.5rem;color:#A0A0A0;line-height:2.5;">'
            'SET STATS ABOVE<br>THEN PRESS ANALYSE</p></div>',
            unsafe_allow_html=True)

# -- TAB 7: MODEL INSIGHTS -----------------------------------------------------
def tab_model(df, df_t, model_data):
    sh("MODEL INSIGHTS", "How does the XGBoost viability classifier work?", pid=777)
    import shap
    mdl, fc = model_data['model'], model_data['feature_cols']

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("ROC-AUC",    "0.9452")
    with c2: st.metric("Accuracy",   "89.3%")
    with c3: st.metric("Train Gens", "1 - 7")
    with c4: st.metric("Test Gen",   "8")

    divider()

    cs, cc = st.columns(2)
    with cs:
        st.markdown("### FEATURE IMPORTANCE (SHAP)")
        sp = Path(__file__).resolve().parent.parent / "models" / "shap_summary.png"
        if sp.exists():
            st.image(str(sp), use_container_width=True)
        else:
            st.warning("SHAP summary image not found. Run the training notebook first.")
    with cc:
        st.markdown("### CONFUSION MATRIX - GEN 8")
        fig_cm = px.imshow(
            [[65,8],[1,10]],
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['Non-Viable','Viable'], y=['Non-Viable','Viable'],
            color_continuous_scale='YlOrRd', text_auto=True,
            title="Confusion Matrix (Gen 8 Test Set)")
        fig_cm.update_traces(textfont_size=16)
        st.plotly_chart(tp(fig_cm, 400), use_container_width=True, key="cm")

    divider()
    st.markdown("### TOP 20 FEATURES BY SHAP IMPORTANCE")
    dm = df_t[df_t['is_viable'].notna()].copy()
    dm['is_viable'] = dm['is_viable'].astype(int)
    with st.spinner("Computing SHAP values..."):
        sv  = shap.TreeExplainer(mdl).shap_values(dm[fc])
        imp = (pd.DataFrame({'Feature': fc, 'Mean |SHAP|': np.abs(sv).mean(axis=0)})
               .sort_values('Mean |SHAP|', ascending=True).tail(20))
    fig_i = px.bar(imp, x='Mean |SHAP|', y='Feature', orientation='h',
                   title="Top 20 Features by Mean Absolute SHAP Value",
                   color='Mean |SHAP|', color_continuous_scale='YlOrRd')
    fig_i.update_layout(coloraxis_showscale=False, yaxis_title="")
    st.plotly_chart(tp(fig_i, 600), use_container_width=True, key="mi")

    divider()
    st.markdown("### OUTLIER SPOTLIGHT")
    st.markdown(
        "<p style='font-family:Nunito;color:#A0A0A0;font-size:0.85rem;'>"
        "Pokemon where the model strongly disagrees with Smogon. "
        "Reveals what base stats and typing cannot capture - "
        "abilities, movepool, and speed tier interactions.</p>",
        unsafe_allow_html=True)

    # Run predictions on all tiered base form Pokemon
    df_tiered_base = df_t[
        (df_t['is_viable'].notna()) &
        (df_t['form_type'] == 'Base')
    ].copy()

    for col in fc:
        if col not in df_tiered_base.columns:
            df_tiered_base[col] = 0

    probs_all = mdl.predict_proba(df_tiered_base[fc])[:, 1]
    df_tiered_base['pred_probability'] = (probs_all * 100).round(1)
    df_tiered_base['pred_viable']      = (probs_all >= model_data['threshold']).astype(bool)
    df_tiered_base['actual_viable']    = df_tiered_base['is_viable'].astype(bool)

    overrated = df_tiered_base[
        (df_tiered_base['pred_viable'] == True) &
        (df_tiered_base['actual_viable'] == False)
    ].sort_values('pred_probability', ascending=False)

    underrated = df_tiered_base[
        (df_tiered_base['pred_viable'] == False) &
        (df_tiered_base['actual_viable'] == True)
    ].sort_values('pred_probability', ascending=True)

    col_over, col_under = st.columns(2)

    with col_over:
        st.markdown(
            '<p style="font-family:Press Start 2P;font-size:0.5rem;color:#F44336;">'
            'OVERRATED BY MODEL</p>'
            '<p style="font-family:Nunito;color:#A0A0A0;font-size:0.8rem;margin-top:-0.3rem;">'
            'Good stats, but ability or movepool lets them down</p>',
            unsafe_allow_html=True)

        # Show top 10 by default
        for _, row in overrated.head(10).iterrows():
            t2   = " / " + row['type_2'] if row['type_2'] != "None" else ""
            surl = sprite_url(int(row['pokedex_id']))
            prob = round(float(row['pred_probability']), 1)
            st.markdown(
                '<div style="display:flex;align-items:center;gap:0.8rem;'
                'background:#0f3460;border-left:3px solid #F44336;'
                'padding:0.5rem 0.7rem;margin-bottom:0.4rem;border-radius:0 6px 6px 0;">'
                '<img src="' + surl + '" width="40" height="40" '
                'style="image-rendering:pixelated;" '
                'onerror="this.style.display=\'none\'">'
                '<div style="flex:1;">'
                '<p style="font-family:Press Start 2P;font-size:0.45rem;'
                'color:#FFCB05;margin:0;">' + row['name'].upper() + '</p>'
                '<p style="font-family:Nunito;font-size:0.75rem;'
                'color:#A0A0A0;margin:0;">'
                + row['type_1'] + t2 + ' - BST ' + str(int(row['bst'])) + '</p>'
                '</div>'
                '<div style="text-align:right;">'
                '<p style="font-family:Press Start 2P;font-size:0.4rem;'
                'color:#F44336;margin:0;">' + str(prob) + '%</p>'
                '<p style="font-family:Nunito;font-size:0.7rem;'
                'color:#A0A0A0;margin:0;">actual: ' + row['tier'] + '</p>'
                '</div></div>',
                unsafe_allow_html=True)

        # Expander for remaining overrated
        remaining = overrated.iloc[10:]
        if len(remaining) > 0:
            with st.expander("SHOW ALL " + str(len(overrated)) + " OVERRATED POKEMON"):
                for _, row in remaining.iterrows():
                    t2   = " / " + row['type_2'] if row['type_2'] != "None" else ""
                    surl = sprite_url(int(row['pokedex_id']))
                    prob = round(float(row['pred_probability']), 1)
                    st.markdown(
                        '<div style="display:flex;align-items:center;gap:0.8rem;'
                        'background:#0f3460;border-left:3px solid #F44336;'
                        'padding:0.5rem 0.7rem;margin-bottom:0.4rem;'
                        'border-radius:0 6px 6px 0;">'
                        '<img src="' + surl + '" width="40" height="40" '
                        'style="image-rendering:pixelated;" '
                        'onerror="this.style.display=\'none\'">'
                        '<div style="flex:1;">'
                        '<p style="font-family:Press Start 2P;font-size:0.45rem;'
                        'color:#FFCB05;margin:0;">' + row['name'].upper() + '</p>'
                        '<p style="font-family:Nunito;font-size:0.75rem;'
                        'color:#A0A0A0;margin:0;">'
                        + row['type_1'] + t2 + ' - BST ' + str(int(row['bst'])) + '</p>'
                        '</div>'
                        '<div style="text-align:right;">'
                        '<p style="font-family:Press Start 2P;font-size:0.4rem;'
                        'color:#F44336;margin:0;">' + str(prob) + '%</p>'
                        '<p style="font-family:Nunito;font-size:0.7rem;'
                        'color:#A0A0A0;margin:0;">actual: ' + row['tier'] + '</p>'
                        '</div></div>',
                        unsafe_allow_html=True)

    with col_under:
        st.markdown(
            '<p style="font-family:Press Start 2P;font-size:0.5rem;color:#4CAF50;">'
            'UNDERRATED BY MODEL</p>'
            '<p style="font-family:Nunito;color:#A0A0A0;font-size:0.8rem;margin-top:-0.5rem;">'
            'Weak stats on paper, but ability or niche makes them viable</p>',
            unsafe_allow_html=True)

        if len(underrated) == 0:
            st.markdown(
                '<div style="background:#0f3460;border:1px dashed #A0A0A0;'
                'padding:1rem;border-radius:6px;text-align:center;">'
                '<p style="font-family:Nunito;color:#A0A0A0;font-size:0.85rem;">'
                'No underrated Pokemon found at current threshold.</p>'
                '</div>',
                unsafe_allow_html=True)
        else:
            for _, row in underrated.iterrows():
                t2   = " / " + row['type_2'] if row['type_2'] != "None" else ""
                surl = sprite_url(int(row['pokedex_id']))
                prob = round(float(row['pred_probability']), 1)
                st.markdown(
                    '<div style="display:flex;align-items:center;gap:0.8rem;'
                    'background:#0f3460;border-left:3px solid #4CAF50;'
                    'padding:0.5rem 0.7rem;margin-bottom:0.4rem;border-radius:0 6px 6px 0;">'
                    '<img src="' + surl + '" width="40" height="40" '
                    'style="image-rendering:pixelated;" '
                    'onerror="this.style.display=\'none\'">'
                    '<div style="flex:1;">'
                    '<p style="font-family:Press Start 2P;font-size:0.45rem;'
                    'color:#FFCB05;margin:0;">' + row['name'].upper() + '</p>'
                    '<p style="font-family:Nunito;font-size:0.75rem;'
                    'color:#A0A0A0;margin:0;">'
                    + row['type_1'] + t2 + ' - BST ' + str(int(row['bst'])) + '</p>'
                    '</div>'
                    '<div style="text-align:right;">'
                    '<p style="font-family:Press Start 2P;font-size:0.4rem;'
                    'color:#4CAF50;margin:0;">' + str(prob) + '%</p>'
                    '<p style="font-family:Nunito;font-size:0.7rem;'
                    'color:#A0A0A0;margin:0;">actual: ' + row['tier'] + '</p>'
                    '</div></div>',
                    unsafe_allow_html=True)

    divider()
    st.markdown("### TIER DISTRIBUTION IN TRAINING DATA")
    tc = df_t[df_t['tier'] != 'Untiered']['tier'].value_counts().reset_index()
    tc.columns = ['Tier','Count']
    order = ['AG','Uber','OU','BL','UU','BL2','RU','BL3','NU','BL4','PU','ZU']
    tc['Tier'] = pd.Categorical(
        tc['Tier'],
        categories=[t for t in order if t in tc['Tier'].values],
        ordered=True)
    fig_t = px.bar(tc.sort_values('Tier'), x='Tier', y='Count',
                   title="Pokemon Count by Smogon Tier",
                   color='Count', color_continuous_scale='YlOrRd')
    fig_t.update_layout(coloraxis_showscale=False)
    st.plotly_chart(tp(fig_t, 400), use_container_width=True, key="mt")

# -- MAIN ----------------------------------------------------------------------
def main():
    inject_css()

    bulbasaur_url = sprite_url(1, official=True)
    miraidon_url  = sprite_url(1008, official=True)
    lightning     = '<span style="color:#FFCB05;font-size:1.1rem;">&#9889;</span>'

    st.markdown(
        '<div style="text-align:center;padding:0.4rem 0 0.3rem;">'

        # Sprite + title row
        '<div style="display:flex;align-items:center;justify-content:center;gap:0.8rem;">'

        # Left — Bulbasaur
        '<img src="' + bulbasaur_url + '" '
        'style="width:60px;height:60px;object-fit:contain;'
        'filter:drop-shadow(0 0 6px rgba(124,199,76,0.6));" '
        'onerror="this.style.display=\'none\'">'

        # Title
        '<h1 style="font-size:0.95rem !important;margin:0;">'
        + lightning + ' POKEMETA ANALYSER ' + lightning +
        '</h1>'

        # Right — Miraidon
        '<img src="' + miraidon_url + '" '
        'style="width:80px;height:80px;object-fit:contain;'
        'filter:drop-shadow(0 0 6px rgba(99,144,240,0.6));" '
        'onerror="this.style.display=\'none\'">'

        '</div>'

        '<p style="font-family:Nunito;color:#A0A0A0;font-size:0.85rem;margin-top:0.2rem;">'
        'Competitive Meta Shift Tracker - Gens 1 to 9</p>'
        '</div>',
        unsafe_allow_html=True)

    @st.cache_data
    def get_data():
        return load_featured(), load_tiered()

    @st.cache_resource
    def get_model():
        return load_model(), load_tier_model()

    with st.spinner("Loading Pokedex..."):
        df, df_t = get_data()
        md, td   = get_model()

    tabs = st.tabs([
        "HOME", "GENERATIONS", "TYPE DOMINANCE",
        "LEGENDARY IMPACT", "STAT PROFILES",
        "VIABILITY PREDICTOR", "MODEL INSIGHTS",
        "POWER RANKINGS", "GEN 9 PREDICTIONS",
    ])

    with tabs[0]: tab_home(df, df_t)
    with tabs[1]: tab_generations(df)
    with tabs[2]: tab_types(df)
    with tabs[3]: tab_legendary(df)
    with tabs[4]: tab_profiles(df)
    with tabs[5]: tab_predictor(df, df_t, md, td)
    with tabs[6]: tab_model(df, df_t, md)
    with tabs[7]: tab_rankings(df, df_t)
    with tabs[8]: tab_gen9(df, md)

def tab_gen9(df, model_data):
    sh("GEN 9 PREDICTIONS",
       "Model predictions for Scarlet and Violet - no official Smogon tiers yet",
       pid=921)

    # Run predictions
    model        = model_data['model']
    feature_cols = model_data['feature_cols']
    threshold    = model_data['threshold']

    gen9 = df[
        (df['generation'] == 9) &
        (df['form_type'] == 'Base')
    ].copy()

    for col in feature_cols:
        if col not in gen9.columns:
            gen9[col] = 0

    probs = model.predict_proba(gen9[feature_cols])[:, 1]
    preds = (probs >= threshold).astype(int)

    gen9['viable_probability'] = (probs * 100).round(1)
    gen9['predicted_viable']   = preds.astype(bool)
    gen9['predicted_tier']     = gen9['viable_probability'].apply(
        lambda p: 'Likely OU/Uber' if p >= 75
        else ('Possibly UU'  if p >= 50
        else ('Borderline'   if p >= 35
        else 'Likely NU/PU'))
    )
    gen9 = gen9.sort_values('viable_probability', ascending=False).reset_index(drop=True)

    # Summary metrics
    viable     = int(gen9['predicted_viable'].sum())
    not_viable = int((~gen9['predicted_viable']).sum())
    top_prob   = gen9['viable_probability'].iloc[0]
    avg_prob   = gen9['viable_probability'].mean()

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Gen 9 Pokemon",       str(len(gen9)))
    with c2: st.metric("Predicted Viable",    str(viable))
    with c3: st.metric("Predicted Not Viable",str(not_viable))
    with c4: st.metric("Avg Viability Score", str(round(avg_prob, 1)) + "%")

    # Disclaimer
    st.markdown(
        '<div style="background:#0f3460;border-left:3px solid #FFCB05;'
        'padding:0.8rem 1rem;border-radius:0 6px 6px 0;margin:0.8rem 0;">'
        '<p style="font-family:Nunito;color:#FFCB05;font-weight:700;margin:0 0 4px;">NOTE</p>'
        '<p style="font-family:Nunito;color:#A0A0A0;font-size:0.85rem;margin:0;">'
        'These are model predictions based on base stats, typing, and ability scores. '
        'No official Smogon Gen 9 tier data was used. '
        'The model was trained on Gens 1-7 and validated on Gen 8 (ROC-AUC 0.9452). '
        'Movepool and EV spreads are not captured.</p>'
        '</div>',
        unsafe_allow_html=True)

    divider()

    # Top 10 viable with sprites
    st.markdown("### TOP 10 PREDICTED VIABLE")
    top10 = gen9[gen9['predicted_viable']].head(10)

    cols = st.columns(5)
    for i, (_, row) in enumerate(top10.iterrows()):
        with cols[i % 5]:
            prob     = row['viable_probability']
            col_bar  = "#4CAF50" if prob >= 75 else "#FFCB05"
            t2       = " / " + row['type_2'] if row['type_2'] != "None" else ""
            surl     = sprite_url(int(row['pokedex_id']), official=True)
            st.markdown(
                '<div style="background:#0f3460;border:1px solid ' + col_bar + ';'
                'border-radius:8px;padding:0.7rem;text-align:center;margin-bottom:0.5rem;">'
                '<img src="' + surl + '" width="80" height="80" '
                'style="object-fit:contain;" onerror="this.style.display=\'none\'">'
                '<p style="font-family:Press Start 2P;font-size:0.5rem;'
                'color:#FFCB05;margin:4px 0 2px;">' + row['name'].upper() + '</p>'
                '<p style="font-family:Nunito;font-size:0.75rem;'
                'color:#A0A0A0;margin:0;">' + row['type_1'] + t2 + '</p>'
                '<p style="font-family:Press Start 2P;font-size:0.45rem;'
                'color:' + col_bar + ';margin:4px 0 0;">' + str(round(float(prob), 1)) + '%</p>'
                '</div>',
                unsafe_allow_html=True)

    divider()

    # Tier distribution chart
    st.markdown("### PREDICTED TIER DISTRIBUTION")
    tier_order  = ['Likely OU/Uber', 'Possibly UU', 'Borderline', 'Likely NU/PU']
    tier_colours= ['#F95587', '#FFCB05', '#6390F0', '#A0A0A0']
    tier_counts = gen9['predicted_tier'].value_counts().reindex(tier_order, fill_value=0).reset_index()
    tier_counts.columns = ['Tier', 'Count']

    fig_tier = px.bar(
        tier_counts, x='Tier', y='Count',
        title="Gen 9 Predicted Tier Distribution",
        color='Tier',
        color_discrete_sequence=tier_colours,
    )
    fig_tier.update_layout(showlegend=False)
    st.plotly_chart(tp(fig_tier, 380), use_container_width=True, key="g9_tiers")

    divider()

    # Full table with filters
    st.markdown("### FULL GEN 9 PREDICTIONS")

    fc1, fc2 = st.columns(2)
    with fc1:
        tier_filter = st.multiselect(
            "Filter by predicted tier",
            options=tier_order,
            default=tier_order,
            key="g9_tier_filter")
    with fc2:
        type_filter = st.multiselect(
            "Filter by type",
            options=sorted(df['type_1'].unique().tolist()),
            default=[],
            key="g9_type_filter")

    display = gen9[gen9['predicted_tier'].isin(tier_filter)].copy()
    if type_filter:
        display = display[
            display['type_1'].isin(type_filter) |
            display['type_2'].isin(type_filter)
        ]

    display_cols = ['name', 'type_1', 'type_2', 'bst',
                    'hp', 'attack', 'defense',
                    'sp_attack', 'sp_defense', 'speed',
                    'viable_probability', 'predicted_tier']
    display_out = display[display_cols].copy()
    display_out.index = range(1, len(display_out) + 1)
    display_out['viable_probability'] = display_out['viable_probability'].apply(
        lambda x: str(round(float(x), 1)) + '%'
    )
    st.dataframe(display_out, use_container_width=True)

def tab_rankings(df, df_t):
    sh("GENERATION POWER RANKINGS",
       "Which generation produced the most competitively relevant Pokemon?",
       pid=877)

    # Build rankings data
    base = df_t[
        (df_t['form_type'] == 'Base') &
        (df_t['generation'] <= 8)
    ].copy()

    tier_map_r = {
        'AG':'Top','Uber':'Top','OU':'Top','BL':'Top',
        'UU':'Mid','BL2':'Mid','RU':'Mid',
        'BL3':'Low','NU':'Low','BL4':'Low','PU':'Low','ZU':'Low',
    }
    base['tier_class'] = base['tier'].map(tier_map_r)

    summary = base.groupby('generation').agg(
        total=('name','count'),
        tiered=('tier', lambda x:(x!='Untiered').sum()),
        top_tier=('tier_class', lambda x:(x=='Top').sum()),
        viable=('is_viable', lambda x:(x==True).sum()),
        avg_bst_viable=('bst', lambda x: round(
            x[base.loc[x.index,'is_viable']==True].mean(), 1)),
    ).round(1)

    summary['viable_pct'] = (summary['viable'] / summary['tiered'] * 100).round(1)
    summary = summary.sort_values('viable_pct', ascending=False).reset_index()
    summary['rank'] = range(1, len(summary)+1)

    medals = {1:'🥇', 2:'🥈', 3:'🥉'}

    # Top 3 highlight cards
    st.markdown(
        '<p style="font-family:Press Start 2P;font-size:0.5rem;'
        'color:#FFCB05;margin-bottom:0.5rem;">TOP 3 GENERATIONS</p>',
        unsafe_allow_html=True)

    top3_cols = st.columns(3)
    for i, (_, row) in enumerate(summary.head(3).iterrows()):
        gen   = int(row['generation'])
        pid   = GEN_POKEMON.get(gen, 25)
        medal = medals.get(i+1, '')
        col_border = ['#FFD700','#C0C0C0','#CD7F32'][i]

        with top3_cols[i]:
            surl = sprite_url(pid, official=True)
            st.markdown(
                '<div style="background:#0f3460;border:2px solid ' + col_border + ';'
                'border-radius:10px;padding:0.8rem;text-align:center;">'
                '<p style="font-family:Press Start 2P;font-size:0.7rem;margin:0;">'
                + medal + '</p>'
                '<img src="' + surl + '" width="90" height="90" '
                'style="object-fit:contain;" onerror="this.style.display=\'none\'">'
                '<p style="font-family:Press Start 2P;font-size:0.5rem;'
                'color:#FFCB05;margin:6px 0 2px;">GEN ' + str(gen) + '</p>'
                '<p style="font-family:Press Start 2P;font-size:0.65rem;'
                'color:' + col_border + ';margin:0;">'
                + str(row['viable_pct']) + '% VIABLE</p>'
                '<p style="font-family:Nunito;font-size:0.8rem;'
                'color:#A0A0A0;margin:4px 0 0;">'
                + str(int(row['viable'])) + ' viable of '
                + str(int(row['tiered'])) + ' tiered</p>'
                '</div>',
                unsafe_allow_html=True)

    divider()

    # Full rankings chart
    st.markdown(
        '<p style="font-family:Press Start 2P;font-size:0.5rem;'
        'color:#FFCB05;margin-bottom:0.5rem;">VIABLE % BY GENERATION</p>',
        unsafe_allow_html=True)

    fig_rank = px.bar(
        summary,
        x='generation', y='viable_pct',
        color='viable_pct',
        color_continuous_scale='YlOrRd',
        title="% of Tiered Pokemon that Reached Viable Tiers (UU or above)",
        labels={'viable_pct':'Viable %','generation':'Generation'},
        text=summary['viable_pct'].astype(str) + '%',
    )
    fig_rank.update_traces(textposition='outside')
    fig_rank.update_layout(
        coloraxis_showscale=False,
        xaxis=dict(tickmode='linear', dtick=1),
    )
    st.plotly_chart(tp(fig_rank, 420), use_container_width=True, key="rank_bar")

    divider()

    # Top tier count chart
    st.markdown(
        '<p style="font-family:Press Start 2P;font-size:0.5rem;'
        'color:#FFCB05;margin-bottom:0.5rem;">TOP TIER POKEMON PER GENERATION</p>',
        unsafe_allow_html=True)

    fig_top = px.bar(
        summary,
        x='generation', y='top_tier',
        color='top_tier',
        color_continuous_scale='YlOrRd',
        title="Number of Uber/OU/BL Pokemon Introduced per Generation",
        labels={'top_tier':'Top Tier Count','generation':'Generation'},
        text='top_tier',
    )
    fig_top.update_traces(textposition='outside')
    fig_top.update_layout(
        coloraxis_showscale=False,
        xaxis=dict(tickmode='linear', dtick=1),
    )
    st.plotly_chart(tp(fig_top, 420), use_container_width=True, key="rank_top")

    divider()

    # Full table
    st.markdown(
        '<p style="font-family:Press Start 2P;font-size:0.5rem;'
        'color:#FFCB05;margin-bottom:0.5rem;">FULL RANKINGS TABLE</p>',
        unsafe_allow_html=True)

    rows = ""
    for _, row in summary.iterrows():
        gen    = int(row['generation'])
        rank   = int(row['rank'])
        medal  = medals.get(rank, str(rank))
        pid    = GEN_POKEMON.get(gen, 25)
        pct    = row['viable_pct']
        bar_w  = int(pct)
        bar_c  = '#4CAF50' if pct >= 30 else ('#FFCB05' if pct >= 20 else '#F44336')
        avg    = row['avg_bst_viable'] if str(row['avg_bst_viable']) != 'nan' else 'N/A'

        rows += (
            '<tr style="border-bottom:1px solid #1a1a2e;">'
            '<td style="padding:6px;text-align:center;font-family:Press Start 2P;'
            'font-size:0.5rem;">' + medal + '</td>'
            '<td style="padding:6px;text-align:center;">'
            + sprite_img(pid, 36) + '</td>'
            '<td style="padding:6px;text-align:center;font-family:Press Start 2P;'
            'font-size:0.5rem;color:#FFCB05;">GEN ' + str(gen) + '</td>'
            '<td style="padding:6px;text-align:center;">' + str(int(row['total'])) + '</td>'
            '<td style="padding:6px;text-align:center;">' + str(int(row['tiered'])) + '</td>'
            '<td style="padding:6px;text-align:center;color:#F95587;">'
            + str(int(row['top_tier'])) + '</td>'
            '<td style="padding:6px;text-align:center;color:#4CAF50;">'
            + str(int(row['viable'])) + '</td>'
            '<td style="padding:6px;">'
            '<div style="background:#1a1a2e;border-radius:4px;height:10px;">'
            '<div style="background:' + bar_c + ';width:' + str(bar_w) + '%;'
            'height:100%;border-radius:4px;"></div></div>'
            '<span style="font-family:Nunito;font-size:0.75rem;color:#EAEAEA;">'
            + str(pct) + '%</span></td>'
            '<td style="padding:6px;text-align:center;font-family:Nunito;'
            'font-size:0.8rem;">' + str(avg) + '</td>'
            '</tr>'
        )

    headers = ['RANK','','GEN','TOTAL','TIERED','TOP TIER','VIABLE','VIABLE %','AVG BST']
    ths = "".join(
        '<th style="padding:7px;font-family:Press Start 2P;font-size:0.5rem;'
        'color:#FFCB05;text-align:center;">' + h + '</th>'
        for h in headers)

    st.markdown(
        '<table style="width:100%;border-collapse:collapse;background:#0f3460;'
        'border-radius:8px;overflow:hidden;font-family:Nunito,sans-serif;color:#EAEAEA;">'
        '<thead><tr style="background:#CC0000;">' + ths + '</tr></thead>'
        '<tbody>' + rows + '</tbody></table>',
        unsafe_allow_html=True)

    # Caveat note
    st.markdown(
        '<div style="margin-top:1rem;border-left:3px solid #FFCB05;'
        'background:#0f3460;padding:0.7rem 1rem;border-radius:0 6px 6px 0;">'
        '<p style="font-family:Nunito;color:#A0A0A0;font-size:0.82rem;margin:0;">'
        'Rankings based on Smogon tier data covering Gens 1-8. '
        'Gen 9 excluded due to incomplete tier data. '
        'Gen 8 ranking may be understated as its dataset has thinner coverage '
        'than earlier generations.</p>'
        '</div>',
        unsafe_allow_html=True)

if __name__ == "__main__":
    main()