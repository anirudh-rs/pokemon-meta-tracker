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
from src.model import load_model, predict_viability, build_feature_row, get_top_shap_features

# -- Page config ---------------------------------------------------------------
st.set_page_config(
    page_title="PokeMeta Analyser",
    page_icon="~",
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
    font-size:0.42rem !important;
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
def sh(title, sub="", pid=None):
    if pid:
        c1, c2 = st.columns([5, 1])
        with c1:
            st.markdown("### " + title)
            if sub:
                st.markdown(
                    "<p style='color:#A0A0A0;font-size:0.85rem;margin-top:-0.4rem;'>" + sub + "</p>",
                    unsafe_allow_html=True)
        with c2:
            st.markdown(
                '<div style="text-align:right;margin-top:-0.4rem;">' + sprite_img(pid, 64) + '</div>',
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
    sh("POKEMETA ANALYSER", "Competitive meta shifts across all 9 generations")

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Total Pokemon",       str(len(df[df['form_type']=='Base'])))
    with c2: st.metric("Generations",          "9")
    with c3: st.metric("Tiered Pokemon",       str(int(df_t['is_viable'].notna().sum())))
    with c4: st.metric("Competitively Viable", str(int((df_t['is_viable']==True).sum())))

    divider()

    cl, cr = st.columns([3, 2])
    with cl:
        st.markdown("## ABOUT THIS PROJECT")
        lines = [
            "<div style='font-family:Nunito,sans-serif;font-size:0.95rem;line-height:1.9;'>",
            "<p>This dashboard tracks how the Pokemon competitive meta has evolved across all nine",
            "generations, from the original 151 in Red and Blue to the 1025+ of Scarlet and Violet.</p>",
            "<p>Using machine learning and data analysis, we explore:</p>",
            "<ul>",
            "<li>Which generations introduced the most powerful Pokemon</li>",
            "<li>How type dominance has shifted over 25+ years</li>",
            "<li>Whether the legendary power gap is growing or closing</li>",
            "<li>What stats actually determine competitive viability</li>",
            "</ul>",
            "<p>The classifier was trained on Smogon tiers from Gens 1-7, validated on Gen 8,",
            "and achieves a <strong style='color:#FFCB05'>ROC-AUC of 0.932</strong>.</p>",
            "</div>",
        ]
        st.markdown(" ".join(lines), unsafe_allow_html=True)

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
        findings = [
            ("Power Creep",      "Non-legendary BST median rose 48 pts across 9 gens"),
            ("Gap Closing",      "Legendary gap peaked Gen 6 (253 BST), now 116"),
            ("Fairy Disruption", "1.3% to 17.6% of Gen 6 in one generation"),
            ("Speed Rules",      "2nd most important viability factor after BST"),
        ]
        icons = ["up", "sword", "fairy", "bolt"]
        emojis = ["📈", "⚔️", "🧚", "⚡"]
        for (title, desc), emoji in zip(findings, emojis):
            st.markdown(
                '<div style="background:#0f3460;border-left:3px solid #CC0000;'
                'padding:0.5rem 0.7rem;margin-bottom:0.5rem;border-radius:0 6px 6px 0;">'
                '<span style="font-size:0.9rem">' + emoji + '</span>'
                '<strong style="color:#FFCB05;font-family:Nunito"> ' + title + '</strong><br>'
                '<span style="color:#A0A0A0;font-size:0.8rem;font-family:Nunito">' + desc + '</span>'
                '</div>',
                unsafe_allow_html=True)

    divider()
    st.markdown("## DATA SOURCES")
    cards = [
        ("Pokemon Dataset",  "1,194 Pokemon - All 9 Gens<br>Base stats, types, forms"),
        ("Smogon Tiers",     "606 tiered Pokemon - Gens 1-8<br>OU, UU, Uber, RU, NU, PU"),
        ("XGBoost Model",    "Trained Gens 1-7 - Tested Gen 8<br>ROC-AUC: 0.932"),
    ]
    for col, (t, b) in zip(st.columns(3), cards):
        with col:
            st.markdown(
                '<div style="background:#0f3460;border:1px solid #FFCB05;'
                'padding:0.9rem;border-radius:8px;text-align:center;">'
                '<p style="color:#FFCB05;font-family:Nunito;font-weight:700;margin-bottom:4px;">' + t + '</p>'
                '<p style="color:#A0A0A0;font-size:0.82rem;font-family:Nunito;margin:0;">' + b + '</p>'
                '</div>',
                unsafe_allow_html=True)

# -- TAB 2: GENERATION OVERVIEW ------------------------------------------------
def tab_generations(df):
    sh("GENERATION OVERVIEW", "How has Pokemon power evolved across generations?", pid=6)

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
            "font-size:0.45rem;color:#FFCB05;'>" + str(g) + "</td>"
            "<td style='padding:5px;text-align:center;'>" + str(int(r['Total'])) + "</td>"
            "<td style='padding:5px;text-align:center;color:" + bc + ";'>" + str(r['Avg BST']) + "</td>"
            "<td style='padding:5px;text-align:center;'>" + str(r['Med BST']) + "</td>"
            "<td style='padding:5px;text-align:center;color:#F95587;'>" + str(int(r['Legs'])) + "</td>"
            "<td style='padding:5px;text-align:center;'>" + str(r['Dual%']) + "%</td>"
            "</tr>"
        )

    headers = ['', 'GEN', 'COUNT', 'AVG BST', 'MED BST', 'LEGS', 'DUAL%']
    ths = "".join(
        '<th style="padding:7px;font-family:Press Start 2P;font-size:0.38rem;color:#FFCB05;">'
        + h + '</th>' for h in headers)
    st.markdown(
        '<table style="width:100%;border-collapse:collapse;background:#0f3460;'
        'border-radius:8px;overflow:hidden;font-family:Nunito,sans-serif;color:#EAEAEA;">'
        '<thead><tr style="background:#CC0000;">' + ths + '</tr></thead>'
        '<tbody>' + rows + '</tbody></table>',
        unsafe_allow_html=True)

# -- TAB 3: TYPE DOMINANCE -----------------------------------------------------
def tab_types(df):
    sh("TYPE DOMINANCE", "How has type composition shifted across generations?", pid=130)

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
       "How have legendary Pokemon disrupted competitive balance?", pid=150)

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
       "Are modern Pokemon more specialised than earlier generations?", pid=445)

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

# -- TAB 6: VIABILITY PREDICTOR ------------------------------------------------
def tab_predictor(df, model_data):
    sh("VIABILITY PREDICTOR",
       "Design a Pokemon and see if it would be competitively viable", pid=778)

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
        "<p style='font-family:Press Start 2P;font-size:0.42rem;color:#FFCB05;"
        "margin-bottom:0.3rem;'>QUICK PRESETS</p>",
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
        "<p style='font-family:Press Start 2P;font-size:0.42rem;color:#FFCB05;"
        "margin-bottom:0.3rem;'>STATS AND TYPING</p>",
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
            '<div style="background:#0f3460;border:1px solid #FFCB05;padding:0.35rem;'
            'border-radius:6px;text-align:center;margin-top:0.15rem;">'
            '<span style="font-family:Press Start 2P;font-size:0.38rem;color:#A0A0A0;">BST</span><br>'
            '<span style="font-family:Press Start 2P;font-size:0.95rem;color:#FFCB05;">'
            + str(bst) + '</span></div>',
            unsafe_allow_html=True)
        st.markdown("<div style='margin-top:0.3rem;'></div>", unsafe_allow_html=True)
        go_btn = st.button("ANALYSE", use_container_width=True, key="analyse_btn")

    divider()

    if go_btn:
        st.session_state["show_results"] = True

    if st.session_state.get("show_results"):
        with st.spinner("Consulting the Pokedex..."):
            try:
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
                    '<p style="font-family:Press Start 2P;font-size:0.42rem;color:#A0A0A0;margin:0;">'
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
    sh("MODEL INSIGHTS", "How does the XGBoost viability classifier work?", pid=248)
    import shap
    mdl, fc = model_data['model'], model_data['feature_cols']

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("ROC-AUC",    "0.932")
    with c2: st.metric("Accuracy",   "86.6%")
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
            [[64,7],[4,7]],
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

    st.markdown(
        '<div style="text-align:center;padding:0.4rem 0 0.3rem;">'
        '<h1 style="font-size:0.95rem !important;">~ POKEMETA ANALYSER ~</h1>'
        '<p style="font-family:Nunito;color:#A0A0A0;font-size:0.85rem;margin-top:-0.3rem;">'
        'Competitive Meta Shift Tracker - Gens 1 to 9</p>'
        '</div>',
        unsafe_allow_html=True)

    @st.cache_data
    def get_data():
        return load_featured(), load_tiered()

    @st.cache_resource
    def get_model():
        return load_model()

    with st.spinner("Loading Pokedex..."):
        df, df_t = get_data()
        md       = get_model()

    tabs = st.tabs([
        "HOME", "GENERATIONS", "TYPE DOMINANCE",
        "LEGENDARY IMPACT", "STAT PROFILES",
        "VIABILITY PREDICTOR", "MODEL INSIGHTS",
    ])

    with tabs[0]: tab_home(df, df_t)
    with tabs[1]: tab_generations(df)
    with tabs[2]: tab_types(df)
    with tabs[3]: tab_legendary(df)
    with tabs[4]: tab_profiles(df)
    with tabs[5]: tab_predictor(df, md)
    with tabs[6]: tab_model(df, df_t, md)

if __name__ == "__main__":
    main()