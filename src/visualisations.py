"""
visualisations.py — Reusable Plotly chart functions.

Each function takes a DataFrame (from data_loader.load_featured()) and returns
a Plotly figure. Pure functions, no file I/O, no Streamlit dependencies —
so they work identically in notebooks and the Streamlit app.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


# Consistent colour scheme across all charts
COLOUR_SEQUENCE = px.colors.qualitative.Set2

# Official Pokémon type colours — used whenever we colour by type
TYPE_COLOURS = {
    "Normal": "#A8A77A", "Fire": "#EE8130", "Water": "#6390F0",
    "Electric": "#F7D02C", "Grass": "#7AC74C", "Ice": "#96D9D6",
    "Fighting": "#C22E28", "Poison": "#A33EA1", "Ground": "#E2BF65",
    "Flying": "#A98FF3", "Psychic": "#F95587", "Bug": "#A6B91A",
    "Rock": "#B6A136", "Ghost": "#735797", "Dragon": "#6F35FC",
    "Dark": "#705746", "Steel": "#B7B7CE", "Fairy": "#D685AD",
    "None": "#CCCCCC",
}

# Category ordering — weakest-to-strongest archetype, useful for stacked plots
LEGENDARY_CATEGORY_ORDER = ["None", "Ultra Beast", "Paradox", "Mythical", "Traditional"]

LEGENDARY_CATEGORY_COLOURS = {
    "None": "#6390F0",
    "Ultra Beast": "#A33EA1",
    "Paradox": "#EE8130",
    "Mythical": "#F7D02C",
    "Traditional": "#F95587",
}


def plot_bst_by_generation(df: pd.DataFrame, base_forms_only: bool = True) -> go.Figure:
    """
    Box plot of BST per generation, split by legendary category.

    Args:
        df: Feature-engineered Pokémon DataFrame (from load_featured)
        base_forms_only: If True, filter to canonical base forms.
                         Recommended for generation-level aggregates.

    Returns:
        Plotly Figure ready for fig.show() or st.plotly_chart().
    """
    if base_forms_only:
        df = df[df['form_type'] == 'Base'].copy()

    fig = px.box(
        df,
        x="generation",
        y="bst",
        color="legendary_category",
        category_orders={"legendary_category": LEGENDARY_CATEGORY_ORDER},
        color_discrete_map=LEGENDARY_CATEGORY_COLOURS,
        title="Base Stat Total by Generation — Split by Legendary Category",
        labels={"bst": "Base Stat Total", "generation": "Generation",
                "legendary_category": "Category"},
        points="outliers",
    )
    fig.update_layout(height=550, boxmode="group", legend_title_text="Category")
    return fig

def plot_type_composition_absolute(df: pd.DataFrame, base_forms_only: bool = True) -> go.Figure:
    """Stacked area chart of type counts per generation."""
    if base_forms_only:
        df = df[df['form_type'] == 'Base'].copy()

    type_columns = [c for c in df.columns if c.startswith('has_type_')]
    gen_type_counts = (
        df.groupby('generation')[type_columns]
        .sum()
        .rename(columns=lambda c: c.replace('has_type_', ''))
    )

    gen_type_long = (
        gen_type_counts
        .reset_index()
        .melt(id_vars='generation', var_name='type', value_name='count')
    )

    fig = px.area(
        gen_type_long,
        x='generation', y='count', color='type',
        color_discrete_map=TYPE_COLOURS,
        title="Type Composition Across Generations — Stacked Counts",
        labels={"count": "Pokémon with this type", "generation": "Generation"},
        category_orders={"type": list(TYPE_COLOURS.keys())},
    )
    fig.update_layout(height=600, legend_title_text="Type")
    return fig


def plot_type_frequency_percentage(df: pd.DataFrame, base_forms_only: bool = True) -> go.Figure:
    """Line chart of type frequency as % of generation."""
    if base_forms_only:
        df = df[df['form_type'] == 'Base'].copy()

    type_columns = [c for c in df.columns if c.startswith('has_type_')]
    gen_type_counts = (
        df.groupby('generation')[type_columns]
        .sum()
        .rename(columns=lambda c: c.replace('has_type_', ''))
    )
    gen_totals = df.groupby('generation').size()
    gen_type_pct = gen_type_counts.div(gen_totals, axis=0) * 100

    gen_type_pct_long = (
        gen_type_pct
        .reset_index()
        .melt(id_vars='generation', var_name='type', value_name='percentage')
    )

    fig = px.line(
        gen_type_pct_long,
        x='generation', y='percentage', color='type',
        color_discrete_map=TYPE_COLOURS,
        title="Type Frequency as % of Generation — Evolution Over Time",
        labels={"percentage": "% of gen with this type", "generation": "Generation"},
        markers=True,
    )
    fig.update_layout(height=600, legend_title_text="Type", yaxis_ticksuffix="%")
    return fig

def plot_legendary_gap(df: pd.DataFrame, base_forms_only: bool = True) -> go.Figure:
    """
    Line chart of mean BST per generation, split by legendary category.

    Shows the 'legendary gap' — how much stronger legendaries are than
    non-legendaries, and how that gap has evolved across generations.
    """
    if base_forms_only:
        df = df[df['form_type'] == 'Base'].copy()

    gap_data = (
        df.groupby(['generation', 'legendary_category'])['bst']
        .mean()
        .unstack()
    )

    fig = go.Figure()

    category_style = {
        'Traditional':  dict(color='#F95587', width=3, dash='solid', size=10),
        'None':         dict(color='#6390F0', width=3, dash='solid', size=10),
        'Mythical':     dict(color='#F7D02C', width=2, dash='dash',  size=8),
        'Ultra Beast':  dict(color='#A33EA1', width=2, dash='dot',   size=8),
        'Paradox':      dict(color='#EE8130', width=2, dash='dot',   size=8),
    }

    display_names = {'None': 'Non-Legendaries'}

    for category in ['Traditional', 'None', 'Mythical', 'Ultra Beast', 'Paradox']:
        if category in gap_data.columns:
            style = category_style[category]
            fig.add_trace(go.Scatter(
                x=gap_data.index,
                y=gap_data[category],
                name=display_names.get(category, category),
                line=dict(color=style['color'], width=style['width'], dash=style['dash']),
                mode='lines+markers',
                marker=dict(size=style['size']),
            ))

    fig.update_layout(
        title="Mean BST by Generation — Legendary Categories vs. Non-Legendaries",
        xaxis_title="Generation",
        yaxis_title="Mean Base Stat Total",
        height=550,
        hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                    bgcolor="rgba(255,255,255,0.8)"),
    )
    return fig


def plot_legendary_gap_median(df: pd.DataFrame, base_forms_only: bool = True) -> go.Figure:
    """
    Bar chart of median BST gap (Traditional legendaries minus non-legendaries)
    per generation. Uses median to avoid Cosmog/Cosmoem distortion in Gen 7.
    """
    if base_forms_only:
        df = df[df['form_type'] == 'Base'].copy()

    gap_data = (
        df.groupby(['generation', 'legendary_category'])['bst']
        .median()
        .unstack()
    )

    gap_data['median_gap'] = gap_data['Traditional'] - gap_data['None']

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=gap_data.index,
        y=gap_data['median_gap'],
        marker_color='#F95587',
        name='Median BST Gap',
    ))

    fig.update_layout(
        title="Legendary Power Gap per Generation<br><sup>Traditional Median − Non-Legendary Median</sup>",
        xaxis_title="Generation",
        yaxis_title="BST Gap (Median)",
        height=450,
        showlegend=False,
        annotations=[
            dict(
                x=5.5, y=265,
                text="Gap peaks Gen 6 (253)",
                showarrow=True, arrowhead=2,
                ax=60, ay=-30,
                font=dict(size=11),
            ),
            dict(
                x=8, y=105,
                text="Smallest gap: Gen 8 (105)",
                showarrow=True, arrowhead=2,
                ax=-60, ay=-30,
                font=dict(size=11),
            ),
        ]
    )
    return fig

def plot_stat_profiles(df: pd.DataFrame, base_forms_only: bool = True,
                       legendaries: bool = False) -> go.Figure:
    """
    Stacked bar chart of stat profile distribution per generation.

    Args:
        df: Feature-engineered DataFrame.
        base_forms_only: Filter to canonical base forms (recommended).
        legendaries: If False (default), excludes legendary-category Pokémon
                     so legendary stat extremes don't skew the picture.

    Returns:
        Plotly Figure.
    """
    if base_forms_only:
        df = df[df['form_type'] == 'Base'].copy()
    if not legendaries:
        df = df[df['legendary_category'] == 'None'].copy()

    profile_order = [
        'Physical Sweeper', 'Special Sweeper', 'Mixed Attacker',
        'Balanced',
        'Physical Wall', 'Special Wall', 'Defensive Wall'
    ]

    profile_colours = {
        'Physical Sweeper': '#EE8130',
        'Special Sweeper':  '#F95587',
        'Mixed Attacker':   '#F7D02C',
        'Balanced':         '#6390F0',
        'Physical Wall':    '#7AC74C',
        'Special Wall':     '#96D9D6',
        'Defensive Wall':   '#A8A77A',
    }

    profile_counts = (
        df.groupby(['generation', 'stat_profile'])
        .size()
        .unstack(fill_value=0)
    )

    profile_pct = profile_counts.div(profile_counts.sum(axis=1), axis=0) * 100

    profile_pct_long = (
        profile_pct
        .reset_index()
        .melt(id_vars='generation', var_name='stat_profile', value_name='percentage')
    )

    fig = px.bar(
        profile_pct_long,
        x='generation',
        y='percentage',
        color='stat_profile',
        color_discrete_map=profile_colours,
        category_orders={'stat_profile': profile_order},
        title="Stat Profile Distribution per Generation — Non-Legendary Base Forms",
        labels={"percentage": "% of generation", "generation": "Generation",
                "stat_profile": "Stat Profile"},
        barmode='stack',
    )
    fig.update_layout(
        height=550,
        yaxis_ticksuffix="%",
        legend_title_text="Stat Profile"
    )
    return fig

# Offensive type coverage — how many types each type hits super-effectively
_OFFENSIVE_COVERAGE = {
    "Normal":   [],
    "Fire":     ["Grass", "Ice", "Bug", "Steel"],
    "Water":    ["Fire", "Ground", "Rock"],
    "Electric": ["Water", "Flying"],
    "Grass":    ["Water", "Ground", "Rock"],
    "Ice":      ["Grass", "Ground", "Flying", "Dragon"],
    "Fighting": ["Normal", "Ice", "Rock", "Dark", "Steel"],
    "Poison":   ["Grass", "Fairy"],
    "Ground":   ["Fire", "Electric", "Poison", "Rock", "Steel"],
    "Flying":   ["Grass", "Fighting", "Bug"],
    "Psychic":  ["Fighting", "Poison"],
    "Bug":      ["Grass", "Psychic", "Dark"],
    "Rock":     ["Fire", "Ice", "Flying", "Bug"],
    "Ghost":    ["Psychic", "Ghost"],
    "Dragon":   ["Dragon"],
    "Dark":     ["Psychic", "Ghost"],
    "Steel":    ["Ice", "Rock", "Fairy"],
    "Fairy":    ["Fighting", "Dragon", "Dark"],
    "None":     [],
}


def _count_super_effective(type_1: str, type_2: str) -> int:
    """Count unique types hit super-effectively by a dual typing."""
    coverage = set(_OFFENSIVE_COVERAGE.get(type_1, []))
    coverage |= set(_OFFENSIVE_COVERAGE.get(type_2, []))
    return len(coverage)


def plot_type_coverage(df: pd.DataFrame, base_forms_only: bool = True,
                       legendaries: bool = False) -> go.Figure:
    """
    Line chart of mean offensive type coverage per generation.

    Coverage = number of unique types hit super-effectively by a Pokémon's
    dual typing. Error bars show standard deviation within each generation.
    """
    if base_forms_only:
        df = df[df['form_type'] == 'Base'].copy()
    if not legendaries:
        df = df[df['legendary_category'] == 'None'].copy()

    df = df.copy()
    df['coverage_count'] = df.apply(
        lambda row: _count_super_effective(row['type_1'], row['type_2']), axis=1
    )

    coverage_by_gen = (
        df.groupby('generation')['coverage_count']
        .agg(['mean', 'std'])
        .round(2)
        .reset_index()
    )

    fig = px.line(
        coverage_by_gen,
        x='generation',
        y='mean',
        markers=True,
        error_y='std',
        title="Mean Offensive Type Coverage per Generation — Non-Legendary Base Forms",
        labels={"mean": "Mean types hit super-effectively", "generation": "Generation"},
    )
    fig.update_traces(line=dict(color='#EE8130', width=3), marker=dict(size=10))
    fig.update_layout(height=450, yaxis_range=[0, 8])
    return fig

def plot_dual_type_prevalence(df: pd.DataFrame, base_forms_only: bool = True,
                               legendaries: bool = False) -> go.Figure:
    """
    Bar chart of dual-type Pokémon percentage per generation.
    Red dashed line marks the overall mean across all generations.
    """
    if base_forms_only:
        df = df[df['form_type'] == 'Base'].copy()
    if not legendaries:
        df = df[df['legendary_category'] == 'None'].copy()

    dual_type_by_gen = (
        df.groupby('generation')['num_types']
        .agg(
            dual_count=lambda x: (x == 2).sum(),
            total='count',
        )
    )
    dual_type_by_gen['dual_pct'] = (
        dual_type_by_gen['dual_count'] / dual_type_by_gen['total'] * 100
    ).round(1)

    overall_mean = dual_type_by_gen['dual_pct'].mean()

    fig = px.bar(
        dual_type_by_gen.reset_index(),
        x='generation',
        y='dual_pct',
        title="Dual-Type Prevalence per Generation — Non-Legendary Base Forms",
        labels={"dual_pct": "% Dual-Type", "generation": "Generation"},
        color='dual_pct',
        color_continuous_scale='Blues',
    )
    fig.update_layout(
        height=450,
        yaxis_ticksuffix="%",
        coloraxis_showscale=False,
    )
    fig.add_hline(
        y=overall_mean,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Overall mean: {overall_mean:.1f}%",
        annotation_position="top right",
    )
    return fig