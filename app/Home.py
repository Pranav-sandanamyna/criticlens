import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ============================================================
# PAGE CONFIG — Must be first Streamlit call
# ============================================================
st.set_page_config(
    page_title="CriticLens",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS — Dark professional theme
# ============================================================
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }

    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #00d4aa;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #aaaaaa;
        margin-top: 4px;
    }

    .recommendation-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-left: 4px solid #00d4aa;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .recommendation-title {
        font-size: 1rem;
        font-weight: 700;
        color: #00d4aa;
        margin-bottom: 6px;
    }
    .recommendation-text {
        font-size: 0.9rem;
        color: #cccccc;
        line-height: 1.5;
    }

    .search-result {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 20px;
        margin-top: 16px;
    }

    .quadrant-safe    { color: #00d4aa; font-weight: 700; }
    .quadrant-hidden  { color: #4da6ff; font-weight: 700; }
    .quadrant-overhyped { color: #ff6b6b; font-weight: 700; }
    .quadrant-avoid   { color: #888888; font-weight: 700; }

    h1, h2, h3 { color: #ffffff !important; }
    .stSelectbox label { color: #aaaaaa; }
    div[data-testid="metric-container"] {
        background: #1a1a2e;
        border: 1px solid #2a2a4a;
        border-radius: 8px;
        padding: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def load_data():
    # Works both locally and on Streamlit Cloud
    possible_paths = [
    Path("data/processed/imdb_final.csv"),
    Path("../data/processed/imdb_final.csv"),
    Path(__file__).parent.parent / "data" / "processed" / "imdb_final.csv",
]
    for path in possible_paths:
        if path.exists():
            df = pd.read_csv(path)
            df["primary_genre"] = df["genre"].str.split(",").str[0].str.strip()
            df["votes"] = pd.to_numeric(
                df["votes"].astype(str).str.replace(",", ""), errors="coerce"
            )
            return df
    st.error("Data file not found. Make sure imdb_final.csv exists in data/processed/")
    st.stop()

df = load_data()

QUADRANT_COLORS = {
    "Safe Bet":   "#00d4aa",
    "Hidden Gem": "#4da6ff",
    "Overhyped":  "#ff6b6b",
    "Avoid":      "#888888",
}


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🎬 CriticLens")
    st.markdown("*Can critic sentiment predict a movie's streaming value?*")
    st.divider()

    st.markdown("### Filters")
    selected_genres = st.multiselect(
        "Filter by Genre",
        options=sorted(df["primary_genre"].dropna().unique()),
        default=[],
        placeholder="All genres"
    )

    year_min = int(df["year"].min())
    year_max = int(df["year"].max())
    year_range = st.slider(
        "Release Year Range",
        min_value=year_min,
        max_value=year_max,
        value=(year_min, year_max),
    )

    selected_quadrants = st.multiselect(
        "Filter by Quadrant",
        options=["Safe Bet", "Hidden Gem", "Overhyped", "Avoid"],
        default=["Safe Bet", "Hidden Gem", "Overhyped", "Avoid"],
    )

    st.divider()
    st.markdown("### About")
    st.markdown("""
    Built by **Pranav** as Step 1 of a larger data project.

    Tech stack: Python · BeautifulSoup ·
    VADER · Pandas · Plotly · Streamlit

    *Something bigger is coming. 👀*
    """)

# Apply filters
filtered_df = df.copy()
if selected_genres:
    filtered_df = filtered_df[filtered_df["primary_genre"].isin(selected_genres)]
filtered_df = filtered_df[
    (filtered_df["year"] >= year_range[0]) &
    (filtered_df["year"] <= year_range[1])
]
if selected_quadrants:
    filtered_df = filtered_df[filtered_df["quadrant"].isin(selected_quadrants)]


# ============================================================
# HEADER
# ============================================================
st.markdown("# 🎬 CriticLens")
st.markdown(
    "### Can critic sentiment predict a movie's long-term streaming value?"
)
st.markdown(
    "An analytics project analyzing IMDB Top 250 movies — "
    "combining sentiment analysis with longevity scoring to identify "
    "which films make the best long-term streaming licensing bets."
)
st.divider()


# ============================================================
# METRICS ROW
# ============================================================
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Movies Analyzed", len(filtered_df))
with col2:
    safe = len(filtered_df[filtered_df["quadrant"] == "Safe Bet"])
    st.metric("Safe Bets", safe)
with col3:
    gems = len(filtered_df[filtered_df["quadrant"] == "Hidden Gem"])
    st.metric("Hidden Gems", gems)
with col4:
    hyped = len(filtered_df[filtered_df["quadrant"] == "Overhyped"])
    st.metric("Overhyped", hyped)
with col5:
    avg_sent = filtered_df["sentiment_compound"].mean()
    st.metric("Avg Sentiment", f"{avg_sent:.3f}")

st.divider()


# ============================================================
# MOVIE SEARCH
# ============================================================
st.markdown("## 🔍 Movie Lookup")
st.markdown("Search any movie from the Top 250 to see its sentiment and longevity profile.")

movie_titles = [""] + sorted(df["title"].dropna().tolist())

selected_movie = st.selectbox(
    "🔍 Search a movie — type any letter to filter",
    options=movie_titles,
    index=0,
)

search_query = selected_movie

if search_query:
    results = df[df["title"].str.contains(search_query, case=False, na=False)]

    if results.empty:
        st.warning(f"No movies found matching '{search_query}'")
    else:
        for _, row in results.iterrows():
            quadrant_class = {
                "Safe Bet":   "quadrant-safe",
                "Hidden Gem": "quadrant-hidden",
                "Overhyped":  "quadrant-overhyped",
                "Avoid":      "quadrant-avoid",
            }.get(row["quadrant"], "")

            quadrant_color = QUADRANT_COLORS.get(row["quadrant"], "#ffffff")

            st.markdown(f"""
            <div class="search-result">
                <h3 style="color: white; margin-bottom: 4px;">
                    #{int(row['rank'])} {row['title']} ({int(row['year'])})
                </h3>
                <p style="color: #aaaaaa; margin-bottom: 16px;">
                    {row['genre']} · {int(row.get('runtime_minutes', 0))} min
                </p>
                <div style="display: flex; gap: 40px; flex-wrap: wrap;">
                    <div>
                        <div style="color: #aaaaaa; font-size: 0.8rem;">IMDB RATING</div>
                        <div style="color: #00d4aa; font-size: 1.5rem; font-weight: 700;">
                            ⭐ {row['rating']}
                        </div>
                    </div>
                    <div>
                        <div style="color: #aaaaaa; font-size: 0.8rem;">SENTIMENT SCORE</div>
                        <div style="color: #4da6ff; font-size: 1.5rem; font-weight: 700;">
                            {row['sentiment_compound']:.3f}
                        </div>
                    </div>
                    <div>
                        <div style="color: #aaaaaa; font-size: 0.8rem;">LONGEVITY SCORE</div>
                        <div style="color: #ffffff; font-size: 1.5rem; font-weight: 700;">
                            {row['longevity_score']:.1f}/100
                        </div>
                    </div>
                    <div>
                        <div style="color: #aaaaaa; font-size: 0.8rem;">LICENSING SIGNAL</div>
                        <div style="color: {quadrant_color}; font-size: 1.5rem; font-weight: 700;">
                            {row['quadrant']}
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

st.divider()


# ============================================================
# VISUAL 1 — QUADRANT SCATTER PLOT
# ============================================================
st.markdown("## 📊 Sentiment vs Longevity — Quadrant Analysis")
st.markdown(
    "Each dot is a movie. Position shows its critic sentiment score vs "
    "long-term audience engagement. Color shows its licensing signal."
)

sentiment_median = filtered_df["sentiment_compound"].median()
longevity_median = filtered_df["longevity_score"].median()

fig_scatter = px.scatter(
    filtered_df,
    x="sentiment_compound",
    y="longevity_score",
    color="quadrant",
    color_discrete_map=QUADRANT_COLORS,
    hover_data={
        "title": True,
        "year": True,
        "rating": True,
        "sentiment_compound": ":.3f",
        "longevity_score": ":.1f",
        "genre": True,
        "quadrant": True,
    },
    labels={
        "sentiment_compound": "Critic Sentiment Score (VADER)",
        "longevity_score":    "Longevity Score (Votes/Year Normalized)",
        "quadrant":           "Licensing Signal",
    },
    title="CriticLens Quadrant — Streaming Licensing Signal Map",
)

fig_scatter.add_vline(
    x=sentiment_median,
    line_dash="dash",
    line_color="#555555",
    annotation_text="Sentiment Median",
    annotation_font_color="#888888",
)
fig_scatter.add_hline(
    y=longevity_median,
    line_dash="dash",
    line_color="#555555",
    annotation_text="Longevity Median",
    annotation_font_color="#888888",
)

fig_scatter.update_layout(
    plot_bgcolor="#0f1117",
    paper_bgcolor="#0f1117",
    font_color="#ffffff",
    height=550,
    legend=dict(
        bgcolor="#1a1a2e",
        bordercolor="#2a2a4a",
        borderwidth=1,
    ),
)
fig_scatter.update_traces(
    marker=dict(size=9, opacity=0.85, line=dict(width=0.5, color="#ffffff"))
)

st.plotly_chart(fig_scatter, use_container_width=True)
st.divider()


# ============================================================
# VISUAL 2 — GENRE BREAKDOWN
# ============================================================
st.markdown("## 🎭 Genre Analysis")
st.markdown(
    "Average sentiment and longevity scores by genre. "
    "Shows where critic language is a reliable signal for streaming value."
)

genre_df = filtered_df.groupby("primary_genre").agg(
    avg_sentiment=("sentiment_compound", "mean"),
    avg_longevity=("longevity_score", "mean"),
    count=("title", "count"),
).reset_index()
genre_df = genre_df[genre_df["count"] >= 3].sort_values("avg_longevity", ascending=False)

col_g1, col_g2 = st.columns(2)

with col_g1:
    fig_genre_long = px.bar(
        genre_df.sort_values("avg_longevity"),
        x="avg_longevity",
        y="primary_genre",
        orientation="h",
        color="avg_longevity",
        color_continuous_scale=["#1a1a2e", "#00d4aa"],
        labels={"avg_longevity": "Avg Longevity Score", "primary_genre": "Genre"},
        title="Avg Longevity Score by Genre",
    )
    fig_genre_long.update_layout(
        plot_bgcolor="#0f1117",
        paper_bgcolor="#0f1117",
        font_color="#ffffff",
        showlegend=False,
        coloraxis_showscale=False,
        height=400,
    )
    st.plotly_chart(fig_genre_long, use_container_width=True)

with col_g2:
    fig_genre_sent = px.bar(
        genre_df.sort_values("avg_sentiment"),
        x="avg_sentiment",
        y="primary_genre",
        orientation="h",
        color="avg_sentiment",
        color_continuous_scale=["#ff6b6b", "#1a1a2e", "#4da6ff"],
        labels={"avg_sentiment": "Avg Sentiment Score", "primary_genre": "Genre"},
        title="Avg Sentiment Score by Genre",
    )
    fig_genre_sent.update_layout(
        plot_bgcolor="#0f1117",
        paper_bgcolor="#0f1117",
        font_color="#ffffff",
        showlegend=False,
        coloraxis_showscale=False,
        height=400,
    )
    st.plotly_chart(fig_genre_sent, use_container_width=True)

st.divider()


# ============================================================
# VISUAL 3 — OUTLIER TABLES
# ============================================================
st.markdown("## 🔥 Overhyped vs Hidden Gems")
st.markdown(
    "The most actionable licensing insights — "
    "movies that defied critic expectations in both directions."
)

col_t1, col_t2 = st.columns(2)

display_cols = ["title", "year", "rating", "sentiment_compound", "longevity_score"]
col_rename = {
    "title":              "Title",
    "year":               "Year",
    "rating":             "Rating",
    "sentiment_compound": "Sentiment",
    "longevity_score":    "Longevity",
}

with col_t1:
    st.markdown("### 🔴 Overhyped — High Sentiment, Low Longevity")
    st.markdown("*Bad long-term licensing bets despite positive buzz*")
    overhyped_df = (
        filtered_df[filtered_df["quadrant"] == "Overhyped"]
        .nlargest(10, "sentiment_compound")[display_cols]
        .rename(columns=col_rename)
        .reset_index(drop=True)
    )
    overhyped_df.index += 1
    st.dataframe(overhyped_df, use_container_width=True, height=380)

with col_t2:
    st.markdown("### 🔵 Hidden Gems — Low Sentiment, High Longevity")
    st.markdown("*Undervalued licensing opportunities*")
    gems_df = (
        filtered_df[filtered_df["quadrant"] == "Hidden Gem"]
        .nlargest(10, "longevity_score")[display_cols]
        .rename(columns=col_rename)
        .reset_index(drop=True)
    )
    gems_df.index += 1
    st.dataframe(gems_df, use_container_width=True, height=380)

st.divider()


# ============================================================
# VISUAL 4 — CORRELATION HEATMAP
# ============================================================
st.markdown("## 🧮 Correlation Matrix")
st.markdown(
    "Relationships between sentiment, longevity, rating, and votes. "
    "Shows which signals are truly independent predictors."
)

corr_cols = ["rating", "votes", "sentiment_compound",
             "sentiment_intensity", "longevity_score", "age_years"]
corr_cols = [c for c in corr_cols if c in filtered_df.columns]

corr_matrix = filtered_df[corr_cols].corr().round(2)
label_map = {
    "rating":               "IMDB Rating",
    "votes":                "Total Votes",
    "sentiment_compound":   "Sentiment",
    "sentiment_intensity":  "Sent. Intensity",
    "longevity_score":      "Longevity",
    "age_years":            "Movie Age",
}
corr_matrix.index   = [label_map.get(c, c) for c in corr_matrix.index]
corr_matrix.columns = [label_map.get(c, c) for c in corr_matrix.columns]

fig_heatmap = go.Figure(data=go.Heatmap(
    z=corr_matrix.values,
    x=corr_matrix.columns.tolist(),
    y=corr_matrix.index.tolist(),
    colorscale="RdBu",
    zmid=0,
    text=corr_matrix.values,
    texttemplate="%{text:.2f}",
    textfont={"size": 11, "color": "white"},
    hoverongaps=False,
))

fig_heatmap.update_layout(
    plot_bgcolor="#0f1117",
    paper_bgcolor="#0f1117",
    font_color="#ffffff",
    height=420,
    title="Correlation Matrix — Key Variables",
)

st.plotly_chart(fig_heatmap, use_container_width=True)
st.divider()


# ============================================================
# BUSINESS RECOMMENDATIONS
# ============================================================
st.markdown("## 💡 Business Recommendations")
st.markdown(
    "Actionable insights for a streaming platform licensing team "
    "based on the sentiment-longevity analysis."
)

best_genre  = genre_df.loc[genre_df["avg_longevity"].idxmax(), "primary_genre"]
worst_genre = genre_df.loc[genre_df["avg_longevity"].idxmin(), "primary_genre"]
n_gems      = len(df[df["quadrant"] == "Hidden Gem"])
n_hyped     = len(df[df["quadrant"] == "Overhyped"])

recommendations = [
    {
        "title": f"01 — Prioritize {best_genre} for Long-Term Licensing",
        "text": (
            f"{best_genre} films show the highest longevity scores in this dataset, "
            "meaning audiences continue engaging with them decades after release. "
            "These represent the strongest long-term licensing ROI and should be "
            "prioritized in catalogue acquisition strategies."
        ),
    },
    {
        "title": "02 — Do Not License Based on Sentiment Score Alone",
        "text": (
            f"{n_hyped} movies show high critic sentiment but below-median longevity. "
            "Positive buzz at release does not guarantee sustained audience engagement. "
            "Sentiment should be treated as one signal among many, weighted alongside "
            "vote trajectory and genre benchmarks."
        ),
    },
    {
        "title": "03 — Target Hidden Gems for Undervalued Catalogue Deals",
        "text": (
            f"{n_gems} movies show low sentiment scores but high longevity — "
            "significantly outperforming critic expectations. These titles are likely "
            "underpriced in licensing negotiations and represent high-ROI opportunities "
            "for long-term catalogue acquisition at below-market rates."
        ),
    },
]

for rec in recommendations:
    st.markdown(f"""
    <div class="recommendation-card">
        <div class="recommendation-title">{rec['title']}</div>
        <div class="recommendation-text">{rec['text']}</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()


# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div style="text-align: center; color: #555555; padding: 20px 0;">
    Built by Pranav · BBA Computational Business Analytics · Mahindra University
    <br>
    <span style="color: #333333; font-size: 0.8rem;">
        This is Step 1. Something bigger is coming. 👀
    </span>
</div>
""", unsafe_allow_html=True)