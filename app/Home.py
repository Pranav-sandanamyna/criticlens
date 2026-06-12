import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import html

# ============================================================
# PAGE CONFIG — must be the first Streamlit command
# ============================================================
st.set_page_config(
    page_title="CriticLens",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS — dark professional theme
# ============================================================
st.markdown(
    """
<style>
    .main { background-color: #0f1117; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }

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

    h1, h2, h3 { color: #ffffff !important; }
    .stSelectbox label { color: #aaaaaa; }
    div[data-testid="metric-container"] {
        background: #1a1a2e;
        border: 1px solid #2a2a4a;
        border-radius: 8px;
        padding: 12px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================
def find_data_file() -> Path | None:
    """Find data/processed/imdb_final.csv locally or on Streamlit Cloud."""
    filename = Path("data") / "processed" / "imdb_final.csv"

    roots = [Path.cwd()]

    try:
        current_file = Path(__file__).resolve()
        roots.extend([current_file.parent, *current_file.parents])
    except NameError:
        pass

    seen = set()
    for root in roots:
        candidate = (root / filename).resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate

    return None


def safe_int(value, default=0):
    try:
        if pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def load_data():
    data_path = find_data_file()

    if data_path is None:
        st.error(
            "Data file not found. Make sure this file exists in your GitHub repo: "
            "data/processed/imdb_final.csv"
        )
        st.stop()

    df = pd.read_csv(data_path)

    required_columns = {
        "title",
        "year",
        "genre",
        "rating",
        "rank",
        "quadrant",
        "sentiment_compound",
        "longevity_score",
    }
    missing = required_columns - set(df.columns)
    if missing:
        st.error(f"Missing required columns in imdb_final.csv: {sorted(missing)}")
        st.stop()

    # Clean and standardize columns
    df["primary_genre"] = df["genre"].astype(str).str.split(",").str[0].str.strip()

    numeric_cols = [
        "rank",
        "year",
        "rating",
        "votes",
        "runtime_minutes",
        "sentiment_compound",
        "sentiment_intensity",
        "longevity_score",
        "age_years",
    ]
    for col in numeric_cols:
        if col in df.columns:
            if col == "votes":
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace(r"[^0-9.]", "", regex=True),
                    errors="coerce",
                )
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


df = load_data()

QUADRANT_COLORS = {
    "Safe Bet": "#00d4aa",
    "Hidden Gem": "#4da6ff",
    "Overhyped": "#ff6b6b",
    "Avoid": "#888888",
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
        placeholder="All genres",
    )

    valid_years = df["year"].dropna()
    year_min = safe_int(valid_years.min(), 1900)
    year_max = safe_int(valid_years.max(), 2025)

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
    st.markdown(
        """
        Built by **Pranav** as Step 1 of a larger data project.

        Tech stack: Python · BeautifulSoup · VADER · Pandas · Plotly · Streamlit

        *Something bigger is coming. 👀*
        """
    )


# ============================================================
# APPLY FILTERS
# ============================================================
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
st.markdown("### Can critic sentiment predict a movie's long-term streaming value?")
st.markdown(
    "An analytics project analyzing IMDB Top 250 movies — combining sentiment analysis "
    "with longevity scoring to identify which films make the best long-term streaming "
    "licensing bets."
)
st.divider()

if filtered_df.empty:
    st.warning("No movies match your current filters. Try changing the sidebar filters.")
    st.stop()


# ============================================================
# METRICS ROW
# ============================================================
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Movies Analyzed", len(filtered_df))
with col2:
    st.metric("Safe Bets", len(filtered_df[filtered_df["quadrant"] == "Safe Bet"]))
with col3:
    st.metric("Hidden Gems", len(filtered_df[filtered_df["quadrant"] == "Hidden Gem"]))
with col4:
    st.metric("Overhyped", len(filtered_df[filtered_df["quadrant"] == "Overhyped"]))
with col5:
    avg_sent = filtered_df["sentiment_compound"].mean()
    st.metric("Avg Sentiment", f"{avg_sent:.3f}")

st.divider()


# ============================================================
# MOVIE SEARCH
# ============================================================
st.markdown("## 🔍 Movie Lookup")
st.markdown("Search any movie from the Top 250 to see its sentiment and longevity profile.")

movie_titles = [""] + sorted(df["title"].dropna().astype(str).tolist())
selected_movie = st.selectbox(
    "🔍 Search a movie — type any letter to filter",
    options=movie_titles,
    index=0,
)

if selected_movie:
    results = df[df["title"].astype(str).str.contains(selected_movie, case=False, na=False)]

    if results.empty:
        st.warning(f"No movies found matching '{selected_movie}'")
    else:
        for _, row in results.iterrows():
            quadrant_color = QUADRANT_COLORS.get(row["quadrant"], "#ffffff")
            runtime = safe_int(row.get("runtime_minutes", 0), 0)
            title = html.escape(str(row["title"]))
            genre = html.escape(str(row["genre"]))
            quadrant = html.escape(str(row["quadrant"]))

            st.markdown(
                f"""
                <div class="search-result">
                    <h3 style="color: white; margin-bottom: 4px;">
                        #{safe_int(row['rank'])} {title} ({safe_int(row['year'])})
                    </h3>
                    <p style="color: #aaaaaa; margin-bottom: 16px;">
                        {genre} · {runtime} min
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
                                {quadrant}
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.divider()


# ============================================================
# VISUAL 1 — QUADRANT SCATTER PLOT
# ============================================================
st.markdown("## 📊 Sentiment vs Longevity — Quadrant Analysis")
st.markdown(
    "Each dot is a movie. Position shows its critic sentiment score vs long-term "
    "audience engagement. Color shows its licensing signal."
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
        "longevity_score": "Longevity Score (Votes/Year Normalized)",
        "quadrant": "Licensing Signal",
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
    legend=dict(bgcolor="#1a1a2e", bordercolor="#2a2a4a", borderwidth=1),
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
    "Average sentiment and longevity scores by genre. Shows where critic language "
    "is a reliable signal for streaming value."
)

genre_df = (
    filtered_df.groupby("primary_genre")
    .agg(
        avg_sentiment=("sentiment_compound", "mean"),
        avg_longevity=("longevity_score", "mean"),
        count=("title", "count"),
    )
    .reset_index()
)
genre_df = genre_df[genre_df["count"] >= 3].sort_values("avg_longevity", ascending=False)

if genre_df.empty:
    st.info("Not enough movies in the current filter to show genre analysis.")
else:
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
    "The most actionable licensing insights — movies that defied critic expectations "
    "in both directions."
)

col_t1, col_t2 = st.columns(2)

display_cols = ["title", "year", "rating", "sentiment_compound", "longevity_score"]
col_rename = {
    "title": "Title",
    "year": "Year",
    "rating": "Rating",
    "sentiment_compound": "Sentiment",
    "longevity_score": "Longevity",
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
    "Relationships between sentiment, longevity, rating, and votes. Shows which "
    "signals are truly independent predictors."
)

corr_cols = [
    "rating",
    "votes",
    "sentiment_compound",
    "sentiment_intensity",
    "longevity_score",
    "age_years",
]
corr_cols = [c for c in corr_cols if c in filtered_df.columns]

if len(corr_cols) < 2:
    st.info("Not enough numeric columns available to show a correlation matrix.")
else:
    corr_matrix = filtered_df[corr_cols].corr().round(2)
    label_map = {
        "rating": "IMDB Rating",
        "votes": "Total Votes",
        "sentiment_compound": "Sentiment",
        "sentiment_intensity": "Sent. Intensity",
        "longevity_score": "Longevity",
        "age_years": "Movie Age",
    }
    corr_matrix.index = [label_map.get(c, c) for c in corr_matrix.index]
    corr_matrix.columns = [label_map.get(c, c) for c in corr_matrix.columns]

    fig_heatmap = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns.tolist(),
            y=corr_matrix.index.tolist(),
            colorscale="RdBu",
            zmid=0,
            text=corr_matrix.values,
            texttemplate="%{text:.2f}",
            textfont={"size": 11, "color": "white"},
            hoverongaps=False,
        )
    )

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
    "Actionable insights for a streaming platform licensing team based on the "
    "sentiment-longevity analysis."
)

n_gems = len(df[df["quadrant"] == "Hidden Gem"])
n_hyped = len(df[df["quadrant"] == "Overhyped"])

if genre_df.empty:
    best_genre = "high-longevity genres"
else:
    best_genre = genre_df.loc[genre_df["avg_longevity"].idxmax(), "primary_genre"]

recommendations = [
    {
        "title": f"01 — Prioritize {best_genre} for Long-Term Licensing",
        "text": (
            f"{best_genre} films show strong longevity scores in this dataset, "
            "meaning audiences continue engaging with them long after release. "
            "These titles should be prioritized in catalogue acquisition strategies."
        ),
    },
    {
        "title": "02 — Do Not License Based on Sentiment Score Alone",
        "text": (
            f"{n_hyped} movies show high critic sentiment but below-median longevity. "
            "Positive buzz at release does not guarantee sustained audience engagement. "
            "Sentiment should be weighted alongside votes, genre benchmarks, and longevity."
        ),
    },
    {
        "title": "03 — Target Hidden Gems for Undervalued Catalogue Deals",
        "text": (
            f"{n_gems} movies show low sentiment scores but high longevity. These titles "
            "may be underpriced in licensing negotiations and can represent high-ROI "
            "catalogue opportunities."
        ),
    },
]

for rec in recommendations:
    st.markdown(
        f"""
        <div class="recommendation-card">
            <div class="recommendation-title">{html.escape(rec['title'])}</div>
            <div class="recommendation-text">{html.escape(rec['text'])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()


# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
<div style="text-align: center; color: #555555; padding: 20px 0;">
    Built by Pranav · BBA Computational Business Analytics · Mahindra University
    <br>
    <span style="color: #333333; font-size: 0.8rem;">
        This is Step 1. Something bigger is coming. 👀
    </span>
</div>
""",
    unsafe_allow_html=True,
)
