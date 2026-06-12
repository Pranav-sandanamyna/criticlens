import streamlit as st
import pandas as pd
from pathlib import Path

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="CriticLens — All Movies",
    page_icon="🎬",
    layout="wide",
)

st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .block-container { padding-top: 2rem; }
    h1, h2, h3 { color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATA
# ============================================================
@st.cache_data
def load_data():
    possible_paths = [
    Path("/mount/src/criticlens/data/processed/imdb_final.csv"),
    Path("data/processed/imdb_final.csv"),
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
    st.error("Data file not found.")
    st.stop()

df = load_data()

QUADRANT_COLORS = {
    "Safe Bet":   "#00d4aa",
    "Hidden Gem": "#4da6ff",
    "Overhyped":  "#ff6b6b",
    "Avoid":      "#888888",
}


# ============================================================
# HEADER
# ============================================================
st.markdown("# 🎬 All 250 Movies")
st.markdown("Browse, filter, and sort the complete IMDB Top 250 dataset with sentiment and longevity scores.")
st.divider()


# ============================================================
# FILTERS ROW
# ============================================================
col1, col2, col3, col4 = st.columns([2, 2, 2, 2])

with col1:
    genre_options = ["All"] + sorted(df["primary_genre"].dropna().unique().tolist())
    selected_genre = st.selectbox("Genre", genre_options)

with col2:
    quadrant_options = ["All", "Safe Bet", "Hidden Gem", "Overhyped", "Avoid"]
    selected_quadrant = st.selectbox("Licensing Signal", quadrant_options)

with col3:
    sort_options = {
        "Rank":            "rank",
        "Longevity Score": "longevity_score",
        "Sentiment Score": "sentiment_compound",
        "IMDB Rating":     "rating",
        "Year (Newest)":   "year",
        "Votes":           "votes",
    }
    selected_sort_label = st.selectbox("Sort By", list(sort_options.keys()))
    selected_sort = sort_options[selected_sort_label]

with col4:
    sort_order = st.selectbox("Order", ["Descending", "Ascending"])
    ascending = sort_order == "Ascending"


# ============================================================
# APPLY FILTERS
# ============================================================
filtered = df.copy()

if selected_genre != "All":
    filtered = filtered[filtered["primary_genre"] == selected_genre]

if selected_quadrant != "All":
    filtered = filtered[filtered["quadrant"] == selected_quadrant]

filtered = filtered.sort_values(selected_sort, ascending=ascending)

st.markdown(f"**Showing {len(filtered)} movies**")
st.divider()


# ============================================================
# MOVIE CARDS
# ============================================================
for _, row in filtered.iterrows():
    quadrant_color = QUADRANT_COLORS.get(row["quadrant"], "#ffffff")
    sent_pct = int(((row["sentiment_compound"] + 1) / 2) * 100)
    long_pct  = int(row["longevity_score"])

    col_left, col_mid, col_right = st.columns([3, 4, 2])

    with col_left:
        st.markdown(f"""
        <div style="padding: 12px 0;">
            <div style="color: #aaaaaa; font-size: 0.75rem;">#{int(row['rank'])}</div>
            <div style="color: #ffffff; font-size: 1rem; font-weight: 700;">{row['title']}</div>
            <div style="color: #888888; font-size: 0.8rem;">{int(row['year'])} · {row['primary_genre']}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_mid:
        st.markdown(f"""
        <div style="padding: 12px 0;">
            <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #aaaaaa; margin-bottom: 3px;">
                <span>Sentiment</span><span>{row['sentiment_compound']:.3f}</span>
            </div>
            <div style="background: #2a2a4a; border-radius: 4px; height: 6px; margin-bottom: 10px;">
                <div style="background: #4da6ff; width: {sent_pct}%; height: 100%; border-radius: 4px;"></div>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #aaaaaa; margin-bottom: 3px;">
                <span>Longevity</span><span>{row['longevity_score']:.1f}/100</span>
            </div>
            <div style="background: #2a2a4a; border-radius: 4px; height: 6px;">
                <div style="background: #00d4aa; width: {long_pct}%; height: 100%; border-radius: 4px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown(f"""
        <div style="padding: 12px 0; text-align: right;">
            <div style="color: #00d4aa; font-size: 1.3rem; font-weight: 700;">⭐ {row['rating']}</div>
            <div style="
                display: inline-block;
                background: {quadrant_color}22;
                border: 1px solid {quadrant_color};
                color: {quadrant_color};
                border-radius: 20px;
                padding: 3px 10px;
                font-size: 0.75rem;
                font-weight: 600;
                margin-top: 6px;
            ">{row['quadrant']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color: #2a2a4a; margin: 0;'>", unsafe_allow_html=True)