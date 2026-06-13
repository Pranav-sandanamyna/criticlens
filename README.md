# 🎬 CriticLens
### Can critic sentiment predict a movie's long-term streaming value?

**Live App:** [criticlens.streamlit.app](https://pranav-sandanamyna-criticlens-apphome-bdtydz.streamlit.app/)

---

## The Business Question

Streaming platforms spend billions licensing content. The conventional wisdom is simple — 
high ratings mean long-term value. But is that actually true?

This project challenges that assumption by asking a different question:

> *Does the emotional language critics use at release predict whether a movie becomes 
> a long-term streaming asset — independent of its initial rating?*

---

## What I Found

The results were counterintuitive.

**Sentiment score vs longevity correlation: -0.01**

Near zero. Critic sentiment at release has almost no relationship with how long 
a movie sustains audience engagement. This directly contradicts the intuition that 
positive buzz predicts lasting value.

**The strongest predictor of longevity was total votes (r = 0.32) — not sentiment, 
not rating.**

### Most Overhyped — High Sentiment, Low Longevity
| Movie | Year | Sentiment | Longevity |
|---|---|---|---|
| Hachi: A Dog's Tale | 2009 | 0.986 | 2.56 |
| Hotel Rwanda | 2004 | 0.985 | 2.19 |
| Before Sunset | 2004 | 0.952 | 1.71 |

### Hidden Gems — Low Sentiment, High Longevity
| Movie | Year | Sentiment | Longevity |
|---|---|---|---|
| Dune: Part Two | 2024 | -0.660 | 100.0 |
| Joker | 2019 | -0.557 | 37.09 |
| The Dark Knight | 2008 | -0.361 | 24.38 |

**The Dark Knight** — rated 9.1 on IMDB, consistently described with dark and 
intense language by critics — is one of the most-watched movies of all time. 
Sentiment didn't predict that. The audience did.

---

## Business Recommendations

**1. Don't license based on sentiment score alone**
Positive critic buzz at release has near-zero correlation with sustained audience 
engagement. Sentiment is a weak standalone signal for licensing decisions.

**2. Hidden Gems are undervalued licensing opportunities**
Movies with negative sentiment but high longevity — like The Dark Knight and Joker — 
are likely underpriced in licensing negotiations despite sustained audience demand.

**3. Vote trajectory is a stronger signal than sentiment**
Total votes showed the strongest correlation with longevity (r = 0.32). 
Platforms should weight vote momentum over critic language when evaluating catalogue value.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Data Source | IMDB Official Datasets (title.basics, title.ratings) |
| Sentiment Analysis | VADER (Valence Aware Dictionary and sEntiment Reasoner) |
| Data Processing | Python · Pandas · NumPy · SciPy |
| Visualization | Plotly · Matplotlib · Seaborn |
| Dashboard | Streamlit |
| Version Control | GitHub |

---

## Project Structure