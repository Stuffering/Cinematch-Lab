"""Streamlit portfolio dashboard for CineMatch Lab."""

# ruff: noqa: E501

from __future__ import annotations

import base64
import hashlib
import html
import urllib.error
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from cinematch import __version__
from cinematch.anomaly import build_user_anomaly_features, detect_user_anomalies
from cinematch.artifacts import load_model_artifact
from cinematch.clustering import (
    assign_user_segments,
    build_user_clustering_features,
    score_user_segments,
)
from cinematch.content import build_movie_feature_matrix
from scripts.recommend_strategy import (
    build_strategy_request_from_context,
    count_user_ratings,
    recommend_for_request,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
POSTER_SOURCE_URL = (
    "https://raw.githubusercontent.com/babu-thomas/movielens-posters/master/"
    "movie_poster.csv"
)
MOVIELENS_GENRE_COLUMNS = [
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]

PALETTE = {
    "navy": "#0B1020",
    "panel": "#111827",
    "panel_2": "#172033",
    "muted": "#94A3B8",
    "text": "#E5E7EB",
    "cyan": "#22D3EE",
    "blue": "#60A5FA",
    "violet": "#A78BFA",
    "pink": "#F472B6",
    "green": "#34D399",
    "amber": "#FBBF24",
    "red": "#FB7185",
}

MODEL_RESULTS = pd.DataFrame(
    [
        {
            "stage": "03",
            "model": "Bias baseline",
            "valid_rmse": 1.089829,
            "valid_mae": 0.858936,
            "test_rmse": 1.032749,
            "test_mae": 0.824633,
            "note": "Best simple baseline",
        },
        {
            "stage": "04",
            "model": "Item-CF",
            "valid_rmse": 1.177426,
            "valid_mae": 0.966152,
            "test_rmse": 1.110922,
            "test_mae": 0.931586,
            "note": "Behavior similarity",
        },
        {
            "stage": "05",
            "model": "Matrix factorization",
            "valid_rmse": 1.082666,
            "valid_mae": 0.857215,
            "test_rmse": 1.030065,
            "test_mae": 0.828554,
            "note": "Latent factors",
        },
        {
            "stage": "07",
            "model": "Supervised Ridge",
            "valid_rmse": 1.085629,
            "valid_mae": 0.862677,
            "test_rmse": 1.043925,
            "test_mae": 0.843649,
            "note": "Metadata + history",
        },
        {
            "stage": "10",
            "model": "Neural embedding",
            "valid_rmse": 1.020133,
            "valid_mae": 0.804521,
            "test_rmse": 1.059852,
            "test_mae": 0.847480,
            "note": "Known users/items subset",
        },
    ]
)

CLUSTER_RESULTS = pd.DataFrame(
    {
        "k": [3, 4, 5],
        "silhouette": [0.300, 0.307, 0.280],
    }
)

ANOMALY_COUNTS = pd.DataFrame(
    {
        "contamination": ["0.01", "0.05", "0.10"],
        "anomalies": [6, 30, 59],
        "users": [590, 590, 590],
    }
)

CAPABILITY_MATRIX = pd.DataFrame(
    [
        {
            "Capability": "Similar movie lookup",
            "Item-CF": "✅",
            "Content": "✅",
            "Hybrid": "✅",
            "Strategy": "✅",
        },
        {
            "Capability": "User top-N recommendation",
            "Item-CF": "✅",
            "Content": "✅",
            "Hybrid": "✅",
            "Strategy": "✅",
        },
        {
            "Capability": "Explicit content signal",
            "Item-CF": "—",
            "Content": "✅",
            "Hybrid": "✅",
            "Strategy": "✅",
        },
        {
            "Capability": "Weighted source blending",
            "Item-CF": "—",
            "Content": "—",
            "Hybrid": "✅",
            "Strategy": "✅",
        },
        {
            "Capability": "Automatic mode selection",
            "Item-CF": "—",
            "Content": "—",
            "Hybrid": "—",
            "Strategy": "✅",
        },
        {
            "Capability": "Artifact reuse path",
            "Item-CF": "—",
            "Content": "—",
            "Hybrid": "—",
            "Strategy": "✅",
        },
    ]
)


def apply_theme() -> None:
    """Apply portfolio-style dashboard CSS."""
    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                radial-gradient(circle at 12% 8%, rgba(96, 165, 250, 0.18), transparent 28%),
                radial-gradient(circle at 88% 2%, rgba(244, 114, 182, 0.14), transparent 30%),
                linear-gradient(135deg, #020617 0%, #0B1020 45%, #111827 100%);
            color: {PALETTE['text']};
        }}
        .block-container {{
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1280px;
        }}
        [data-testid="stSidebar"] {{
            background: rgba(2, 6, 23, 0.88);
            border-right: 1px solid rgba(148, 163, 184, 0.18);
        }}
        h1, h2, h3 {{ letter-spacing: -0.03em; }}
        .hero {{
            padding: 2.0rem;
            border: 1px solid rgba(148, 163, 184, 0.20);
            border-radius: 28px;
            background:
                linear-gradient(135deg, rgba(34, 211, 238, 0.18), rgba(167, 139, 250, 0.10)),
                rgba(15, 23, 42, 0.74);
            box-shadow: 0 24px 80px rgba(0, 0, 0, 0.32);
            backdrop-filter: blur(18px);
            margin-bottom: 1.3rem;
        }}
        .hero-title {{
            font-size: clamp(2.1rem, 5vw, 4.8rem);
            line-height: 0.95;
            font-weight: 850;
            margin-bottom: 0.8rem;
            background: linear-gradient(90deg, #F8FAFC, #22D3EE, #A78BFA);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .hero-subtitle {{
            color: #CBD5E1;
            font-size: 1.08rem;
            max-width: 920px;
            line-height: 1.75;
        }}
        .chip {{
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.45rem 0.75rem;
            margin: 0.25rem 0.35rem 0.25rem 0;
            border-radius: 999px;
            border: 1px solid rgba(148, 163, 184, 0.22);
            background: rgba(15, 23, 42, 0.75);
            color: #DDEAFE;
            font-size: 0.86rem;
        }}
        .metric-card {{
            padding: 1.1rem 1.1rem;
            border-radius: 22px;
            background: linear-gradient(180deg, rgba(30, 41, 59, 0.88), rgba(15, 23, 42, 0.88));
            border: 1px solid rgba(148, 163, 184, 0.18);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 20px 48px rgba(0,0,0,0.22);
            min-height: 132px;
        }}
        .metric-label {{
            color: {PALETTE['muted']};
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.55rem;
        }}
        .metric-value {{
            color: #F8FAFC;
            font-size: 2.05rem;
            font-weight: 800;
            line-height: 1.1;
        }}
        .metric-note {{
            color: #A7F3D0;
            font-size: 0.88rem;
            margin-top: 0.5rem;
        }}
        .section-card {{
            padding: 1.25rem;
            border-radius: 24px;
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.16);
            box-shadow: 0 18px 54px rgba(0, 0, 0, 0.22);
            margin-bottom: 1rem;
        }}
        .callout {{
            padding: 1rem 1.1rem;
            border-radius: 18px;
            border: 1px solid rgba(34, 211, 238, 0.24);
            background: rgba(8, 47, 73, 0.35);
            color: #DFFAFE;
        }}
        .pipeline-step {{
            border-left: 3px solid {PALETTE['cyan']};
            padding: 0.8rem 1rem;
            margin: 0.45rem 0;
            background: rgba(15, 23, 42, 0.64);
            border-radius: 14px;
        }}
        .pipeline-step strong {{ color: #F8FAFC; }}
        .pipeline-step span {{ color: #CBD5E1; }}
        .poster-card {{
            display: block;
            text-decoration: none !important;
            border-radius: 22px;
            overflow: hidden;
            border: 1px solid rgba(148, 163, 184, 0.18);
            background: rgba(15, 23, 42, 0.74);
            box-shadow: 0 18px 48px rgba(0,0,0,0.26);
            transition: transform 160ms ease, border-color 160ms ease;
            aspect-ratio: 2 / 3;
        }}
        .poster-card:hover {{
            transform: translateY(-3px);
            border-color: rgba(34, 211, 238, 0.55);
        }}
        .poster-frame {{
            display: block;
            width: 100%;
            height: 100%;
            background: rgba(15, 23, 42, 0.82);
        }}
        .poster-card img {{
            width: 100%;
            height: 100%;
            display: block;
            object-fit: cover;
        }}
        .poster-meta {{
            padding: 0.78rem 0.82rem 0.9rem 0.82rem;
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-top: 0;
            border-radius: 0 0 18px 18px;
            background: rgba(15, 23, 42, 0.70);
            margin: -0.15rem 0 0.85rem 0;
            min-height: 112px;
        }}
        .poster-title {{
            color: #F8FAFC;
            font-weight: 760;
            font-size: 0.95rem;
            line-height: 1.25;
            min-height: 2.4rem;
        }}
        .poster-subtitle {{
            color: #94A3B8;
            font-size: 0.78rem;
            margin-top: 0.35rem;
        }}
        .poster-score {{
            color: #A7F3D0;
            font-size: 0.82rem;
            margin-top: 0.45rem;
        }}
        .rated-chip {{
            display: inline-block;
            padding: 0.28rem 0.55rem;
            margin: 0.18rem 0.25rem 0.18rem 0;
            border-radius: 999px;
            background: rgba(34, 211, 238, 0.14);
            border: 1px solid rgba(34, 211, 238, 0.24);
            color: #DFFAFE;
            font-size: 0.78rem;
        }}
        div[data-testid="stMetric"] {{
            background: rgba(15, 23, 42, 0.62);
            border: 1px solid rgba(148, 163, 184, 0.14);
            padding: 0.9rem;
            border-radius: 18px;
        }}
        div[data-testid="stDataFrame"] {{
            border-radius: 18px;
            overflow: hidden;
            border: 1px solid rgba(148, 163, 184, 0.16);
        }}
        .stTabs [data-baseweb="tab-list"] {{ gap: 0.35rem; }}
        .stTabs [data-baseweb="tab"] {{
            background: rgba(15, 23, 42, 0.75);
            border: 1px solid rgba(148, 163, 184, 0.14);
            border-radius: 999px;
            color: #CBD5E1;
            padding: 0.55rem 1.0rem;
        }}
        .stTabs [aria-selected="true"] {{
            background: linear-gradient(90deg, rgba(34, 211, 238, 0.32), rgba(167, 139, 250, 0.28));
            color: #F8FAFC;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, note: str, accent: str = "green") -> None:
    """Render a custom metric card."""
    color = PALETTE.get(accent, PALETTE["green"])
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note" style="color:{color};">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    """Render the landing hero block."""
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">CineMatch Lab</div>
            <div class="hero-subtitle">
                A portfolio-ready machine learning recommendation system built on MovieLens 100K.
                The project moves from data validation and classical recommenders to supervised learning,
                neural embeddings, user intelligence, hybrid routing, and reusable model artifacts.
            </div>
            <div style="margin-top:1.1rem;">
                <span class="chip">🎬 MovieLens 100K</span>
                <span class="chip">📈 81 tests passed</span>
                <span class="chip">🧠 Supervised + Neural</span>
                <span class="chip">🧩 Hybrid Recommendation</span>
                <span class="chip">💾 Model Artifacts</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_poster_links() -> pd.DataFrame:
    """Load public MovieLens 100K poster links."""
    try:
        posters = pd.read_csv(
            POSTER_SOURCE_URL,
            names=["movie_id", "poster_url"],
            dtype={"movie_id": "int64", "poster_url": "string"},
        )
    except Exception:
        return pd.DataFrame(columns=["movie_id", "poster_url"])

    posters = posters.dropna(subset=["movie_id", "poster_url"])
    posters = posters.loc[posters["poster_url"].str.startswith("http", na=False)]

    return posters.drop_duplicates(subset=["movie_id"], keep="first")


@st.cache_data(show_spinner=False)
def load_processed_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load processed MovieLens data."""
    ratings_train = pd.read_csv(PROCESSED_DIR / "ratings_train.csv")
    ratings_valid = pd.read_csv(PROCESSED_DIR / "ratings_valid.csv")
    ratings_test = pd.read_csv(PROCESSED_DIR / "ratings_test.csv")
    movies = pd.read_csv(PROCESSED_DIR / "movies.csv")
    users = pd.read_csv(PROCESSED_DIR / "users.csv")
    poster_links = load_poster_links()
    if not poster_links.empty:
        movies = movies.merge(poster_links, on="movie_id", how="left")

    return ratings_train, ratings_valid, ratings_test, movies, users


@st.cache_data(show_spinner=False)
def build_segment_report() -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Build cached clustering segment profiles."""
    ratings_train, _, _, _, users = load_processed_data()
    features = build_user_clustering_features(ratings_train, users)
    numeric_features = features.select_dtypes(include="number")
    segments = assign_user_segments(numeric_features, n_clusters=4)
    silhouette = score_user_segments(numeric_features, segments)
    profile_table = segments.join(features, on="user_id")

    profiles = (
        profile_table.groupby("segment")
        .agg(
            user_count=("user_id", "size"),
            avg_rating_count=("rating_count", "mean"),
            avg_mean_rating=("mean_rating", "mean"),
            avg_age=("age", "mean"),
        )
        .reset_index()
        .sort_values("segment")
    )
    top_occupations = (
        profile_table.groupby("segment")["occupation"]
        .agg(lambda values: values.value_counts().idxmax())
        .rename("top_occupation")
        .reset_index()
    )
    profiles = profiles.merge(top_occupations, on="segment", how="left")

    return profiles, profile_table, silhouette


@st.cache_data(show_spinner=False)
def build_anomaly_report(contamination: float = 0.05) -> pd.DataFrame:
    """Build cached anomaly report."""
    ratings_train, _, _, _, _ = load_processed_data()
    features = build_user_anomaly_features(ratings_train)
    anomalies = detect_user_anomalies(features, contamination=contamination)

    return anomalies.join(features, on="user_id").sort_values(
        "anomaly_score",
        ascending=False,
    )


@st.cache_data(show_spinner=False)
def run_strategy_recommendation(
    user_id: int,
    requested_mode: str,
    n: int,
    candidate_n: int,
    item_cf_weight: float,
    content_weight: float,
    min_rating: float,
    min_common_users: int,
) -> tuple[pd.DataFrame, str, str, int]:
    """Run the recommendation strategy path and cache results."""
    ratings_train, _, _, movies, _ = load_processed_data()
    rating_count = count_user_ratings(ratings_train, user_id)
    request, reason = build_strategy_request_from_context(
        user_id=user_id,
        requested_mode=requested_mode,
        n=n,
        user_rating_count=rating_count,
    )

    if rating_count == 0:
        return pd.DataFrame(), request.mode, reason, rating_count

    recommendations = recommend_for_request(
        request=request,
        ratings=ratings_train,
        movies=movies,
        candidate_n=candidate_n,
        item_cf_weight=item_cf_weight,
        content_weight=content_weight,
        min_rating=min_rating,
        min_common_users=min_common_users,
    )

    return recommendations, request.mode, reason, rating_count


def movie_genres(movie: pd.Series, max_genres: int = 3) -> list[str]:
    """Return display genres for one movie row."""
    genres = []
    for genre in MOVIELENS_GENRE_COLUMNS:
        value = movie.get(genre, 0)
        if pd.notna(value) and int(value) == 1:
            genres.append(genre)

    return genres[:max_genres]


def movie_year(title: str) -> str:
    """Extract display year from a MovieLens title."""
    if "(" in title and title.endswith(")"):
        return title.rsplit("(", 1)[-1].rstrip(")")

    return "MovieLens"


def movie_url(movie: pd.Series) -> str:
    """Return a safe external movie URL."""
    url = str(movie.get("imdb_url", "") or "")
    if url.startswith("http"):
        return url

    return "https://www.imdb.com/"


def poster_placeholder_data_uri(movie: pd.Series) -> str:
    """Return a generated poster placeholder image source."""
    title = str(movie["title"])
    genres = movie_genres(movie, max_genres=2)
    digest = hashlib.md5(title.encode("utf-8")).hexdigest()
    color_a = f"#{digest[:6]}"
    color_b = f"#{digest[6:12]}"
    color_c = f"#{digest[12:18]}"
    safe_title = html.escape(title)
    safe_genres = html.escape(" • ".join(genres) if genres else "MovieLens")
    safe_year = html.escape(movie_year(title))
    svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" width="420" height="630" viewBox="0 0 420 630">
      <defs>
        <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="{color_a}"/>
          <stop offset="52%" stop-color="{color_b}"/>
          <stop offset="100%" stop-color="{color_c}"/>
        </linearGradient>
        <radialGradient id="glow" cx="25%" cy="18%" r="70%">
          <stop offset="0%" stop-color="#FFFFFF" stop-opacity="0.35"/>
          <stop offset="100%" stop-color="#FFFFFF" stop-opacity="0"/>
        </radialGradient>
      </defs>
      <rect width="420" height="630" fill="url(#bg)"/>
      <rect width="420" height="630" fill="url(#glow)"/>
      <rect x="26" y="28" width="368" height="574" rx="30" fill="#020617" opacity="0.34" stroke="#FFFFFF" stroke-opacity="0.30"/>
      <circle cx="332" cy="112" r="42" fill="#FFFFFF" opacity="0.16"/>
      <text x="42" y="88" fill="#E0F2FE" font-family="Arial, sans-serif" font-size="24" font-weight="700">{safe_year}</text>
      <foreignObject x="42" y="180" width="336" height="250">
        <div xmlns="http://www.w3.org/1999/xhtml" style="font-family: Arial, sans-serif; color: #F8FAFC; font-size: 42px; line-height: 1.05; font-weight: 850; letter-spacing: -1px;">
          {safe_title}
        </div>
      </foreignObject>
      <text x="42" y="534" fill="#CFFAFE" font-family="Arial, sans-serif" font-size="22" font-weight="700">{safe_genres}</text>
      <text x="42" y="572" fill="#E5E7EB" font-family="Arial, sans-serif" font-size="16">Click poster to open movie link</text>
    </svg>
    """
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")

    return f"data:image/svg+xml;base64,{encoded}"


@st.cache_data(show_spinner=False, ttl=86400)
def poster_url_is_available(url: str) -> bool:
    """Return whether a poster URL resolves to an image response."""
    request = urllib.request.Request(
        url,
        method="HEAD",
        headers={"User-Agent": "Mozilla/5.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=2.5) as response:
            content_type = response.headers.get("content-type", "")
            return response.status == 200 and content_type.startswith("image/")
    except (TimeoutError, urllib.error.URLError, urllib.error.HTTPError, ValueError):
        return False


def poster_image_src(movie: pd.Series) -> str:
    """Return a real poster URL when available, otherwise a generated fallback."""
    poster_value = movie.get("poster_url", "")
    poster_url = str(poster_value) if pd.notna(poster_value) else ""
    if poster_url.startswith("http") and poster_url_is_available(poster_url):
        return poster_url

    return poster_placeholder_data_uri(movie)


def render_movie_card(
    movie: pd.Series,
    subtitle: str = "",
    score: float | None = None,
) -> None:
    """Render a clickable poster-style movie card."""
    title = html.escape(str(movie["title"]))
    subtitle = html.escape(subtitle)
    poster_src = html.escape(poster_image_src(movie), quote=True)
    fallback_src = html.escape(poster_placeholder_data_uri(movie), quote=True)
    link_url = html.escape(movie_url(movie), quote=True)
    score_html = ""
    if score is not None:
        score_html = f'<div class="poster-score">score {score:.3f}</div>'
    st.markdown(
        f"""
        <a class="poster-card" href="{link_url}" target="_blank" rel="noopener noreferrer">
            <span class="poster-frame">
                <img src="{poster_src}" alt="{title}" onerror="this.onerror=null;this.src='{fallback_src}';" />
            </span>
        </a>
        <div class="poster-meta">
            <div class="poster-title">{title}</div>
            <div class="poster-subtitle">{subtitle}</div>
            {score_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def sample_taste_movies(
    movies: pd.DataFrame,
    ratings_train: pd.DataFrame,
    seed: int,
    batch_size: int,
) -> pd.DataFrame:
    """Sample recognizable movies for taste onboarding."""
    rating_counts = ratings_train.groupby("item_id").size().rename("rating_count")
    pool = movies.merge(
        rating_counts,
        left_on="movie_id",
        right_index=True,
        how="left",
    )
    pool["rating_count"] = pool["rating_count"].fillna(0)
    pool = pool.loc[pool["rating_count"] >= 30].copy()
    if pool.empty:
        pool = movies.copy()

    return pool.sample(
        n=min(batch_size, len(pool)),
        random_state=seed,
    ).sort_values("title")


def collect_taste_ratings(movies: pd.DataFrame) -> pd.DataFrame:
    """Collect current temporary user ratings from Streamlit session state."""
    records = []
    movie_lookup = movies.set_index("movie_id")
    for key, value in st.session_state.items():
        if not key.startswith("taste_rating_"):
            continue
        rating = int(value)
        if rating <= 0:
            continue
        movie_id = int(key.replace("taste_rating_", ""))
        if movie_id not in movie_lookup.index:
            continue
        records.append(
            {
                "movie_id": movie_id,
                "rating": rating,
                "title": movie_lookup.loc[movie_id, "title"],
            }
        )

    return pd.DataFrame(records).sort_values("title") if records else pd.DataFrame()


def clear_taste_ratings() -> None:
    """Clear temporary taste profile ratings."""
    for key in list(st.session_state.keys()):
        if key.startswith("taste_rating_"):
            del st.session_state[key]


def recommend_from_taste_profile(
    movies: pd.DataFrame,
    taste_ratings: pd.DataFrame,
    n: int = 8,
) -> tuple[pd.DataFrame, pd.Series]:
    """Recommend movies from a temporary content preference profile."""
    if taste_ratings.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    movie_features = build_movie_feature_matrix(movies)
    taste_ratings = taste_ratings.loc[
        taste_ratings["movie_id"].isin(movie_features.index)
    ].copy()
    if taste_ratings.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    rated_ids = taste_ratings["movie_id"].tolist()
    weights = taste_ratings.set_index("movie_id")["rating"].astype(float) - 3.0
    if weights.abs().sum() == 0:
        weights = taste_ratings.set_index("movie_id")["rating"].astype(float)
    if weights.abs().sum() == 0:
        weights = pd.Series(1.0, index=rated_ids)

    rated_features = movie_features.loc[rated_ids]
    profile = rated_features.mul(weights, axis=0).sum(axis=0) / weights.abs().sum()
    profile = profile.rename("preference_score")
    scores = movie_features.dot(profile).drop(index=rated_ids, errors="ignore")
    top_scores = scores.sort_values(ascending=False).head(n)
    recommendations = movies.set_index("movie_id").loc[top_scores.index].copy()
    recommendations["score"] = top_scores
    recommendations = recommendations.reset_index()

    return recommendations, profile.sort_values(ascending=False)


def make_bar_chart(
    data: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    xlabel: str = "",
    ylabel: str = "",
    hue: str | None = None,
) -> plt.Figure:
    """Create a themed bar chart."""
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    fig.patch.set_facecolor("#0B1020")
    ax.set_facecolor("#111827")
    if hue is None:
        sns.barplot(data=data, x=x, y=y, ax=ax, color=PALETTE["cyan"])
    else:
        sns.barplot(data=data, x=x, y=y, hue=hue, ax=ax, palette="viridis")
    ax.set_title(title, color="#F8FAFC", fontsize=15, weight="bold", pad=14)
    ax.set_xlabel(xlabel, color="#CBD5E1")
    ax.set_ylabel(ylabel, color="#CBD5E1")
    ax.tick_params(colors="#CBD5E1", axis="x", rotation=20)
    ax.tick_params(colors="#CBD5E1", axis="y")
    ax.grid(color="#334155", alpha=0.35)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    if ax.legend_ is not None:
        ax.legend_.set_title(None)
        ax.legend_.get_frame().set_facecolor("#111827")
        ax.legend_.get_frame().set_edgecolor("#334155")
        for text in ax.legend_.texts:
            text.set_color("#CBD5E1")
    fig.tight_layout()

    return fig


def make_rmse_chart() -> plt.Figure:
    """Render model RMSE comparison."""
    chart_data = MODEL_RESULTS.melt(
        id_vars=["model"],
        value_vars=["valid_rmse", "test_rmse"],
        var_name="split",
        value_name="rmse",
    )
    chart_data["split"] = chart_data["split"].str.replace("_rmse", "", regex=False)

    return make_bar_chart(
        data=chart_data,
        x="model",
        y="rmse",
        hue="split",
        title="Rating Prediction RMSE by Stage",
        ylabel="Lower is better",
    )


def make_mae_chart() -> plt.Figure:
    """Render model MAE comparison."""
    chart_data = MODEL_RESULTS.melt(
        id_vars=["model"],
        value_vars=["valid_mae", "test_mae"],
        var_name="split",
        value_name="mae",
    )
    chart_data["split"] = chart_data["split"].str.replace("_mae", "", regex=False)

    return make_bar_chart(
        data=chart_data,
        x="model",
        y="mae",
        hue="split",
        title="Rating Prediction MAE by Stage",
        ylabel="Lower is better",
    )


def make_silhouette_chart() -> plt.Figure:
    """Render clustering silhouette chart."""
    return make_bar_chart(
        data=CLUSTER_RESULTS,
        x="k",
        y="silhouette",
        title="User Clustering Quality by k",
        xlabel="Number of clusters",
        ylabel="Silhouette score",
    )


def make_anomaly_count_chart() -> plt.Figure:
    """Render anomaly count chart."""
    return make_bar_chart(
        data=ANOMALY_COUNTS,
        x="contamination",
        y="anomalies",
        title="Detected Anomalies by Contamination Setting",
        xlabel="Contamination",
        ylabel="Flagged users",
    )


def make_segment_profile_chart(profiles: pd.DataFrame) -> plt.Figure:
    """Render segment profile comparison."""
    chart_data = profiles.melt(
        id_vars=["segment"],
        value_vars=["avg_rating_count", "avg_mean_rating", "avg_age"],
        var_name="metric",
        value_name="value",
    )

    return make_bar_chart(
        data=chart_data,
        x="segment",
        y="value",
        hue="metric",
        title="User Segment Profiles",
        xlabel="Segment",
        ylabel="Average value",
    )


def make_anomaly_scatter(report: pd.DataFrame) -> plt.Figure:
    """Render anomaly scatter chart."""
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0B1020")
    ax.set_facecolor("#111827")
    sns.scatterplot(
        data=report,
        x="rating_count",
        y="mean_rating",
        hue="is_anomaly",
        size="anomaly_score",
        sizes=(28, 180),
        alpha=0.82,
        palette={False: "#60A5FA", True: "#FB7185"},
        ax=ax,
    )
    ax.set_xscale("log")
    ax.set_title("User Anomaly Map", color="#F8FAFC", fontsize=15, weight="bold", pad=14)
    ax.set_xlabel("Rating count, log scale", color="#CBD5E1")
    ax.set_ylabel("Mean rating", color="#CBD5E1")
    ax.tick_params(colors="#CBD5E1")
    ax.grid(color="#334155", alpha=0.35)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    if ax.legend_ is not None:
        ax.legend_.set_title(None)
        ax.legend_.get_frame().set_facecolor("#111827")
        ax.legend_.get_frame().set_edgecolor("#334155")
        for text in ax.legend_.texts:
            text.set_color("#CBD5E1")
    fig.tight_layout()

    return fig


def render_overview() -> None:
    """Render overview tab."""
    ratings_train, ratings_valid, ratings_test, movies, users = load_processed_data()

    cols = st.columns(4)
    with cols[0]:
        metric_card("Users", f"{users['user_id'].nunique():,}", "MovieLens profile table", "cyan")
    with cols[1]:
        metric_card("Movies", f"{movies['movie_id'].nunique():,}", "Genre-rich catalog", "violet")
    with cols[2]:
        total_ratings = len(ratings_train) + len(ratings_valid) + len(ratings_test)
        metric_card("Ratings", f"{total_ratings:,}", "Train / valid / test splits", "green")
    with cols[3]:
        metric_card("Tests", "81", "Full validation passed", "amber")

    left, right = st.columns([1.0, 1.2])
    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Machine Learning Pipeline")
        steps = [
            ("01-02", "Data ingestion, cleaning, validation, and EDA"),
            ("03-05", "Baselines, item-CF, and matrix factorization"),
            ("06-07", "Content features and supervised Ridge prediction"),
            ("08-09", "User clustering and anomaly detection"),
            ("10", "Neural embedding rating model"),
            ("11-12", "Hybrid recommendation and strategy routing"),
            ("13-14", "Reusable artifacts and final validation"),
        ]
        for stage, text in steps:
            st.markdown(
                f'<div class="pipeline-step"><strong>Stage {stage}</strong><br><span>{text}</span></div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Recommendation Capability Matrix")
        st.dataframe(CAPABILITY_MATRIX, hide_index=True, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="callout">
        <strong>Portfolio framing:</strong> this project is not only a model zoo.
        It shows a complete ML recommendation workflow: data preparation,
        candidate generation, prediction, user intelligence, routing, validation,
        and model artifact reuse.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_progress() -> None:
    """Render model performance visualizations."""
    cols = st.columns(3)
    with cols[0]:
        best_valid = MODEL_RESULTS.loc[MODEL_RESULTS["valid_rmse"].idxmin()]
        metric_card("Best Valid RMSE", f"{best_valid['valid_rmse']:.3f}", best_valid["model"], "green")
    with cols[1]:
        best_test = MODEL_RESULTS.loc[MODEL_RESULTS["test_rmse"].idxmin()]
        metric_card("Best Test RMSE", f"{best_test['test_rmse']:.3f}", best_test["model"], "cyan")
    with cols[2]:
        metric_card("Neural Scope", "Known subset", "Users/items seen in training", "amber")

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.pyplot(make_rmse_chart(), use_container_width=True)
    with chart_right:
        st.pyplot(make_mae_chart(), use_container_width=True)

    st.subheader("Model Results Table")
    st.dataframe(
        MODEL_RESULTS.round(6),
        hide_index=True,
        use_container_width=True,
    )
    st.info(
        "Neural embedding results are shown on the known-user/item subset. "
        "They are useful for learning but should not be read as full-split metrics."
    )


def render_recommendation_lab() -> None:
    """Render interactive recommendation tab."""
    ratings_train, _, _, _, _ = load_processed_data()
    user_ids = sorted(ratings_train["user_id"].unique().tolist())

    st.markdown(
        """
        <div class="callout">
        Try the recommendation strategy layer. In <strong>auto</strong> mode,
        users with history route to hybrid recommendation; cold-start users route
        to content mode, though the current content profile still requires at least
        one rating to build preferences.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("recommendation_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            user_id = st.selectbox("User ID", user_ids, index=user_ids.index(1) if 1 in user_ids else 0)
            requested_mode = st.selectbox("Recommendation mode", ["auto", "hybrid", "content", "item_cf"])
        with c2:
            top_n = st.slider("Top-N", min_value=5, max_value=20, value=10, step=1)
            candidate_n = st.slider("Hybrid candidates per source", 10, 50, 25, step=5)
        with c3:
            item_cf_weight = st.slider("Item-CF weight", 0.0, 1.0, 0.7, step=0.05)
            content_weight = st.slider("Content weight", 0.0, 1.0, 0.3, step=0.05)
        run_button = st.form_submit_button("Run recommendation", use_container_width=True)

    if run_button:
        with st.spinner("Generating recommendations..."):
            recommendations, selected_mode, reason, rating_count = run_strategy_recommendation(
                user_id=int(user_id),
                requested_mode=requested_mode,
                n=int(top_n),
                candidate_n=int(candidate_n),
                item_cf_weight=float(item_cf_weight),
                content_weight=float(content_weight),
                min_rating=4.0,
                min_common_users=20,
            )
        cols = st.columns(4)
        with cols[0]:
            metric_card("Selected Mode", selected_mode, requested_mode, "cyan")
        with cols[1]:
            metric_card("Training Ratings", f"{rating_count:,}", "User history size", "green")
        with cols[2]:
            metric_card("Top-N", str(top_n), "Returned recommendations", "violet")
        with cols[3]:
            metric_card("Candidate Pool", str(candidate_n), "Per hybrid source", "amber")
        st.caption(reason)

        if recommendations.empty:
            st.warning("No recommendations found with the current settings.")
        else:
            score_column = "hybrid_score" if selected_mode == "hybrid" else "score"
            display = recommendations.copy()
            display[score_column] = display[score_column].round(3)
            st.dataframe(display, hide_index=True, use_container_width=True)
    else:
        st.markdown(
            """
            <div class="section-card">
            <h3>Ready to demo</h3>
            <p style="color:#CBD5E1;">
            Choose a user and click <strong>Run recommendation</strong>. The app will
            show which strategy was selected, why it was selected, and the resulting
            movie recommendation table.
            </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_taste_studio() -> None:
    """Render a visual content-profile recommendation studio."""
    ratings_train, _, _, movies, _ = load_processed_data()

    st.markdown(
        """
        <div class="callout">
        Build a temporary taste profile by rating a few popular movies. The page
        turns those ratings into a genre preference vector and recommends unseen
        titles with similar content signals.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "taste_seed" not in st.session_state:
        st.session_state["taste_seed"] = 42

    top_controls = st.columns([1, 1, 2])
    with top_controls[0]:
        batch_size = st.slider("Movies to rate", min_value=6, max_value=12, value=9, step=3)
    with top_controls[1]:
        recommendation_count = st.slider("Recommendations", min_value=4, max_value=12, value=8, step=2)
    with top_controls[2]:
        refresh, clear = st.columns(2)
        with refresh:
            if st.button("Refresh sample", use_container_width=True):
                st.session_state["taste_seed"] += 1
        with clear:
            if st.button("Clear ratings", use_container_width=True):
                clear_taste_ratings()

    sampled_movies = sample_taste_movies(
        movies=movies,
        ratings_train=ratings_train,
        seed=int(st.session_state["taste_seed"]),
        batch_size=int(batch_size),
    )

    st.subheader("Rate A Few Movies")
    movie_columns = st.columns(3)
    for index, (_, movie) in enumerate(sampled_movies.iterrows()):
        with movie_columns[index % 3]:
            genres = ", ".join(movie_genres(movie)) or movie_year(str(movie["title"]))
            render_movie_card(movie, subtitle=genres)
            st.select_slider(
                "Rating",
                options=[0, 1, 2, 3, 4, 5],
                value=st.session_state.get(f"taste_rating_{int(movie['movie_id'])}", 0),
                format_func=lambda value: "Skip" if value == 0 else f"{value} / 5",
                key=f"taste_rating_{int(movie['movie_id'])}",
                label_visibility="collapsed",
            )

    taste_ratings = collect_taste_ratings(movies)
    recommendations, profile = recommend_from_taste_profile(
        movies=movies,
        taste_ratings=taste_ratings,
        n=int(recommendation_count),
    )

    st.divider()
    summary_cols = st.columns(3)
    with summary_cols[0]:
        metric_card("Rated Movies", f"{len(taste_ratings)}", "Temporary session profile", "cyan")
    with summary_cols[1]:
        strongest_genre = profile.index[0] if not profile.empty else "Waiting"
        metric_card("Top Signal", str(strongest_genre).title(), "Highest profile weight", "green")
    with summary_cols[2]:
        metric_card("Candidates", f"{len(movies):,}", "MovieLens catalog", "violet")

    if taste_ratings.empty:
        st.info("Rate at least one movie above to generate visual recommendations.")
        return

    left, right = st.columns([1, 1.3])
    with left:
        st.subheader("Your Temporary Ratings")
        st.dataframe(taste_ratings, hide_index=True, use_container_width=True)
        if not profile.empty:
            profile_chart = (
                profile.loc[profile.abs() > 0]
                .sort_values(ascending=False)
                .head(8)
                .rename_axis("genre")
                .reset_index(name="preference_score")
            )
            if not profile_chart.empty:
                st.pyplot(
                    make_bar_chart(
                        profile_chart,
                        x="genre",
                        y="preference_score",
                        title="Strongest Content Signals",
                        xlabel="Genre",
                        ylabel="Preference score",
                    ),
                    use_container_width=True,
                )

    with right:
        st.subheader("Recommended For This Taste")
        if recommendations.empty:
            st.warning("No recommendations found from the current taste profile.")
            return

        rec_columns = st.columns(4)
        for index, (_, movie) in enumerate(recommendations.iterrows()):
            with rec_columns[index % 4]:
                genres = ", ".join(movie_genres(movie)) or movie_year(str(movie["title"]))
                render_movie_card(
                    movie,
                    subtitle=genres,
                    score=float(movie["score"]),
                )


def render_user_insights() -> None:
    """Render clustering and anomaly insights."""
    profiles, _, silhouette = build_segment_report()
    anomaly_report = build_anomaly_report(contamination=0.05)
    anomaly_count = int(anomaly_report["is_anomaly"].sum())

    cols = st.columns(4)
    with cols[0]:
        metric_card("Selected k", "4", "Best checked silhouette", "cyan")
    with cols[1]:
        metric_card("Silhouette", f"{silhouette:.3f}", "Segment separation", "green")
    with cols[2]:
        metric_card("Anomalies", f"{anomaly_count}/590", "contamination=0.05", "red")
    with cols[3]:
        top_user = int(anomaly_report.iloc[0]["user_id"])
        metric_card("Top anomaly", f"User {top_user}", "Highest anomaly score", "amber")

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.pyplot(make_silhouette_chart(), use_container_width=True)
    with chart_right:
        st.pyplot(make_anomaly_count_chart(), use_container_width=True)

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.pyplot(make_segment_profile_chart(profiles), use_container_width=True)
        st.dataframe(profiles.round(3), hide_index=True, use_container_width=True)
    with chart_right:
        st.pyplot(make_anomaly_scatter(anomaly_report), use_container_width=True)
        st.dataframe(
            anomaly_report.head(10).round(3),
            hide_index=True,
            use_container_width=True,
        )


def render_engineering() -> None:
    """Render engineering validation and artifacts tab."""
    artifact_path = MODELS_DIR / "supervised_ridge_sample.joblib"
    artifact_exists = artifact_path.exists()

    cols = st.columns(4)
    with cols[0]:
        metric_card("Tests", "81 passed", "pytest full suite", "green")
    with cols[1]:
        metric_card("Lint", "Passed", "ruff check .", "cyan")
    with cols[2]:
        metric_card("Submission", "Clean", "No learning markers", "violet")
    with cols[3]:
        metric_card("Artifact", "Found" if artifact_exists else "Missing", "models/*.joblib", "amber")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Validation Commands")
        st.code(
            """python -m pytest
python -m ruff check .
python scripts/check_submission.py""",
            language="bash",
        )
        st.subheader("Artifact Workflow")
        st.code(
            """python scripts/train_supervised.py --eval-split valid --alpha 1.0 --model-output models/supervised_ridge.joblib
python scripts/evaluate_supervised_artifact.py --artifact-path models/supervised_ridge.joblib --eval-split valid""",
            language="bash",
        )
    with right:
        st.subheader("Saved Artifact Metadata")
        if artifact_exists:
            artifact = load_model_artifact(artifact_path)
            metadata = artifact.get("metadata", {})
            st.json(metadata)
            st.caption(
                f"Feature columns saved: {len(artifact.get('feature_columns', []))}; "
                f"target: {artifact.get('target_column', 'unknown')}"
            )
        else:
            st.warning(
                "No sample artifact found. Run the supervised training command with "
                "--model-output to create one."
            )


def main() -> None:
    """Run the Streamlit dashboard."""
    st.set_page_config(
        page_title="CineMatch Lab Portfolio",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_theme()

    st.sidebar.title("🎬 CineMatch Lab")
    st.sidebar.caption(f"Version {__version__}")
    st.sidebar.markdown(
        """
        A machine learning portfolio dashboard for recommendation systems,
        user behavior analysis, and reusable model workflows.
        """
    )
    st.sidebar.divider()
    st.sidebar.success("Local validation: 81 tests passed")
    st.sidebar.info("Use Recommendation Lab or Taste Studio for interactive demos.")

    render_hero()

    tabs = st.tabs(
        [
            "Overview",
            "Model Progress",
            "Recommendation Lab",
            "Taste Studio",
            "User Insights",
            "Engineering",
        ]
    )

    with tabs[0]:
        render_overview()
    with tabs[1]:
        render_model_progress()
    with tabs[2]:
        render_recommendation_lab()
    with tabs[3]:
        render_taste_studio()
    with tabs[4]:
        render_user_insights()
    with tabs[5]:
        render_engineering()


if __name__ == "__main__":
    main()
