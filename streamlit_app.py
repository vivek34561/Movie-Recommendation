import os
import streamlit as st
from tmdb_recommender import (
    get_api_key,
    get_session,
    search_movie,
    get_recommendations,
    recommend_top5,
    build_poster_url,
    fetch_genres,
    hybrid_score,
    fetch_watch_providers,
)

st.set_page_config(page_title="TMDB Movie Recommender", page_icon="🎬", layout="wide")

st.title("TMDB Movie Recommender")
st.write("Enter a movie title to get 5 recommendations. Uses TMDB API.")

with st.sidebar:
    st.header("Filters")
    lang = st.selectbox("Language", ["en-US", "hi-IN", "es-ES", "fr-FR", "de-DE"], index=0)
    year_range = st.slider("Release year range", 1950, 2025, (2000, 2025))
    min_rating = st.slider("Minimum rating", 0.0, 10.0, 6.0, 0.5)
    alpha = st.slider("Similarity weight (α)", 0.0, 1.0, 0.7, 0.05)
    beta = st.slider("Rating weight (β)", 0.0, 1.0, 0.2, 0.05)
    gamma = st.slider("Popularity weight (γ)", 0.0, 1.0, 0.1, 0.05)
    region = st.selectbox("Region (providers)", ["US", "IN", "GB", "CA", "AU"], index=0)
    provider_filter = st.text_input("Provider filter (e.g., Netflix)", value="")

api_key = get_api_key()
if not api_key:
    with st.expander("Set TMDB API Key"):
        api_key = st.text_input("TMDB API Key (v3 key or v4 token)", type="password")
        if api_key:
            os.environ["TMDB_API_KEY"] = api_key

if not api_key:
    st.warning("Provide TMDB API key via environment variable TMDB_API_KEY or the input above.")

title = st.text_input("Movie title", value="Inception")
run = st.button("Recommend")

@st.cache_data(show_spinner=False)
def cached_search_and_genres(api_key: str, title: str, lang: str):
    session = get_session(api_key)
    inp = search_movie(session, title, api_key)
    gens = fetch_genres(session, api_key, language=lang)
    return inp, gens

@st.cache_data(show_spinner=False)
def cached_candidates(api_key: str, movie_id: int):
    session = get_session(api_key)
    return get_recommendations(session, movie_id, api_key)

@st.cache_data(show_spinner=False)
def cached_providers(api_key: str, movie_id: int):
    session = get_session(api_key)
    providers = fetch_watch_providers(session, movie_id, api_key)
    return providers

if run and api_key and title.strip():
    try:
        inp, gens = cached_search_and_genres(api_key, title, lang)
        if not inp:
            st.error("Movie not found. Try another title.")
        else:
            cands = cached_candidates(api_key, inp["id"]) if inp.get("id") else []
            # Apply filters
            def year_ok(d: str | None) -> bool:
                if not d:
                    return True
                try:
                    y = int(d[0:4])
                    return year_range[0] <= y <= year_range[1]
                except Exception:
                    return True
            filtered = [m for m in cands if year_ok(m.get("release_date")) and float(m.get("vote_average") or 0.0) >= float(min_rating)]
            # Rank by TF‑IDF similarity then hybrid score
            recs = recommend_top5(inp, filtered or cands)
            recs = hybrid_score(recs, alpha=alpha, beta=beta, gamma=gamma)
            st.subheader(f"Top {min(5, len(recs))} recommendations for '{inp['title']}'")
            cols = st.columns(5)
            for i, r in enumerate(recs[:5]):
                with cols[i % 5]:
                    poster = build_poster_url(r.get("poster_path"))
                    if poster:
                        st.image(poster, use_container_width=True)
                    st.markdown(f"**{r.get('title')}**")
                    year = (r.get("release_date") or "")[0:4]
                    st.caption(f"{year} • ⭐ {r.get('vote_average')} • score {r.get('hybrid_score')}")
                    st.write((r.get("overview") or "")[:200])
                    # Providers
                    providers = cached_providers(api_key, r.get("id"))
                    prov = providers.get(region) if isinstance(providers, dict) else None
                    names = []
                    if prov and prov.get("flatrate"):
                        names = [p.get("provider_name") for p in prov.get("flatrate") if isinstance(p, dict)]
                    if provider_filter:
                        if names and provider_filter.lower() not in ", ".join([n.lower() for n in names]):
                            continue
                    if names:
                        st.caption("Available on: " + ", ".join(names))
                    st.link_button("TMDB", r.get("tmdb_url"), use_container_width=True)
    except Exception as e:
        st.error(f"Error: {e}")
