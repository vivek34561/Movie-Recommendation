import os
import streamlit as st
from tmdb_recommender import (
    get_api_key,
    get_session,
    search_movie,
    get_recommendations,
    recommend_top5,
    build_poster_url,
)

st.set_page_config(page_title="TMDB Movie Recommender", page_icon="🎬", layout="wide")

st.title("TMDB Movie Recommender")
st.write("Enter a movie title to get 5 recommendations. Uses TMDB API.")

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

if run and api_key and title.strip():
    try:
        session = get_session(api_key)
        inp = search_movie(session, title, api_key)
        if not inp:
            st.error("Movie not found. Try another title.")
        else:
            cands = get_recommendations(session, inp["id"], api_key)
            if not cands:
                st.info("No recommendations or similar titles found.")
            else:
                recs = recommend_top5(inp, cands)
                st.subheader(f"Top {len(recs)} recommendations for '{inp['title']}'")
                cols = st.columns(5)
                for i, r in enumerate(recs):
                    with cols[i % 5]:
                        poster = build_poster_url(r.get("poster_path"))
                        if poster:
                            st.image(poster, use_column_width=True)
                        st.markdown(f"**{r.get('title')}**")
                        year = (r.get("release_date") or "")[0:4]
                        st.caption(f"{year} • ⭐ {r.get('vote_average')}")
                        st.write((r.get("overview") or "")[:200])
                        st.link_button("TMDB", r.get("tmdb_url"), use_container_width=True)
    except Exception as e:
        st.error(f"Error: {e}")
