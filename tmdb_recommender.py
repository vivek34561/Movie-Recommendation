import os
import typing as t
import requests

BASE_URL = "https://api.themoviedb.org/3"
IMAGE_BASE_URL = "https://image.tmdb.org/t/p"
POSTER_SIZE = "w342"


def get_api_key() -> t.Optional[str]:
    key = os.environ.get("TMDB_API_KEY")
    return key


def get_session(api_key: str) -> requests.Session:
    session = requests.Session()
    session.headers.update({"Accept": "application/json"})
    # v4 token (Bearer) starts with eyJ typically
    if api_key and api_key.startswith("eyJ"):
        session.headers.update({"Authorization": f"Bearer {api_key}"})
    return session


def tmdb_get(session: requests.Session, path: str, params: dict | None = None, api_key: str | None = None) -> dict:
    url = f"{BASE_URL}{path}"
    params = params.copy() if params else {}
    # If not using Bearer, pass api_key query
    if not session.headers.get("Authorization") and api_key:
        params["api_key"] = api_key
    resp = session.get(url, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()


def search_movie(session: requests.Session, title: str, api_key: str) -> t.Optional[dict]:
    if not title or not title.strip():
        return None
    data = tmdb_get(session, "/search/movie", params={"query": title.strip()}, api_key=api_key)
    results = data.get("results", [])
    if not results:
        return None
    best = results[0]
    mid = best.get("id")
    details = tmdb_get(session, f"/movie/{mid}", api_key=api_key) if mid else {}
    genres = details.get("genres", []) if isinstance(details, dict) else []
    return {
        "id": mid,
        "title": best.get("title"),
        "overview": details.get("overview") or best.get("overview") or "",
        "genres": [g.get("name") for g in genres if isinstance(g, dict)],
        "release_date": best.get("release_date") or details.get("release_date"),
        "vote_average": details.get("vote_average") or best.get("vote_average"),
        "poster_path": best.get("poster_path") or details.get("poster_path"),
    }


def get_recommendations(session: requests.Session, movie_id: int, api_key: str, page: int = 1) -> list[dict]:
    rec = tmdb_get(session, f"/movie/{movie_id}/recommendations", params={"page": page}, api_key=api_key)
    recs = rec.get("results", [])
    sim = tmdb_get(session, f"/movie/{movie_id}/similar", params={"page": page}, api_key=api_key)
    sims = sim.get("results", [])
    by_id: dict[int, dict] = {}
    for m in recs + sims:
        if isinstance(m, dict) and m.get("id"):
            by_id[m["id"]] = m
    enriched: list[dict] = []
    for m in by_id.values():
        mid = m.get("id")
        det = tmdb_get(session, f"/movie/{mid}", api_key=api_key) if mid else {}
        genres = det.get("genres", []) if isinstance(det, dict) else []
        enriched.append({
            "id": mid,
            "title": m.get("title") or det.get("title"),
            "overview": det.get("overview") or m.get("overview") or "",
            "genres": [g.get("name") for g in genres if isinstance(g, dict)],
            "release_date": m.get("release_date") or det.get("release_date"),
            "vote_average": det.get("vote_average") or m.get("vote_average"),
            "poster_path": m.get("poster_path") or det.get("poster_path"),
        })
    return enriched


def build_poster_url(path: str | None, size: str = POSTER_SIZE) -> t.Optional[str]:
    if not path:
        return None
    return f"{IMAGE_BASE_URL}/{size}{path}"


# TF-IDF based recommender
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def _doc_from(m: dict) -> str:
    genres = " ".join(m.get("genres", []) or [])
    overview = (m.get("overview") or "").strip()
    return f"{overview} {genres}".strip()


def recommend_top5(input_movie: dict, candidates: list[dict]) -> list[dict]:
    docs = [_doc_from(input_movie)] + [_doc_from(m) for m in candidates]
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    X = vec.fit_transform(docs)
    sims = (X[1:] @ X[0].T).toarray().ravel()
    order = np.argsort(-sims)
    top_idx = order[:5]
    top: list[dict] = []
    for i in top_idx:
        m = candidates[i]
        score = float(sims[i])
        top.append({
            "id": m.get("id"),
            "title": m.get("title"),
            "similarity": round(score, 4),
            "tmdb_url": f"https://www.themoviedb.org/movie/{m.get('id')}",
            "release_date": m.get("release_date"),
            "vote_average": m.get("vote_average"),
            "poster_path": m.get("poster_path"),
            "genres": m.get("genres"),
            "overview": m.get("overview"),
        })
    return top
