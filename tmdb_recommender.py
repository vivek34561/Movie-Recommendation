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
    # Add retry strategy to handle transient network resets/timeouts
    try:
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        retry = Retry(
            total=3,
            connect=3,
            read=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
    except Exception:
        # If urllib3 Retry isn't available, proceed without retries
        pass
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
            "popularity": det.get("popularity") or m.get("popularity"),
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
            "popularity": m.get("popularity"),
        })
    return top

def fetch_genres(session: requests.Session, api_key: str, language: str | None = None) -> list[dict]:
    params = {"language": language} if language else {}
    data = tmdb_get(session, "/genre/movie/list", params=params, api_key=api_key)
    return data.get("genres", [])

def hybrid_score(items: list[dict], alpha: float = 0.7, beta: float = 0.2, gamma: float = 0.1) -> list[dict]:
    # Normalize vote_average (0-10) and popularity (min-max) then combine
    votes = [float(i.get("vote_average") or 0.0) for i in items]
    pops = [float(i.get("popularity") or 0.0) for i in items]
    min_pop = min(pops) if pops else 0.0
    max_pop = max(pops) if pops else 1.0
    def norm_pop(x: float) -> float:
        if max_pop == min_pop:
            return 0.0
        return (x - min_pop) / (max_pop - min_pop)
    out = []
    for i, it in enumerate(items):
        sim = float(it.get("similarity") or 0.0)
        va = votes[i] / 10.0  # 0..1
        pn = norm_pop(pops[i])
        score = alpha * sim + beta * va + gamma * pn
        o = dict(it)
        o["hybrid_score"] = round(score, 4)
        out.append(o)
    # sort by hybrid score desc
    out.sort(key=lambda x: x.get("hybrid_score"), reverse=True)
    return out

def fetch_watch_providers(session: requests.Session, movie_id: int, api_key: str) -> dict:
    data = tmdb_get(session, f"/movie/{movie_id}/watch/providers", api_key=api_key)
    return data.get("results", {})

def fetch_videos(session: requests.Session, movie_id: int, api_key: str, language: str | None = None) -> list[dict]:
    params = {"language": language} if language else {}
    data = tmdb_get(session, f"/movie/{movie_id}/videos", params=params, api_key=api_key)
    return data.get("results", [])

def fetch_reviews(session: requests.Session, movie_id: int, api_key: str, language: str | None = None, page: int = 1) -> list[dict]:
    params = {"language": language, "page": page} if language else {"page": page}
    data = tmdb_get(session, f"/movie/{movie_id}/reviews", params=params, api_key=api_key)
    return data.get("results", [])

# Sentiment analysis using VADER (lightweight)
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()
except Exception:
    _vader = None

def analyze_sentiment(text: str) -> dict:
    if not text:
        return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}
    if _vader is None:
        lower = text.lower()
        pos_words = ["good", "great", "amazing", "excellent", "love", "best", "fantastic"]
        neg_words = ["bad", "terrible", "awful", "worst", "hate", "boring"]
        pos = sum(w in lower for w in pos_words)
        neg = sum(w in lower for w in neg_words)
        total = max(1, pos + neg)
        compound = (pos - neg) / total
        return {"neg": max(0.0, -compound), "neu": 1.0 - abs(compound), "pos": max(0.0, compound), "compound": compound}
    scores = _vader.polarity_scores(text)
    return scores

def summarize_review_sentiment(reviews: list[dict]) -> dict:
    import numpy as _np
    comps = []
    for r in reviews:
        txt = r.get("content") or ""
        s = analyze_sentiment(txt)
        comps.append(s.get("compound", 0.0))
        r["sentiment"] = s
    if not comps:
        return {"avg_compound": 0.0, "label": "Neutral"}
    avg = float(_np.mean(comps))
    label = "Positive" if avg >= 0.2 else ("Negative" if avg <= -0.2 else "Neutral")
    return {"avg_compound": round(avg, 4), "label": label}
