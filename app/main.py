"""FastAPI backend: auth, catalog browse/search, interaction tracking,
and the recommendation service (application layer of the architecture).
"""
import os
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import db
from .recommender import RecommenderEngine

BASE = os.path.dirname(__file__)
PROCESSED = os.environ.get(
    "DATA_DIR", os.path.join(BASE, "..", "data", "processed"))
STATIC = os.path.join(BASE, "static")

app = FastAPI(title="Next-Generation E-Commerce Recommender")
engine = RecommenderEngine()
catalog = pd.DataFrame()
train_history = {}


@app.on_event("startup")
def startup():
    global catalog, train_history
    db.init()
    catalog = pd.read_csv(os.path.join(PROCESSED, "products.csv")).fillna("")
    train_path = os.path.join(PROCESSED, "train.csv")
    if os.path.exists(train_path):
        t = pd.read_csv(train_path)
        train_history = {u: dict(zip(g["asin"], g["rating"]))
                         for u, g in t.groupby("user")}
    engine.load()


# ---------------- auth ----------------
class Credentials(BaseModel):
    username: str
    password: str


def current_user(req: Request):
    user = db.user_for_token(req.cookies.get("session", ""))
    if not user:
        raise HTTPException(401, "not signed in")
    return user


@app.post("/api/signup")
def signup(creds: Credentials, res: Response):
    if not creds.username or len(creds.password) < 4:
        raise HTTPException(400, "username required; password >= 4 chars")
    if not db.create_user(creds.username, creds.password):
        raise HTTPException(409, "username already taken")
    res.set_cookie("session", db.create_session(creds.username),
                   httponly=True, samesite="lax")
    return {"ok": True, "username": creds.username}


@app.post("/api/login")
def login(creds: Credentials, res: Response):
    if not db.verify_user(creds.username, creds.password):
        raise HTTPException(401, "invalid credentials")
    res.set_cookie("session", db.create_session(creds.username),
                   httponly=True, samesite="lax")
    return {"ok": True, "username": creds.username}


@app.post("/api/logout")
def logout(req: Request, res: Response):
    db.drop_session(req.cookies.get("session", ""))
    res.delete_cookie("session")
    return {"ok": True}


@app.get("/api/me")
def me(req: Request):
    return {"username": current_user(req)}


@app.post("/api/demo")
def demo_login(res: Response):
    """Sign in as a random real reviewer from the dataset - demonstrates
    personalized recommendations instantly (cold-start users still work
    via the live-interaction layer)."""
    if not train_history:
        raise HTTPException(503, "dataset not loaded")
    rng = np.random.default_rng()
    user = list(train_history)[int(rng.integers(len(train_history)))]
    res.set_cookie("session", db.create_session(user),
                   httponly=True, samesite="lax")
    return {"ok": True, "username": user,
            "history_size": len(train_history[user])}


# ---------------- catalog ----------------
@app.get("/api/categories")
def categories():
    return {"categories": sorted(
        c for c in catalog["category"].unique() if c)[:60]}


@app.get("/api/products")
def products(req: Request, q: str = "", category: str = "",
             page: int = 1, size: int = 24):
    current_user(req)
    df = catalog
    if q:
        ql = q.lower()
        df = df[df["title"].str.lower().str.contains(ql, regex=False) |
                df["brand"].str.lower().str.contains(ql, regex=False)]
    if category:
        df = df[df["category"] == category]
    total = len(df)
    start = (page - 1) * size
    rows = df.iloc[start:start + size]
    return {"total": total, "page": page, "products": rows.to_dict("records")}


@app.get("/api/products/{asin}")
def product_detail(req: Request, asin: str):
    user = current_user(req)
    row = catalog[catalog["asin"] == asin]
    if row.empty:
        raise HTTPException(404, "product not found")
    db.log_interaction(user, asin, "view")
    rec = row.iloc[0].to_dict()
    rec["similar"] = engine.similar_products(asin)
    rec["similar_products"] = [
        catalog[catalog["asin"] == s["asin"]].iloc[0].to_dict()
        for s in rec["similar"]
        if not catalog[catalog["asin"] == s["asin"]].empty]
    return rec


# ---------------- interactions ----------------
class Interaction(BaseModel):
    asin: str
    event: str            # view | click | rate | purchase
    rating: Optional[float] = None


@app.post("/api/interactions")
def track(req: Request, body: Interaction):
    user = current_user(req)
    if body.event not in ("view", "click", "rate", "purchase"):
        raise HTTPException(400, "unknown event")
    db.log_interaction(user, body.asin, body.event, body.rating)
    return {"ok": True}


# ---------------- recommendations ----------------
@app.get("/api/recommendations")
def recommend(req: Request, k: int = 10):
    user = current_user(req)
    history = dict(train_history.get(user, {}))
    history.update(db.live_items(user))          # real-time layer
    recs = engine.recommend(history, k=k)
    for r in recs:
        row = catalog[catalog["asin"] == r["asin"]]
        if not row.empty:
            r["product"] = row.iloc[0].to_dict()
    return {"count": len(recs), "recommendations": recs,
            "cold_start": len(history) < 5}


# ---------------- static UI ----------------
@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC), name="static")
