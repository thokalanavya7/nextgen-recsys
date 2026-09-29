"""SQLite persistence for users, sessions, and live interaction events.

The product catalog and trained model live in memory (loaded from the
processed dataset); everything the running app collects - signups and
clicks/views/ratings/purchases - lands here so recommendations update in
real time.
"""
import hashlib
import os
import secrets
import sqlite3
import time
from contextlib import contextmanager

DB_PATH = os.environ.get(
    "APP_DB",
    os.path.join(os.path.dirname(__file__), "..", "app.db"))

EVENT_WEIGHT = {"view": 1.0, "click": 2.0, "rate": None, "purchase": 4.0}


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init():
    with _conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS users(
            username TEXT PRIMARY KEY,
            pass_hash TEXT NOT NULL,
            created REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sessions(
            token TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            created REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS interactions(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            asin TEXT NOT NULL,
            event TEXT NOT NULL,
            rating REAL,
            ts REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS ix_interactions_user
            ON interactions(username);
        """)


def _hash(pw: str) -> str:
    return hashlib.sha256(("recsys:" + pw).encode()).hexdigest()


def create_user(username: str, password: str) -> bool:
    try:
        with _conn() as c:
            c.execute("INSERT INTO users VALUES(?,?,?)",
                      (username, _hash(password), time.time()))
        return True
    except sqlite3.IntegrityError:
        return False


def verify_user(username: str, password: str) -> bool:
    with _conn() as c:
        row = c.execute("SELECT pass_hash FROM users WHERE username=?",
                        (username,)).fetchone()
    return bool(row) and row["pass_hash"] == _hash(password)


def create_session(username: str) -> str:
    token = secrets.token_hex(24)
    with _conn() as c:
        c.execute("INSERT INTO sessions VALUES(?,?,?)",
                  (token, username, time.time()))
    return token


def user_for_token(token: str):
    if not token:
        return None
    with _conn() as c:
        row = c.execute("SELECT username FROM sessions WHERE token=?",
                        (token,)).fetchone()
    return row["username"] if row else None


def drop_session(token: str):
    with _conn() as c:
        c.execute("DELETE FROM sessions WHERE token=?", (token,))


def log_interaction(username, asin, event, rating=None):
    with _conn() as c:
        c.execute(
            "INSERT INTO interactions(username,asin,event,rating,ts) "
            "VALUES(?,?,?,?,?)",
            (username, asin, event, rating, time.time()))


def live_items(username):
    """Aggregate a user's live interactions into {asin: strength} on the
    same 1..5 scale as the trained model."""
    with _conn() as c:
        rows = c.execute(
            "SELECT asin,event,rating FROM interactions "
            "WHERE username=? ORDER BY ts", (username,)).fetchall()
    items = {}
    for r in rows:
        if r["event"] == "rate" and r["rating"]:
            items[r["asin"]] = max(items.get(r["asin"], 0),
                                   float(r["rating"]))
        else:
            items[r["asin"]] = max(items.get(r["asin"], 0),
                                   EVENT_WEIGHT.get(r["event"], 1.0))
    return items
