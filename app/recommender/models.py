"""Individual recommendation components and the hybrid scorer.

This is an original implementation inspired by (not copied from) the
literature cited in docs/references.md:
  * content-based filtering on product text (TF-IDF + cosine),
  * item-item collaborative filtering on the user-item matrix,
  * association-rule mining (FP-Growth) over user "baskets",
  * a weighted hybrid with a popularity prior for cold start.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


class PopularityModel:
    """Bayesian-average popularity; the cold-start fallback."""

    name = "popularity"

    def fit(self, train: pd.DataFrame):
        stats = train.groupby("asin")["rating"].agg(["count", "mean"])
        global_mean = train["rating"].mean()
        m = 10.0  # prior strength
        stats["score"] = (
            (stats["count"] * stats["mean"] + m * global_mean)
            / (stats["count"] + m)
        )
        self.scores = stats["score"].to_dict()
        self.default = global_mean / 5.0
        return self

    def score(self, items, user_items=None):
        return np.array([self.scores.get(a, self.default) for a in items]) / 5.0


class ContentBasedModel:
    """User profile = rating-weighted mean of item TF-IDF vectors."""

    name = "content"

    def fit(self, products: pd.DataFrame):
        self.asins = products["asin"].tolist()
        self.asin_index = {a: i for i, a in enumerate(self.asins)}
        self.vectorizer = TfidfVectorizer(
            max_features=40000, ngram_range=(1, 2), min_df=3,
            stop_words="english", sublinear_tf=True,
        )
        self.item_vectors = normalize(
            self.vectorizer.fit_transform(products["doc"].fillna("")))
        return self

    def user_profile(self, user_items: dict):
        """user_items: {asin: rating}. Returns a normalized 1xV vector."""
        idx, w = [], []
        for asin, rating in user_items.items():
            if asin in self.asin_index:
                idx.append(self.asin_index[asin])
                w.append(max(rating, 0.1))
        if not idx:
            return None
        w = np.asarray(w, dtype=np.float32)
        w /= w.sum()
        prof = (self.item_vectors[idx].multiply(w[:, None])).sum(axis=0)
        return normalize(np.asarray(prof))

    def score(self, items, user_items):
        prof = self.user_profile(user_items or {})
        out = np.zeros(len(items), dtype=np.float32)
        if prof is None:
            return out
        idx = [self.asin_index[a] for a in items if a in self.asin_index]
        if not idx:
            return out
        sims = cosine_similarity(prof, self.item_vectors[idx]).ravel()
        pos = {self.asin_index[a]: i for i, a in enumerate(items)
             if a in self.asin_index}
        for j, i in enumerate(idx):
            out[pos[i]] = sims[j]
        return out

    def nearest(self, asin, n=5):
        if asin not in self.asin_index:
            return []
        v = self.item_vectors[self.asin_index[asin]]
        sims = cosine_similarity(v, self.item_vectors).ravel()
        top = np.argpartition(-sims, n + 1)[: n + 1]
        top = top[np.argsort(-sims[top])]
        return [(self.asins[i], float(sims[i])) for i in top
                if self.asins[i] != asin][:n]


class CollaborativeModel:
    """Item-item collaborative filtering with cosine similarity.

    Stores, for every item, its top-K most similar items, so scoring a
    user is a weighted sum over the neighbours of items they interacted
    with - fast enough for real-time requests.
    """

    name = "collaborative"

    def __init__(self, n_neighbors=50):
        self.n_neighbors = n_neighbors

    def fit(self, train: pd.DataFrame, asins):
        self.asin_index = {a: i for i, a in enumerate(asins)}
        users = {u: i for i, u in enumerate(train["user"].unique())}
        rows = train["user"].map(users).to_numpy()
        cols = train["asin"].map(self.asin_index).to_numpy()
        data = train["rating"].to_numpy(dtype=np.float32)
        mat = csr_matrix((data, (rows, cols)),
                         shape=(len(users), len(asins)))
        # mean-centre columns (implicit Pearson-style item-item sim)
        means = np.asarray(mat.mean(axis=0)).ravel()
        mat = mat.tocsc()
        mat.data -= np.repeat(means, np.diff(mat.indptr))
        mat = mat.tocsr()
        sim = cosine_similarity(mat.T, dense_output=False)
        self.neighbors = {}
        indptr, indices, vals = sim.indptr, sim.indices, sim.data
        for i in range(sim.shape[0]):
            s, e = indptr[i], indptr[i + 1]
            if e <= s:
                continue
            nbrs = sorted(zip(indices[s:e], vals[s:e]),
                          key=lambda t: -t[1])[: self.n_neighbors]
            self.neighbors[asins[i]] = [(asins[j], float(v))
                                        for j, v in nbrs if v > 0]
        self.item_means = dict(zip(asins, means))
        return self

    def score(self, items, user_items):
        out = np.zeros(len(items), dtype=np.float32)
        if not user_items:
            return out
        num = defaultdict(float)
        den = defaultdict(float)
        for asin, rating in user_items.items():
            for nbr, sim in self.neighbors.get(asin, []):
                num[nbr] += sim * float(rating)
                den[nbr] += abs(sim)
        q = {a: i for i, a in enumerate(items)}
        for nbr, s in num.items():
            if nbr in q and den[nbr] > 0:
                # normalised to roughly [0,1] from the 1..5 rating scale
                out[q[nbr]] = np.clip(s / den[nbr] / 5.0, 0, 1)
        return out


class AssociationRuleModel:
    """FP-Growth association rules mined from users' purchase baskets.

    A user's 'basket' is the set of items they rated >= threshold. Rules
    X -> Y are scored with confidence*lift so real-time lookup only needs
    membership tests against the user's item set.
    """

    name = "association"

    def __init__(self, min_support=0.002, min_confidence=0.15,
                 rating_threshold=4.0, max_baskets=60000):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.rating_threshold = rating_threshold
        self.max_baskets = max_baskets

    def fit(self, train: pd.DataFrame):
        from mlxtend.frequent_patterns import fpgrowth, association_rules
        from mlxtend.preprocessing import TransactionEncoder

        pos = train[train["rating"] >= self.rating_threshold]
        baskets = (pos.groupby("user")["asin"]
                     .apply(lambda s: list(dict.fromkeys(s))[:50])
                     .tolist())
        baskets = [b for b in baskets if len(b) >= 2]
        if len(baskets) > self.max_baskets:
            rng = np.random.default_rng(42)
            keep = rng.choice(len(baskets), self.max_baskets, replace=False)
            baskets = [baskets[i] for i in keep]
        self.n_baskets = len(baskets)
        te = TransactionEncoder()
        arr = te.fit(baskets).transform(baskets)
        df = pd.DataFrame(arr, columns=te.columns_)
        freq = fpgrowth(df, min_support=self.min_support, use_colnames=True)
        if len(freq) == 0:
            # fall back to a looser threshold once before giving up
            freq = fpgrowth(df, min_support=self.min_support / 10,
                            use_colnames=True)
        if len(freq) == 0:
            self.rules = pd.DataFrame()
            self.by_antecedent = defaultdict(list)
            return self
        rules = association_rules(freq, metric="confidence",
                                  min_threshold=self.min_confidence)
        self.rules = rules
        # antecedent frozenset -> list[(consequent, conf, lift)]
        self.by_antecedent = defaultdict(list)
        for _, r in rules.iterrows():
            key = frozenset(r["antecedents"])
            for c in r["consequents"]:
                self.by_antecedent[key].append(
                    (c, float(r["confidence"]), float(r["lift"])))
        return self

    def score(self, items, user_items):
        out = np.zeros(len(items), dtype=np.float32)
        owned = set((user_items or {}).keys())
        if not owned:
            return out
        q = {a: i for i, a in enumerate(items)}
        for antecedent, cons in self.by_antecedent.items():
            if not antecedent <= owned:
                continue
            for c, conf, lift in cons:
                if c in q:
                    out[q[c]] = max(out[q[c]], min(conf * lift, 5.0) / 5.0)
        return out

    def explain(self, user_items, target):
        """Best rule that recommends `target` for this user."""
        owned = set(user_items or {})
        best = None
        for antecedent, cons in self.by_antecedent.items():
            if antecedent <= owned:
                for c, conf, lift in cons:
                    if c == target and (best is None or conf * lift > best[2]):
                        best = (sorted(antecedent), conf, conf * lift)
        return best
