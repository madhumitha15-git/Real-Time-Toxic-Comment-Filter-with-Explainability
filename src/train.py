import sys, time, json, joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline, make_union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from src.text import clean

# Usage: python train.py 1.0  (no bias mitigation, baseline)
#        python train.py 3.0  (identity-mention non-toxic comments upweighted, final model)
W = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
BAND = 0.12          # keep in sync with BAND in app.py
N_ROWS = 150_000     # lower to 300_000 if training is too slow

IDENT = ["male", "female", "black", "white", "muslim", "jewish",
         "christian", "homosexual_gay_or_lesbian", "psychiatric_or_mental_illness"]

df = pd.read_csv("data/train.csv", usecols=["comment_text", "target"] + IDENT)
df = df.sample(min(len(df), N_ROWS), random_state=42)
df["comment_text"] = df["comment_text"].astype(str).str[:600]
y = (df.target >= 0.5).astype(int)
ident = df[IDENT].fillna(0).max(axis=1) >= 0.5

# same random_state in every run, so the W=1.0 and W=3.0 runs share one split
X_tr, X_te, y_tr, y_te, id_tr, id_te = train_test_split(
    df.comment_text, y, ident, test_size=0.1, stratify=y, random_state=42)

# bias mitigation: upweight NON-toxic comments that mention identity terms
w = np.where(id_tr & (y_tr == 0), W, 1.0)

vec = make_union(
    TfidfVectorizer(preprocessor=clean, ngram_range=(1, 2), min_df=3,
                    max_features=150_000, sublinear_tf=True, dtype=np.float32),
    TfidfVectorizer(preprocessor=clean, analyzer="char_wb", ngram_range=(3, 5),
                    min_df=5, max_features=150_000, sublinear_tf=True, dtype=np.float32),
)
model = make_pipeline(vec, LogisticRegression(
    C=4, class_weight="balanced", max_iter=300, solver="liblinear"))
model.fit(X_tr, y_tr, logisticregression__sample_weight=w)

# ---- evaluation ----
p = model.predict_proba(X_te)[:, 1]
f1, thr = max((f1_score(y_te, p >= t), t) for t in [i / 20 for i in range(4, 18)])

yt, it = y_te.values, id_te.values
m_ident = it & (yt == 0)       # non-toxic comments that mention an identity group
m_other = ~it & (yt == 0)      # all other non-toxic comments
fpr_ident = ((p >= thr) & m_ident).sum() / max(m_ident.sum(), 1)
fpr_other = ((p >= thr) & m_other).sum() / max(m_other.sum(), 1)

# uncertainty band: how much is decided automatically, and how good is it
conf = np.abs(p - thr) > BAND
f1_auto = f1_score(yt[conf], p[conf] >= thr) if conf.sum() else float("nan")

# latency on a single comment (one warm-up call first so it's not a cold start)
model.predict_proba(["warm up"])
t0 = time.perf_counter()
model.predict_proba(["you are a stooopid idiot"])
lat = (time.perf_counter() - t0) * 1000

metrics = {
    "identity_weight": W,
    "f1": round(float(f1), 3),
    "threshold": thr,
    "latency_ms": round(lat, 1),
    "fpr_identity_nontoxic": round(float(fpr_ident), 3),
    "fpr_other_nontoxic": round(float(fpr_other), 3),
    "n_identity_nontoxic": int(m_ident.sum()),
    "auto_decided_pct": round(float(conf.mean()) * 100, 1),
    "f1_auto_decided": round(float(f1_auto), 3),
}
print(metrics)
json.dump(metrics, open(f"metrics_w{W}.json", "w"), indent=2)

# only the final (mitigated) model is saved for the app
if W != 1.0:
    joblib.dump({"model": model, "threshold": thr}, "model.joblib")