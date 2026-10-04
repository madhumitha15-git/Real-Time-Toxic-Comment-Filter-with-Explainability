import joblib

b = joblib.load("model.joblib")
m, thr = b["model"], b["threshold"]
cases = [
    ("you are a stooopid idiot", "toxic"),
    ("what an id1ot", "toxic"),
    ("you f*cking moron", "toxic"),
    ("SHUT UP you worthless loser!!!", "toxic"),
    ("Oh great, another genius idea from the expert.", "toxic (sarcasm)"),
    ("Thanks for sharing, this was helpful.", "non-toxic"),
    ("I am a proud Muslim woman", "non-toxic"),
    ("This movie was sick, loved it", "non-toxic"),
]
rows = ["| Comment | Expected | Score | Verdict |", "|---|---|---|---|"]
for text, exp in cases:
    p = m.predict_proba([text])[0, 1]
    rows.append(f"| {text} | {exp} | {p:.2f} | {'toxic' if p >= thr else 'non-toxic'} |")
open("edge_cases_results.md", "w").write("\n".join(rows))
print("\n".join(rows))