import sys, joblib, pandas as pd

b = joblib.load("model.joblib")
df = pd.read_csv(sys.argv[1])
col = sys.argv[2] if len(sys.argv) > 2 else "comment_text"
out = sys.argv[3] if len(sys.argv) > 3 else "predictions.csv"
p = b["model"].predict_proba(df[col].astype(str))[:, 1]
df["toxic_prob"] = p
df["prediction"] = (p >= b["threshold"]).astype(int)   # plain binary decision, no review band
df.to_csv(out, index=False)
print("saved", out)