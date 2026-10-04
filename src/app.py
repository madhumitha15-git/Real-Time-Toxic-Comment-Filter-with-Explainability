import time, html, joblib, streamlit as st

st.set_page_config(page_title="Toxic Comment Filter", page_icon="🛡️")
BAND = 0.12   # must match BAND in src/train.py
SARCASM = ["oh great", "yeah right", "thanks a lot", "real genius",
           "nice job", "sure, genius", "how smart", "wow, such"]

@st.cache_resource
def load():
    b = joblib.load("model.joblib")
    b["model"].predict_proba(["warm up"])
    return b

bundle = load()
model, thr = bundle["model"], bundle["threshold"]

def explain(words):
    base = model.decision_function([""])[0]      # score of an empty comment
    logits = model.decision_function(words)      # each word scored alone, one batched call
    return [max(0.0, l - base) for l in logits]

def render(words, impact):
    top = max(impact) if impact else 0
    cutoff = max(0.3, 0.2 * top)
    out = []
    for w, s in zip(words, impact):
        a = s / top if top > 0 else 0
        style = f"background:rgba(255,75,75,{0.25 + 0.65 * a:.2f});" if s >= cutoff else ""
        out.append(f"<span style='{style}padding:2px 4px;border-radius:4px'>{html.escape(w)}</span>")
    return " ".join(out)
st.title("🛡️ Toxic Comment Filter")
st.caption("TF-IDF (word + character n-grams) + Logistic Regression, with word-level explanations")

examples = {
    "Neutral": "Thanks for sharing, this was really helpful.",
    "Obvious abuse": "You are a worthless idiot and everyone hates you.",
    "Misspelled": "what a stooopid take, you f*cking id1ot",
    "Sarcasm": "Oh great, another genius idea from the expert.",
}
choice = st.radio("Try an example:", list(examples), horizontal=True)
text = st.text_area("Comment", examples[choice], height=100)

if st.button("Analyze", type="primary") and text.strip():
    t0 = time.perf_counter()
    prob = model.predict_proba([text])[0, 1]
    pred_ms = (time.perf_counter() - t0) * 1000

    words = text.split()[:40]
    t1 = time.perf_counter()
    impact = explain(words)
    exp_ms = (time.perf_counter() - t1) * 1000

    if prob >= thr + BAND:
        verdict = "TOXIC"
    elif prob <= thr - BAND:
        verdict = "Non-toxic"
    else:
        verdict = "Needs human review"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Verdict", verdict)
    c2.metric("Toxicity score", f"{prob:.2f}")
    c3.metric("Prediction", f"{pred_ms:.0f} ms")
    c4.metric("Explanation", f"{exp_ms:.0f} ms")

    if verdict == "Non-toxic":
        st.success("No words pushed this comment toward toxic.")
    else:
        st.markdown("**Words driving the decision** (darker = stronger):")
        st.markdown(render(words, impact), unsafe_allow_html=True)
        with st.expander("Per-word scores"):
            st.table({"word": words, "impact": [round(float(s), 2) for s in impact]})


    if verdict == "Non-toxic" and any(c in text.lower() for c in SARCASM):
        st.warning("Possible sarcasm. Tone is hard for this model, so a human should review this one.")