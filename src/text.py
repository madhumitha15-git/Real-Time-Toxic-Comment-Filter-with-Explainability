import re

LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a",
                      "5": "s", "7": "t", "@": "a", "$": "s"})

def _fix_token(m):
    return m.group(0).translate(LEET)

def clean(t):
    t = str(t).lower()
    t = re.sub(r"http\S+", " url ", t)
    # normalize symbols/digits only inside words, so "id1ot" -> "idiot" but "2024" stays
    t = re.sub(r"(?<=[a-z])[013457@$](?=[a-z])", _fix_token, t)
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)   # stooopid -> stoopid
    t = re.sub(r"[^a-z0-9'\s*!]", " ", t)
    return t