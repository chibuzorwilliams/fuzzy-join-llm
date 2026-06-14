"""
Independent re-computation of TF-IDF blocking recall under each transformation,
replicating the experiment's setup (filter BOTH catalogs to matched records,
fit TF-IDF on left+right, check if each left record's true right match is in top-k).

This validates the load-bearing claim for the paper's thesis:
'blocking recall remains >95% even under corruption'.
"""
import pandas as pd, numpy as np, sys
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path("/Users/jhelvy/gh/research/fuzzy-join-llm/data")

DSETS = {
    "abt-buy": dict(d="abt-buy", left="abt", right="buy",
                    mapfile="abt_buy_perfect_mapping.csv"),
    "amazon-google": dict(d="amazon-google", left="amazon", right="google",
                          mapfile="amazon_googleproducts_perfectmapping.csv"),
}
TRANS = ["original", "ciphered_letters", "ciphered_words", "scrambled"]

def load_csv(p):
    for enc in ["utf-8", "latin-1", "iso-8859-1", "cp1252"]:
        try: return pd.read_csv(p, encoding=enc)
        except Exception: continue
    raise ValueError(f"cannot load {p}")

def prep(df):
    df = df.copy()
    cols = [c for c in df.columns[1:]
            if pd.api.types.is_string_dtype(df[c])
            and c.lower() not in ("year", "price", "id")]
    txt = df[cols].fillna("").astype(str).agg(" ".join, axis=1)
    df["text"] = txt.str.lower().str.strip()
    return df

def run(name, top_k=50):
    cfg = DSETS[name]
    base = ROOT / cfg["d"]
    mp = load_csv(base / "data_original" / cfg["mapfile"])
    lcol_m, rcol_m = mp.columns[0], mp.columns[1]
    matched_left = set(mp[lcol_m].astype(str))
    matched_right = set(mp[rcol_m].astype(str))
    truth = set(zip(mp[lcol_m].astype(str), mp[rcol_m].astype(str)))

    print(f"\n=== {name}  (top_k={top_k}) ===")
    print(f"{'transformation':18s} {'block_recall':>12s} {'found/total':>14s}")
    for t in TRANS:
        if t == "original":
            lf = base / "data_original" / f"{cfg['left']}.csv"
            rf = base / "data_original" / f"{cfg['right']}.csv"
            if not lf.exists():  # case variants
                for cand in base.glob("data_original/*.csv"):
                    if cand.stem.lower() == cfg["left"]: lf = cand
                    if cand.stem.lower() == cfg["right"]: rf = cand
        else:
            lf = base / "data_test" / f"{cfg['left']}_{t}.csv"
            rf = base / "data_test" / f"{cfg['right']}_{t}.csv"
        L, R = load_csv(lf), load_csv(rf)
        lid, rid = L.columns[0], R.columns[0]
        # filter to matched records (as run_experiments.py does)
        L = L[L[lid].astype(str).isin(matched_left)].copy()
        R = R[R[rid].astype(str).isin(matched_right)].copy()
        L, R = prep(L), prep(R)
        vec = TfidfVectorizer()
        vec.fit(pd.concat([L["text"], R["text"]]))
        sim = cosine_similarity(vec.transform(L["text"]), vec.transform(R["text"]))
        rids = R[rid].astype(str).values
        found = total = 0
        for i in range(len(L)):
            lid_i = str(L.iloc[i][lid])
            true_r = {r for (l, r) in truth if l == lid_i}
            if not true_r: continue
            total += 1
            topk = set(rids[sim[i].argsort()[::-1][:top_k]])
            if true_r & topk: found += 1
        print(f"{t:18s} {found/total:12.4f} {f'{found}/{total}':>14s}")

if __name__ == "__main__":
    for n in DSETS: run(n, top_k=50)
