"""
5-fold cross-validated threshold selection: choose tau on 4 folds, evaluate on the
held-out fold, report mean +/- std F1/P/R. Removes the 'threshold tuned on the test
set' bias and yields error bars. No experiments re-run: 7 methods use saved per-record
scores; Soft TF-IDF scores are recomputed (real IDF-weighted + Jaro-Winkler version).
"""
import numpy as np, pandas as pd, jellyfish
from pathlib import Path
from collections import defaultdict
from scipy.sparse import identity, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

RES = Path("/Users/jhelvy/gh/research/fuzzy-join-llm/results")
DATA = Path("/Users/jhelvy/gh/research/fuzzy-join-llm/data")
CFG = {
 "abt-buy": dict(left="abt", right="buy", mp="abt-buy/data_original/abt_buy_perfect_mapping.csv"),
 "amazon-google": dict(left="amazon", right="google", mp="amazon-google/data_original/amazon_googleproducts_perfectmapping.csv"),
}
TRANS = ["original", "ciphered_letters", "ciphered_words", "scrambled"]
GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)
SEED, K = 42, 5

def load(p):
    for e in ["utf-8","latin-1","cp1252"]:
        try: return pd.read_csv(p, encoding=e)
        except Exception: pass

def prep(df):
    df = df.copy()
    cols = [c for c in df.columns[1:] if pd.api.types.is_string_dtype(df[c])
            and c.lower() not in ("year","price","id")]
    df["text"] = df[cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower().str.strip()
    return df

def truth_set(ds):
    mp = load(DATA / CFG[ds]["mp"]); return set(zip(mp.iloc[:,0].astype(str), mp.iloc[:,1].astype(str)))

def prf(true_sub, score_sub, tau):
    pred = score_sub >= tau
    tp = int((pred & true_sub).sum()); fp = int((pred & ~true_sub).sum()); fn = len(true_sub) - tp
    P = tp/(tp+fp) if tp+fp else 0.0; R = tp/(tp+fn) if tp+fn else 0.0
    return (2*P*R/(P+R) if P+R else 0.0), P, R

def cv(true, score):
    rng = np.random.default_rng(SEED); idx = rng.permutation(len(true)); folds = np.array_split(idx, K)
    F=[];P=[];R=[]
    for f in range(K):
        te = folds[f]; tr = np.concatenate([folds[j] for j in range(K) if j!=f])
        tau = max(GRID, key=lambda t: prf(true[tr], score[tr], t)[0])
        fF,fP,fR = prf(true[te], score[te], tau); F.append(fF);P.append(fP);R.append(fR)
    return np.mean(F),np.std(F),np.mean(P),np.mean(R)

def fuzzy_matrix(vocab, theta=0.9):
    buckets=defaultdict(list)
    for i,w in enumerate(vocab): buckets[w[0] if w else ""].append(i)
    r=[];c=[];v=[]
    for idxs in buckets.values():
        for a in range(len(idxs)):
            i=idxs[a]; wi=vocab[i]
            for b in range(a+1,len(idxs)):
                j=idxs[b]; wj=vocab[j]
                if abs(len(wi)-len(wj))>3: continue
                s=jellyfish.jaro_winkler_similarity(wi,wj)
                if s>=theta: r+=[i,j];c+=[j,i];v+=[s,s]
    return csr_matrix((v,(r,c)),shape=(len(vocab),len(vocab)))

def soft_tfidf_scores(ds,t):
    cfg=CFG[ds]; base=DATA/ds; mp=load(base/"data_original"/Path(cfg['mp']).name)
    ml=set(mp.iloc[:,0].astype(str)); mr=set(mp.iloc[:,1].astype(str))
    if t=="original":
        lf=base/"data_original"/f"{cfg['left']}.csv"; rf=base/"data_original"/f"{cfg['right']}.csv"
        if not lf.exists():
            for c in (base/"data_original").glob("*.csv"):
                if c.stem.lower()==cfg['left']: lf=c
                if c.stem.lower()==cfg['right']: rf=c
    else:
        lf=base/"data_test"/f"{cfg['left']}_{t}.csv"; rf=base/"data_test"/f"{cfg['right']}_{t}.csv"
    L=load(lf); R=load(rf); lid0,rid0=L.columns[0],R.columns[0]
    L=L[L[lid0].astype(str).isin(ml)].copy(); R=R[R[rid0].astype(str).isin(mr)].copy()
    L,R=prep(L),prep(R)
    vec=TfidfVectorizer(norm="l2"); vec.fit(pd.concat([L["text"],R["text"]]))
    vocab=vec.get_feature_names_out(); Lm=vec.transform(L["text"]); Rm=vec.transform(R["text"])
    M=fuzzy_matrix(vocab); I=identity(len(vocab),format="csr")
    S=np.asarray((Lm@(I+M)@Rm.T).todense())
    rids=R[rid0].astype(str).values; lids=L[lid0].astype(str).values
    bi=S.argmax(axis=1)
    return S[np.arange(len(S)),bi], lids, rids[bi]

out=[]
print(f"{'dataset':14s}{'method':20s}{'trans':16s}{'F1 mean':>9s}{'std':>7s}{'P':>7s}{'R':>7s}")
for ds in CFG:
    truth=truth_set(ds)
    for mdir in sorted((RES/ds).iterdir()):
        if not mdir.is_dir() or mdir.name=="soft_tfidf": continue
        for t in TRANS:
            pq=mdir/f"{t}.parquet"
            if not pq.exists(): continue
            df=pd.read_parquet(pq)
            pid=df["pred_id_right"].astype(str).values; lid=df["id_left"].astype(str).values
            score=df["similarity_score"].fillna(0).values
            score=np.where((pid!="")&(pid!="None"),score,0.0)
            true=np.array([(lid[i],pid[i]) in truth for i in range(len(df))])
            m,s,P,R=cv(true,score); out.append((ds,mdir.name,t,m,s,P,R))
            print(f"{ds:14s}{mdir.name:20s}{t:16s}{m:9.3f}{s:7.3f}{P:7.3f}{R:7.3f}")
    for t in TRANS:
        score,lid,pid=soft_tfidf_scores(ds,t)
        true=np.array([(lid[i],pid[i]) in truth for i in range(len(lid))])
        m,s,P,R=cv(true,score); out.append((ds,"soft_tfidf",t,m,s,P,R))
        print(f"{ds:14s}{'soft_tfidf':20s}{t:16s}{m:9.3f}{s:7.3f}{P:7.3f}{R:7.3f}")

_out = Path(__file__).resolve().parent.parent / "results" / "cv_results.csv"
pd.DataFrame(out,columns=["dataset","method","transformation","f1_mean","f1_std","precision","recall"]).to_csv(
    _out, index=False)
print(f"\nsaved -> {_out}")
