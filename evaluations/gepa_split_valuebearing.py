"""Recompute Table 06-TAB-gepa on the 489 value-bearing cells, under both matchers.
Run from evaluations/Baseline-PipelineA-PipelineB/. Reproduces 06-TAB-main as a sanity check.
"""
import pandas as pd, numpy as np, random

a = pd.read_json("Baseline-PipelineA-PipelineB_ynorm.json")
V = ["Base", "A", "B", "G"]
for c in [f"{p}_{v}" for p in ["value", "unit", "label"] for v in V]:
    a[c] = a[c].apply(lambda x: np.nan if x is None else x)

def hit_any(r, ec, gc):                       # membership, as in 02-Eval cell 8
    return (r[gc] in r[ec]) if isinstance(r[ec], list) else pd.isna(r[gc])

def hit_ex(r, ec, gc):                        # set equality, as in 02-Eval cell 8
    g, e = r[gc], r[ec]
    gs = set(g) if isinstance(g, list) else (set() if pd.isna(g) else {g})
    es = set(e) if isinstance(e, list) else (set() if pd.isna(e) else {e})
    return gs == es

for v in V:
    a[v + "_any"] = a.apply(hit_any, args=(f"value_{v}", "value"), axis=1)
    a[v + "_ex"]  = a.apply(hit_ex,  args=(f"value_{v}", "value"), axis=1)

vb = a["value"].notna()
print("sanity 06-TAB-main, all cells ANY:", {v: round(a[v+"_any"].mean()*100, 2) for v in V})
print("sanity 06-TAB-main, VB ANY:      ", {v: round(a.loc[vb, v+"_any"].mean()*100, 2) for v in V})

# same seeded split as 02-Eval-PipeB.ipynb cells 33-34
h = a.groupby("report_name")["B_any"].all()
T, F = h[h].index.tolist(), h[~h].index.tolist()
random.seed(42)
train = set(random.sample(T, round(len(T)*.6)) + random.sample(F, round(len(F)*.6)))

print("\n=== B -> B_G by split ===")
for nm, m0 in [("train (32)", a.report_name.isin(train)), ("held-out (22)", ~a.report_name.isin(train))]:
    for scope_lbl, m in [("all cells", m0), ("value-bearing", m0 & vb)]:
        for mt in ["any", "ex"]:
            b, g = a.loc[m, "B_"+mt].mean()*100, a.loc[m, "G_"+mt].mean()*100
            print(f"{nm:14s} {scope_lbl:14s} {mt.upper():4s} n={m.sum():5d}  "
                  f"B={b:6.2f}  B_G={g:6.2f}  delta={g-b:+6.2f}")

print("\n=== Net cell movement B -> B_G (value-bearing, ANY) ===")
won  = a[vb & ~a.B_any &  a.G_any].groupby("scope").size()
lost = a[vb &  a.B_any & ~a.G_any].groupby("scope").size()
print(pd.DataFrame({"won": won, "lost": lost}).fillna(0).astype(int).assign(net=lambda x: x.won-x.lost).to_string())

################
### OUTPUT:
# sanity 06-TAB-main, all cells ANY: {'Base': np.float64(94.66), 'A': np.float64(95.47), 'B': np.float64(95.06), 'G': np.float64(96.88)}
# sanity 06-TAB-main, VB ANY:       {'Base': np.float64(89.57), 'A': np.float64(92.23), 'B': np.float64(90.8), 'G': np.float64(93.05)}

# === B -> B_G by split ===
# train (32)     all cells      ANY  n= 1316  B= 94.60  B_G= 96.73  delta= +2.13
# train (32)     all cells      EX   n= 1316  B= 93.01  B_G= 96.35  delta= +3.34
# train (32)     value-bearing  ANY  n=  274  B= 87.96  B_G= 89.78  delta= +1.82
# train (32)     value-bearing  EX   n=  274  B= 80.29  B_G= 87.96  delta= +7.66
# held-out (22)  all cells      ANY  n=  892  B= 95.74  B_G= 97.09  delta= +1.35
# held-out (22)  all cells      EX   n=  892  B= 95.74  B_G= 95.96  delta= +0.22
# held-out (22)  value-bearing  ANY  n=  215  B= 94.42  B_G= 97.21  delta= +2.79
# held-out (22)  value-bearing  EX   n=  215  B= 94.42  B_G= 92.56  delta= -1.86

# === Net cell movement B -> B_G (value-bearing, ANY) ===
#        won  lost  net
# scope                
# 1        1     5   -4
# 2lb     24     3   21
# 2mb      4     7   -3
# 3        1     4   -3
#################