"""Definition audit for NHANES questions every model missed (Q15, Q36, Q42).

For each question: (1) positive controls -- component statistics with known published values, computed with the same
variables, to show the variables are read correctly; (2) alternative definitions -- does the truth move?
Usage (from the repository root): python3 -I nhanes/nhanes_audit.py  -> nhanes/nhanes_audit.csv
"""
import numpy as np, pandas as pd


def load(n):
    t = pd.read_sas(f"nhanes/raw/P_{n}.xpt").set_index("SEQN")
    num = t.select_dtypes("number").columns
    t[num] = t[num].mask(t[num].abs() < 1e-10, 0.0)
    return t


D = load("DEMO")
for n in ["BPXO", "DPQ", "CBC", "SMQ", "SLQ", "BMX", "COT"]:
    t = load(n); D = D.join(t[[c for c in t.columns if c not in D.columns]])
A = D[D.RIDAGEYR >= 20].copy()
A["pulse"] = A[["BPXOPLS1", "BPXOPLS2", "BPXOPLS3"]].mean(axis=1)
items = [f"DPQ0{i}0" for i in range(1, 10)]
A["phq9"] = A[items].where(A[items].apply(lambda c: c.between(0, 3))).sum(axis=1, min_count=9)
A["never"] = A.SMQ020 == 2
A["daily"] = (A.SMQ020 == 1) & (A.SMQ040 == 1)
A["current"] = (A.SMQ020 == 1) & A.SMQ040.isin([1, 2])
A["former"] = (A.SMQ020 == 1) & (A.SMQ040 == 3)
A["notpreg"] = A.RIDEXPRG != 1


def wm(d, c, w="WTMECPRP"):
    d = d[d[w] > 0].dropna(subset=[c]); return np.average(d[c], weights=d[w])


def wdiff(d, c, g, w="WTMECPRP"):
    d = d[d[w] > 0].dropna(subset=[c]); a, b = d[d[g] == 1], d[d[g] == 0]
    return np.average(a[c], weights=a[w]) - np.average(b[c], weights=b[w])


def wcorr(d, x, y, w="WTMECPRP"):
    d = d[d[w] > 0].dropna(subset=[x, y]); ww = d[w] / d[w].sum()
    mx, my = (ww * d[x]).sum(), (ww * d[y]).sum()
    return (ww * (d[x] - mx) * (d[y] - my)).sum() / np.sqrt((ww * (d[x] - mx) ** 2).sum() * (ww * (d[y] - my) ** 2).sum())


def jk(d, fn, w="WTMECPRP"):
    """stratified delete-one-PSU jackknife SE"""
    d = d[d[w] > 0].copy(); est = fn(d); var = 0
    for h, g in d.groupby("SDMVSTRA"):
        ps = g.SDMVPSU.unique(); nh = len(ps)
        if nh < 2:
            continue
        reps = []
        for p in ps:
            e = d.copy(); drop = (e.SDMVSTRA == h) & (e.SDMVPSU == p)
            e.loc[(e.SDMVSTRA == h) & ~drop, w] *= nh / (nh - 1); reps.append(fn(e[~drop]))
        var += (nh - 1) / nh * ((np.array(reps) - est) ** 2).sum()
    return est, var ** .5


rows = []
def rec(q, kind, label, d, fn, ref="", se=True):
    est, s = jk(d, fn) if se else (fn(d), np.nan)
    rows.append(dict(question=q, check=kind, definition=label, estimate=round(float(est), 4),
                     ci_low=round(float(est - 1.96 * s), 4) if se else "", ci_high=round(float(est + 1.96 * s), 4) if se else "",
                     n=len(d), published_reference=ref))
    print(f"{q} | {kind:9} | {label:70} {est:8.3f}" + (f"  [{est-1.96*s:.3f}, {est+1.96*s:.3f}]" if se else "") + f"  n={len(d)}  {ref}")


# ---------------- Q15: pulse, PHQ-9 item 9 >= 1 vs 0 ----------------
q = A[A.DPQ090.between(0, 3)].copy(); q["g"] = (q.DPQ090 >= 1).astype(int)
rec("Q15", "control", "% adults with PHQ-9 item 9 >= 1", q, lambda d: 100 * np.average(d.g, weights=d.WTMECPRP),
    "NHANES 2005-16: ~3-4% of adults")
rec("Q15", "control", "mean resting pulse, adults (bpm)", q.dropna(subset=["pulse"]), lambda d: wm(d, "pulse"),
    "NHANES adults: ~71-73 bpm")
s = A.dropna(subset=["pulse"]).copy(); s = s[s.never | s.daily]; s["g"] = s.daily.astype(int)
rec("Q15", "control", "pulse diff, daily smokers - never (same pulse variable)", s, lambda d: wdiff(d, "pulse", "g"),
    "smokers +3-5 bpm expected")
p = A[A.phq9.notna()].dropna(subset=["pulse"]).copy(); p["g"] = (p.phq9 >= 10).astype(int)
rec("Q15", "control", "pulse diff, PHQ-9 >= 10 vs < 10 (same weights/sample frame)", p, lambda d: wdiff(d, "pulse", "g"),
    "depression +1-3 bpm in literature")
base = q.dropna(subset=["pulse"])
rec("Q15", "truth", "as specified (mean of 3 readings)", base, lambda d: wdiff(d, "pulse", "g"))
b1 = q.dropna(subset=["BPXOPLS1"]).copy(); b1["pulse"] = b1.BPXOPLS1
rec("Q15", "alt", "first reading only", b1, lambda d: wdiff(d, "pulse", "g"))
b3 = q.dropna(subset=["BPXOPLS1", "BPXOPLS2", "BPXOPLS3"])
rec("Q15", "alt", "require all 3 readings", b3, lambda d: wdiff(d, "pulse", "g"))
bb = base.copy(); bb["g"] = (bb.DPQ090 >= 2).astype(int); bb = bb[(bb.DPQ090 >= 2) | (bb.DPQ090 == 0)]
rec("Q15", "alt", "item 9 >= 2 (more than half the days) vs 0", bb, lambda d: wdiff(d, "pulse", "g"))
rec("Q15", "alt", "unweighted", base.assign(WTMECPRP=1.0), lambda d: wdiff(d, "pulse", "g"), se=False)
bp = base[base.phq9.notna()]
rec("Q15", "alt", "only participants with complete PHQ-9", bp, lambda d: wdiff(d, "pulse", "g"))
for lo, hi in [(20, 39), (40, 59), (60, 80)]:
    rec("Q15", "stratum", f"age {lo}-{hi}", base[base.RIDAGEYR.between(lo, hi)], lambda d: wdiff(d, "pulse", "g"))

# ---------------- Q36: RDW, women 20+, daily smokers vs never ----------------
w = A[(A.RIAGENDR == 2) & A.notpreg & A.LBXRDW.notna()].copy()
rec("Q36", "control", "mean RDW, non-pregnant women 20+ (%)", w, lambda d: wm(d, "LBXRDW"), "NHANES: ~13.5-14%")
rec("Q36", "control", "% daily smokers among women 20+ with SMQ", w[w.SMQ020.isin([1, 2])],
    lambda d: 100 * np.average(d.daily, weights=d.WTMECPRP), "NHIS 2018: ~11-12% current, ~9% daily women")
ww = w[w.never | w.daily].copy(); ww["g"] = ww.daily.astype(int)
rec("Q36", "truth", "as specified (daily vs never)", ww, lambda d: wdiff(d, "LBXRDW", "g"))
m = A[(A.RIAGENDR == 1) & A.LBXRDW.notna() & (A.never | A.daily)].copy(); m["g"] = m.daily.astype(int)
rec("Q36", "control", "same contrast in men", m, lambda d: wdiff(d, "LBXRDW", "g"), "")
c = w[w.never | w.current].copy(); c["g"] = c.current.astype(int)
rec("Q36", "alt", "current (every OR some days) vs never", c, lambda d: wdiff(d, "LBXRDW", "g"))
f = w[w.never | w.former].copy(); f["g"] = f.former.astype(int)
rec("Q36", "alt", "former vs never", f, lambda d: wdiff(d, "LBXRDW", "g"))
ck = w[w.never | w.daily].copy(); ck = ck[(ck.daily & (ck.LBXCOT >= 10)) | (ck.never & (ck.LBXCOT < 1))]
ck["g"] = ck.daily.astype(int)
rec("Q36", "alt", "cotinine-confirmed (daily & cot>=10 vs never & cot<1)", ck, lambda d: wdiff(d, "LBXRDW", "g"))
wi = A[(A.RIAGENDR == 2) & A.LBXRDW.notna() & (A.never | A.daily)].copy(); wi["g"] = wi.daily.astype(int)
rec("Q36", "alt", "pregnant women not excluded", wi, lambda d: wdiff(d, "LBXRDW", "g"))
wr = ww[ww.LBXRDW <= 20]
rec("Q36", "alt", "RDW > 20% dropped (outliers)", wr, lambda d: wdiff(d, "LBXRDW", "g"))
ww["lrdw"] = np.log(ww.LBXRDW)
rec("Q36", "alt", "geometric: difference in log RDW x 100 (~% difference)", ww, lambda d: 100 * wdiff(d, "lrdw", "g"))
for lo, hi in [(20, 39), (40, 59), (60, 80)]:
    rec("Q36", "stratum", f"age {lo}-{hi}", ww[ww.RIDAGEYR.between(lo, hi)], lambda d: wdiff(d, "LBXRDW", "g"))

# ---------------- Q42: corr(log WBC, weekday sleep) ----------------
z = A[(A.LBXWBCSI > 0)].copy(); z["lwbc"] = np.log(z.LBXWBCSI)
zs = z[z.SLD012.notna()]
rec("Q42", "control", "mean weekday sleep, adults (h)", zs, lambda d: wm(d, "SLD012"), "NHANES 2017-20: ~7.6 h")
rec("Q42", "control", "corr(log WBC, BMI)", z, lambda d: wcorr(d, "lwbc", "BMXBMI"), "positive, ~0.15-0.25")
z["cs"] = z.current.astype(float).where(z.SMQ020.isin([1, 2]))
rec("Q42", "control", "corr(log WBC, current smoking)", z, lambda d: wcorr(d, "lwbc", "cs"), "positive, ~0.15-0.25")
rec("Q42", "truth", "as specified (log WBC, weekday sleep)", zs, lambda d: wcorr(d, "lwbc", "SLD012"))
rec("Q42", "alt", "WBC not logged", zs, lambda d: wcorr(d, "LBXWBCSI", "SLD012"))
rec("Q42", "alt", "WBC > 25 dropped", zs[zs.LBXWBCSI <= 25], lambda d: wcorr(d, "lwbc", "SLD012"))
zw = z.dropna(subset=["SLD013"])
rec("Q42", "alt", "weekend sleep instead", zw, lambda d: wcorr(d, "lwbc", "SLD013"))
za = z.dropna(subset=["SLD012", "SLD013"]).copy(); za["avg"] = (5 * za.SLD012 + 2 * za.SLD013) / 7
rec("Q42", "alt", "weekly average sleep (5 weekday + 2 weekend)/7", za, lambda d: wcorr(d, "lwbc", "avg"))
zs2 = zs.copy(); zs2["short"] = (zs2.SLD012 < 6).astype(float)
rec("Q42", "alt", "short sleep (<6 h) indicator instead of hours", zs2, lambda d: wcorr(d, "lwbc", "short"))
zs2["rw"], zs2["rs"] = zs2.lwbc.rank(), zs2.SLD012.rank()
rec("Q42", "alt", "rank (Spearman-type) correlation", zs2, lambda d: wcorr(d, "rw", "rs"))
rec("Q42", "alt", "unweighted", zs.assign(WTMECPRP=1.0), lambda d: wcorr(d, "lwbc", "SLD012"), se=False)

pd.DataFrame(rows).to_csv("nhanes/nhanes_audit.csv", index=False)
print("wrote nhanes/nhanes_audit.csv")
