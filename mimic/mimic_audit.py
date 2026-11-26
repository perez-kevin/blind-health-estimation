"""Definition audit for M01 (AMA, Medicaid 18-39) and M41 (IV methylprednisolone, asthma 18-49).

Same privacy rules as mimic_truth.py: aggregates only; groups < 20 patients suppressed; counts 1-10 shown as "<11";
category listings (discharge locations, insurance values, drug/route strings) show counts only, never patient rows.

Usage (same folder argument as mimic_truth.py):
    python3 -I mimic_audit.py "/path/to/mimic-iv full" [mimic_audit.json]
"""
import json, os, sys
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings("ignore")

ROOT = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "mimic_audit.json"
MIN_PATIENTS, MIN_CELL = 20, 11
if os.environ.get("DEMO_NO_SUPPRESS") == "1":  # only for the open-access demo
    MIN_PATIENTS, MIN_CELL = 0, 0


def path(name):
    for p in (os.path.join(ROOT, f"{name}.csv.gz"), os.path.join(ROOT, "hosp", f"{name}.csv.gz")):
        if os.path.exists(p):
            return p
    raise FileNotFoundError(name)


print("loading tables ...", flush=True)
adm = pd.read_csv(path("admissions"), parse_dates=["admittime", "dischtime"])
pat = pd.read_csv(path("patients"))
dx = pd.read_csv(path("diagnoses_icd"), usecols=["hadm_id", "seq_num", "icd_code", "icd_version"], dtype={"icd_code": str})
rx = pd.read_csv(path("prescriptions"), usecols=["hadm_id", "drug", "route"], dtype={"drug": str, "route": str})
A = adm.merge(pat[["subject_id", "gender", "anchor_age", "anchor_year"]], on="subject_id", how="left")
A["age"] = A.anchor_age + (A.admittime.dt.year - A.anchor_year)
A = A.set_index("hadm_id", drop=False)
dx["icd_code"] = dx.icd_code.str.strip().str.upper()
drug_l, route_u = rx.drug.fillna("").str.lower(), rx.route.fillna("").str.upper()
out = {"estimates": [], "listings": {}}


def dxset(icd9=(), icd10=(), principal=False, regex9=None, regex10=None):
    d = dx[dx.seq_num == 1] if principal else dx
    m = pd.Series(False, index=d.index)
    if icd9: m |= (d.icd_version == 9) & d.icd_code.str.startswith(tuple(icd9))
    if icd10: m |= (d.icd_version == 10) & d.icd_code.str.startswith(tuple(icd10))
    if regex9: m |= (d.icd_version == 9) & d.icd_code.str.match(regex9)
    if regex10: m |= (d.icd_version == 10) & d.icd_code.str.match(regex10)
    return A.hadm_id.isin(set(d.loc[m, "hadm_id"]))


def rxset(drug_pat, route_mask=None):
    m = drug_l.str.contains(drug_pat, regex=True)
    if route_mask is not None: m &= route_mask
    return A.hadm_id.isin(set(rx.loc[m, "hadm_id"].dropna().astype(int)))


def cnt(n):
    return int(n) if (n == 0 or n >= MIN_CELL) else "<11"


def prop(qid, label, base, outcome, kind="alt"):
    d = A[base]; y = outcome[base].astype(float).values
    n, k, npat = len(d), int(y.sum()), d.subject_id.nunique()
    if n == 0:
        print(f"{qid} | {kind:8} | {label:72} EMPTY"); return
    if npat < MIN_PATIENTS or 0 < k < MIN_CELL or 0 < n - k < MIN_CELL:
        print(f"{qid} | {kind:8} | {label:72} SUPPRESSED (small cell)")
        out["estimates"].append(dict(question=qid, check=kind, definition=label, suppressed=True)); return
    p = y.mean()
    s = pd.Series((y - p) / n).groupby(d.subject_id.values).sum().values
    se = np.sqrt(s @ s * len(s) / (len(s) - 1))
    print(f"{qid} | {kind:8} | {label:72} {100*p:6.2f}%  [{100*(p-1.96*se):.2f}, {100*(p+1.96*se):.2f}]  n_adm={n} n_pat={npat}")
    out["estimates"].append(dict(question=qid, check=kind, definition=label, estimate=100 * p,
                                 ci95=[100 * (p - 1.96 * se), 100 * (p + 1.96 * se)], n_admissions=n, n_patients=npat))


def listing(name, series, top=40):
    vc = series.fillna("(missing)").value_counts().head(top)
    out["listings"][name] = {str(k): cnt(v) for k, v in vc.items()}
    print(f"\n--- {name} (top {top}, counts; 1-10 masked) ---")
    for k, v in out["listings"][name].items():
        print(f"  {v!s:>9}  {k}")
    print()


# ======================= M01: AMA among Medicaid admissions aged 18-39 =======================
AMA = A.discharge_location == "AGAINST ADVICE"
young, medicaid = A.age.between(18, 39), A.insurance == "Medicaid"
listing("M01 insurance values (all admissions)", A.insurance)
listing("M01 discharge_location values (Medicaid 18-39)", A.loc[young & medicaid, "discharge_location"])
listing("M01 admission_type values (Medicaid 18-39)", A.loc[young & medicaid, "admission_type"])
OB = dxset(regex9=r"^(6[3-7]\d|V2[2-4]|V27|V3\d)", regex10=r"^(O|Z3[0-9]|Z37|Z38)")
NB = dxset(regex9=r"^V3\d", regex10=r"^Z38")
EMERG = A.admission_type.isin(["EW EMER.", "DIRECT EMER.", "URGENT", "EU OBSERVATION", "DIRECT OBSERVATION",
                               "OBSERVATION ADMIT", "AMBULATORY OBSERVATION"])
prop("M01", "control: AMA, all admissions", A.hadm_id.notna(), AMA, "control")
prop("M01", "control: AMA, all adults 18-39", young, AMA, "control")
prop("M01", "control: AMA, Medicaid all ages", medicaid, AMA, "control")
prop("M01", "control: discharge_location missing, Medicaid 18-39", young & medicaid, A.discharge_location.isna(), "control")
prop("M01", "control: obstetric/delivery admission, Medicaid 18-39", young & medicaid, OB, "control")
prop("M01", "as specified", young & medicaid, AMA, "truth")
prop("M01", "exclude obstetric/delivery admissions", young & medicaid & ~OB, AMA)
prop("M01", "men only", young & medicaid & (A.gender == "M"), AMA)
prop("M01", "women only, non-obstetric", young & medicaid & (A.gender == "F") & ~OB, AMA)
prop("M01", "emergency/urgent/observation admissions only", young & medicaid & EMERG, AMA)
prop("M01", "elective admissions only", young & medicaid & (A.admission_type == "ELECTIVE"), AMA)
prop("M01", "exclude missing discharge_location from denominator", young & medicaid & A.discharge_location.notna(), AMA)
prop("M01", "exclude in-hospital deaths", young & medicaid & (A.hospital_expire_flag == 0), AMA)
prop("M01", "with substance-use code (F10-F19 / 291-292,303-305)", young & medicaid &
     dxset(regex9=r"^(29[12]|30[345])", regex10=r"^F1\d"), AMA)
pat_any = A[young & medicaid].groupby("subject_id").discharge_location.apply(lambda s: (s == "AGAINST ADVICE").any())
k, n = int(pat_any.sum()), len(pat_any)
print(f"M01 | alt      | {'per PATIENT: any AMA discharge among Medicaid 18-39':72} "
      + (f"{100*k/n:6.2f}%  n_pat={n}" if n and k >= MIN_CELL and n - k >= MIN_CELL else "SUPPRESSED/EMPTY"))
out["estimates"].append(dict(question="M01", check="alt", definition="per patient: any AMA",
                             estimate=100 * k / n if n and k >= MIN_CELL and n - k >= MIN_CELL else None, n_patients=n))

# ======================= M41: IV methylprednisolone among asthma admissions aged 18-49 =======================
asth = A.age.between(18, 49) & dxset(["493"], ["J45"])
IV = route_u.str.contains("IV")
MP = "methylpred|medrol"
listing("M41 drug strings matching methylpred/medrol/solu (all prescriptions)",
        rx.drug[drug_l.str.contains(r"methylpred|medrol|solu", regex=True)])
listing("M41 routes of methylprednisolone prescriptions", rx.route[drug_l.str.contains(MP, regex=True)])
listing("M41 systemic steroid drug strings (prednis|hydrocort|dexameth)",
        rx.drug[drug_l.str.contains(r"prednis|hydrocort|dexameth", regex=True)], top=30)
pos = dx.icd_code.str.startswith(("493", "J45"))
listing("M41 asthma code positions (seq_num 1 = principal, all ages)", dx.loc[pos, "seq_num"].clip(upper=6).astype(str).replace("6", "6+"))
anyrx = A.hadm_id.isin(set(rx.hadm_id.dropna().astype(int)))
prop("M41", "control: admission has any prescription at all", asth, anyrx, "control")
prop("M41", "control: asthma is the principal diagnosis", asth, dxset(["493"], ["J45"], principal=True), "control")
prop("M41", "control: any IV methylpred, ALL adult admissions", A.age >= 18, rxset(MP, IV), "control")
prop("M41", "as specified ('methylprednisolone', route contains IV)", asth, rxset("methylprednisolone", IV), "truth")
prop("M41", "methylpred or medrol, route contains IV", asth, rxset(MP, IV))
prop("M41", "methylpred, any route", asth, rxset(MP))
prop("M41", "any systemic steroid (methylpred/prednisone/prednisolone/hydrocort/dexameth), any route", asth,
     rxset(r"methylpred|medrol|prednis|hydrocort|dexameth", ~route_u.str.contains("INHAL|NEB|TOP|OPHTH|OTIC|NU|PR|EX")))
prop("M41", "restrict to admissions with any prescription", asth & anyrx, rxset("methylprednisolone", IV))
EXAC = dxset(regex9=r"^493\d[12]", regex10=r"^J45(\.?\d[12]|9[01][12])")
PRIN = dxset(["493"], ["J45"], principal=True)
prop("M41", "asthma PRINCIPAL diagnosis", A.age.between(18, 49) & PRIN, rxset("methylprednisolone", IV))
prop("M41", "asthma exacerbation / status asthmaticus codes", A.age.between(18, 49) & EXAC, rxset("methylprednisolone", IV))
prop("M41", "asthma principal, any systemic steroid any route", A.age.between(18, 49) & PRIN,
     rxset(r"methylpred|medrol|prednis|hydrocort|dexameth", ~route_u.str.contains("INHAL|NEB|TOP|OPHTH|OTIC|NU|PR|EX")))
prop("M41", "ICD-10 era only (J45)", A.age.between(18, 49) & dxset([], ["J45"]), rxset("methylprednisolone", IV), "stratum")
prop("M41", "ICD-9 era only (493)", A.age.between(18, 49) & dxset(["493"], []), rxset("methylprednisolone", IV), "stratum")

json.dump(out, open(OUT, "w"), indent=2, default=str)
print(f"\nSaved aggregates to {OUT}")
