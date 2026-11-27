"""Interval informativeness: coverage with 95% CI, normalised interval width and interval score (no new API calls).

Width = (hi90 - lo90) / E_q, where E_q is the question's reference error (reference_errors.json), so widths are
comparable across questions; reported as the median over questions of the per-question median over runs.
Interval score (Winkler / Gneiting-Raftery, alpha = 0.1) = width + (2/alpha) * distance of the truth outside the
interval. A configuration's interval score ratio = 2 ** mean over questions of log2(score / reference score), with the
reference score = median score of the 19 current configurations on that question (1 = typical, lower is better).
95% CIs: bootstrap over questions (1,000 resamples). Writes figures/interval_quality.csv.
Usage (from the repository root): python3 -I scripts/relative_error.py && python3 -I scripts/interval_quality.py
"""
import json, glob, os, re, csv
from collections import defaultdict
import numpy as np

DS = {"NHANES": ("nhanes/nhanes_ground_truth.json", "nhanes/results"), "MIMIC-IV": ("mimic/mimic_ground_truth.json", "mimic/results"),
      "All of Us": ("aou/aou_ground_truth.json", "aou/results")}
REF = json.load(open("reference_errors.json"))
EARLIER = {"opus46_api", "opus48_api", "opus5_api", "gpt41_api", "gpt5_api", "gpt55_api", "gem3_flash_api", "gem31_pro_api"}
DROP = {"gpt62_astra_health_api", "gpt62_sol_health_api", "gpt4o_api", "gem25_pro_api"}
ALPHA = 0.1
runs = defaultdict(lambda: defaultdict(list))  # tag -> (ds, q) -> [(covered, width/E, score/E)]
for ds, (g, d) in DS.items():
    gt = {k: v["estimate"] for k, v in json.load(open(g)).items() if isinstance(v, dict) and "estimate" in v}
    for f in glob.glob(f"{d}/predictions_*_api_r[0-9].json"):
        tag = re.match(r"predictions_(.+)_r\d\.json", os.path.basename(f)).group(1)
        if tag in DROP:
            continue
        for x in json.load(open(f)):
            y, lo, hi, E = gt[x["id"]], x["lo90"], x["hi90"], REF[f"{ds}|{x['id']}"]
            s = (hi - lo) + 2 / ALPHA * (max(lo - y, 0) + max(y - hi, 0))
            runs[tag][(ds, x["id"])].append((lo <= y <= hi, (hi - lo) / E, s / E))
qs = list(runs["fable_api"])
ref_s = {q: np.median([v[2] for t in runs if t not in EARLIER for v in runs[t][q]]) for q in qs}
rng = np.random.default_rng(0)
out = []
for t, d in runs.items():
    cov = np.array([np.mean([v[0] for v in d[q]]) for q in qs])
    wid = np.array([np.median([v[1] for v in d[q]]) for q in qs])
    sc = np.array([np.mean([np.log2(max(v[2], 1e-12) / ref_s[q]) for v in d[q]]) for q in qs])
    idx = [rng.integers(0, len(qs), len(qs)) for _ in range(1000)]
    out.append(dict(tag=t, coverage=100 * cov.mean(), cov_lo=np.percentile([100 * cov[i].mean() for i in idx], 2.5),
                    cov_hi=np.percentile([100 * cov[i].mean() for i in idx], 97.5), width=np.median(wid),
                    score=2 ** sc.mean(), score_lo=np.percentile([2 ** sc[i].mean() for i in idx], 2.5),
                    score_hi=np.percentile([2 ** sc[i].mean() for i in idx], 97.5)))
out.sort(key=lambda r: r["score"])
with open("figures/interval_quality.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader()
    for r in out:
        w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
for r in out:
    print(f'{r["tag"]:24} cov {r["coverage"]:5.1f} ({r["cov_lo"]:.0f}-{r["cov_hi"]:.0f})  width {r["width"]:.2f}  '
          f'score {r["score"]:.2f} ({r["score_lo"]:.2f}-{r["score_hi"]:.2f})')
cur = [r for r in out if r["tag"] not in EARLIER]
fl = [r for r in cur if r["tag"] not in {"gem38_flash_api", "sonnet_api", "haiku_api", "gpt6_luna_api"}]
print("current flagship widths:", round(min(r["width"] for r in fl), 2), "-", round(max(r["width"] for r in fl), 2),
      "| small:", [round(r["width"], 2) for r in cur if r not in fl])
