"""Supplementary Data 1 (Excel): model results, question difficulty, README.
Usage (from Bench/): run relative_error.py, api_plots.py, api_suppl_plots.py first; then python3 -I scripts/build_suppl_data.py
"""
import csv
import openpyxl
from openpyxl.styles import Font

PROV = {"claude": "Anthropic", "openai": "OpenAI", "gemini": "Google"}
res = list(csv.DictReader(open("figures/api_results.csv")))
qn = {r["tag"]: r for r in csv.DictReader(open("figures/relative_error.csv"))}
name2tag = dict((v, k) for k, v in eval(open("scripts/api_plots.py").read().split("\nNAME = ", 1)[1].split("\nEARLIER")[0]).items())
wb = openpyxl.Workbook(); rd = wb.active; rd.title = "README"
rd.append(["Sheet", "Contents"])
rd.append(["Model results", "One row per model configuration, ordered by error ratio. Mean of 3 API runs per dataset (150 "
           "questions). Generation: current = the 19 configurations released September-November 2026, which define the "
           "reference errors; earlier = models released April 2025-July 2026 at default effort, scored on the same scale. "
           "Cost is given for current models only (list prices on 25 November 2026)."])
rd.append(["Question difficulty", "One row per question, computed over the 19 current configurations (57 runs per question)."])
rd.append(["Error ratio", "2 ** mean over runs and questions of log2(max(|error|, 0.05*ref) / ref); ref = median absolute "
           "error of the current configurations on that question. 1 = typical, lower is better."])
rd.append(["Within 10%", "Share of questions with |estimate - truth| <= 0.10 * |truth|."])
rd.append(["90% interval coverage", "Share of questions whose true value lies inside the model's 90% interval; 95% CI by bootstrap over questions."])
rd.append(["Median interval width", "Median over questions of (hi90 - lo90) / reference error of the question."])
rd.append(["Interval score ratio", "Winkler interval score at alpha = 0.1 (width + 20 x distance of the truth outside the interval), "
           "as 2 ** mean log2(score / median score of the current configurations on that question). 1 = typical, lower is better."])
ws = wb.create_sheet("Model results")
ws.append(["Model (reasoning effort)", "Vendor", "Generation", "Error ratio", "95% CI low", "95% CI high", "Error ratio: NHANES",
           "Error ratio: MIMIC-IV", "Error ratio: All of Us", "Within 10% (%): NHANES", "Within 10% (%): MIMIC-IV",
           "Within 10% (%): All of Us", "Within 10% (%): average", "Run range low", "Run range high",
           "Median relative error (%)", "90% interval coverage (%)", "Coverage 95% CI low", "Coverage 95% CI high",
           "Median interval width (x reference error)", "Interval score ratio", "Interval score 95% CI low",
           "Interval score 95% CI high", "Cost per question (USD)"])
iq = {r["tag"]: r for r in csv.DictReader(open("figures/interval_quality.csv"))}  # scripts/interval_quality.py
f = lambda x: None if x in ("nan", "") else float(x)
for r in res:
    q = qn[name2tag[r["model"]]]; early = "(" in r["model"]
    ws.append([r["model"], PROV[r["provider"]], "earlier" if early else "current", f(r["error_ratio"]), f(r["error_ratio_ci_lo"]),
               f(r["error_ratio_ci_hi"]), round(float(q["nhanes"]), 2), round(float(q["mimic"]), 2), round(float(q["aou"]), 2),
               f(r["nhanes_within10"]), f(r["mimic_within10"]), f(r["aou_within10"]), f(r["avg_within10"]), f(r["repeat_min"]),
               f(r["repeat_max"]), f(r["median_rel_err"]), f(r["coverage90"]), *[round(float(iq[name2tag[r["model"]]][k]), 2)
               for k in ("cov_lo", "cov_hi", "width", "score", "score_lo", "score_hi")],
               None if early else f(r["cost_per_question_usd"])])
ws2 = wb.create_sheet("Question difficulty")
ws2.append(["Dataset", "Question ID", "Unit", "True value", "Share of runs within 10%",
            "Share of runs with truth inside 90% interval", "Median prediction / truth", "Runs", "Question text"])
for r in csv.DictReader(open("figures/question_difficulty.csv")):
    ws2.append([r["dataset"], r["question"], r["unit"], float(r["truth"]), float(r["share_runs_within10"]),
                float(r["share_runs_truth_in_90"]), f(r["median_pred_over_truth"]), int(r["n_runs"]), r["text"]])
for s in wb:
    for c in s[1]: c.font = Font(bold=True)
wb.save("Supplementary_Data_1.xlsx"); print("rows:", ws.max_row - 1, ws2.max_row - 1)
