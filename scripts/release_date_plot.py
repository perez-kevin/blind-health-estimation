"""Error ratio by model release date: progression from 2025 to the current models.

Usage (from the repository root): python3 -I scripts/relative_error.py && python3 -I scripts/api_plots.py && python3 -I scripts/release_date_plot.py
  -> figures/fig_release_date.png
Release dates: Anthropic and OpenAI from their models APIs (created date); Gemini from the Gemini API changelog
(first API availability). Each model is shown at its best reasoning-effort setting.
"""
import csv, datetime as dt
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
from matplotlib.lines import Line2D

SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8b8a84", "#e9e8e4"
COL = {"claude": "#2a78d6", "openai": "#1baf7a", "gemini": "#eb6834"}
PROV = {"claude": "Anthropic Claude", "openai": "OpenAI GPT", "gemini": "Google Gemini"}
plt.rcParams.update({"font.family": "sans-serif", "font.size": 11, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "savefig.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": GRID, "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2})
RELEASE = {  # model family -> release date
    "GPT-4.1": "2025-04-10", "GPT-5": "2025-08-05", "Gemini 3 Flash": "2025-12-17", "Opus 4.6": "2026-02-04",
    "Gemini 3.1 Pro": "2026-02-19", "GPT-5.5": "2026-04-22", "Opus 4.8": "2026-05-28", "Opus 5": "2026-07-24",
    "Gemini 3.8 Flash": "2026-09-02", "GPT-6 luna": "2026-09-14", "Opus 5.5": "2026-09-21", "Sonnet 5.5": "2026-09-28",
    "Haiku 5.5": "2026-10-07", "Gemini 4 argon": "2026-10-13", "GPT-6.2 astra": "2026-10-31", "GPT-6.2 sol": "2026-11-02",
    "Fable 5.2": "2026-11-09", "Gemini 4 carbon": "2026-11-10", "Gemini 4 hydrogen": "2026-11-12",
    "Gemini 4 neon": "2026-11-18"}
SMALL = {"Sonnet 5.5", "Haiku 5.5", "GPT-6 luna", "Gemini 3.8 Flash", "Gemini 3 Flash"}
LABEL = {  # label offsets (points) for the models named on the chart
    "GPT-4.1": (0, 14), "GPT-5": (0, -14), "Gemini 3 Flash": (0, -14), "Opus 4.6": (0, 14),
    "Gemini 3.1 Pro": (0, -14), "GPT-5.5": (0, 14), "Opus 4.8": (0, -14), "Opus 5": (0, 14),
    "Fable 5.2": (-10, 0), "GPT-6.2 astra": (-10, 0), "Gemini 4 neon": (8, 0)}

best = {}
for r in csv.DictReader(open("figures/api_results.csv")):
    base = r["model"].split(" · ")[0].split(" (")[0]
    q = float(r["error_ratio"])
    if base not in best or q < best[base][0]:
        best[base] = (q, r["provider"])

fig, ax = plt.subplots(figsize=(8.5, 6.2))  # same shape as figures/api_cost.png (Figure 3a)
ax.set_yscale("log")
ax.axvspan(dt.date(2026, 9, 1), dt.date(2026, 11, 22), color="#f1f0ec", zorder=0)
ax.text(dt.date(2026, 9, 4), 0.505, "current models\n(reference panel)", fontsize=8.5, color=MUTED, va="top")
for p, c in COL.items():  # vendor frontier: best error ratio available to date (flagship tier)
    pts = sorted((dt.date.fromisoformat(RELEASE[b]), q) for b, (q, pr) in best.items() if pr == p and b not in SMALL)
    xs, ys, cur = [], [], np.inf
    for d, q in pts:
        cur = min(cur, q); xs.append(d); ys.append(cur)
    ax.step(xs + [dt.date(2026, 11, 22)], ys + [ys[-1]], where="post", color=c, lw=1.4, alpha=0.5, zorder=2)
for b, (q, p) in best.items():
    d = dt.date.fromisoformat(RELEASE[b])
    ax.scatter(d, q, s=46 if b in SMALL else 80, marker="s" if b in SMALL else "o", color=COL[p], edgecolor=SURF,
               linewidth=1.5, zorder=3)
    if b in LABEL:
        dx, dy = LABEL[b]
        ax.annotate(b, (d, q), xytext=(dx, dy), textcoords="offset points", fontsize=8.8, color=INK2, va="center",
                    ha="right" if dx < 0 else ("left" if dx > 0 else "center"))
ax.axhline(1, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
ax.text(dt.date(2025, 3, 20), 1.02, "typical error of current models", fontsize=8.5, color=MUTED, va="top")
ax.set_ylim(1.75, 0.48)  # same range as Figure 3a  # inverted: better (lower error) is up
ax.set_yticks([0.5, 0.7, 1, 1.4], ["0.5", "0.7", "1", "1.4"]); ax.yaxis.set_minor_locator(mtick.NullLocator())
ax.set_xlim(dt.date(2025, 3, 1), dt.date(2027, 1, 31))
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10])); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.grid(color=GRID, lw=0.8, zorder=0)
ax.set_xlabel("Model release date"); ax.set_ylabel("Error ratio (log scale; lower is better, shown higher)")
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=c, markersize=9, label=PROV[p]) for p, c in COL.items()] +
          [Line2D([], [], marker="s", ls="", color=INK2, markersize=7, label="small / low-cost model"),
           Line2D([], [], color=INK2, lw=1.4, alpha=0.5, label="best flagship to date")],
          frameon=False, loc="upper left", fontsize=8.5)
fig.text(0.09, 0.95, "Accuracy by release date", fontsize=15, fontweight="bold", color=INK)
fig.text(0.09, 0.9, "Best effort setting per model · earlier models at default effort", fontsize=9.5, color=INK2)
fig.subplots_adjust(left=0.1, right=0.97, top=0.86, bottom=0.1)
fig.savefig("figures/fig_release_date.png", dpi=220)
fl = [(dt.date.fromisoformat(RELEASE[b]).toordinal(), q) for b, (q, _) in best.items() if b not in SMALL]
x, y = zip(*fl); rk = lambda v: np.argsort(np.argsort(v))
print("flagship models:", len(fl), "Spearman(date, best error ratio) =", round(float(np.corrcoef(rk(x), rk(y))[0, 1]), 2))
for b in sorted(best, key=lambda b: RELEASE[b]): print(RELEASE[b], b, best[b][0])
