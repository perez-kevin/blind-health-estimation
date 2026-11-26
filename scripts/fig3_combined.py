"""Figure 3: (a) error ratio vs API cost, (b) error ratio by release date, stacked into one image.
Usage (from Bench/): run api_plots.py and release_date_plot.py first; then python3 -I scripts/fig3_combined.py
"""
from PIL import Image, ImageDraw, ImageFont

a, b = Image.open("figures/api_cost.png").convert("RGB"), Image.open("figures/fig_release_date.png").convert("RGB")
W = max(a.width, b.width)
a = a.resize((W, round(a.height * W / a.width)), Image.LANCZOS)
b = b.resize((W, round(b.height * W / b.width)), Image.LANCZOS)
gap = 40
out = Image.new("RGB", (W, a.height + gap + b.height), (252, 252, 251))
out.paste(a, (0, 0)); out.paste(b, (0, a.height + gap))
d = ImageDraw.Draw(out)
try:
    f = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 80)
except OSError:
    f = ImageFont.load_default()
d.text((40, 40), "a", font=f, fill=(11, 11, 11)); d.text((40, a.height + gap + 40), "b", font=f, fill=(11, 11, 11))
out.save("figures/fig3_cost_release.png", dpi=(220, 220)); print(out.size)
