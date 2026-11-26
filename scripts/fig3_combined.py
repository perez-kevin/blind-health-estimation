"""Figure 3: (a) error ratio vs API cost, (b) error ratio by release date, side by side in one image.
Usage (from Bench/): run api_plots.py and release_date_plot.py first; then python3 -I scripts/fig3_combined.py
"""
from PIL import Image, ImageDraw, ImageFont

a, b = Image.open("figures/api_cost.png").convert("RGB"), Image.open("figures/fig_release_date.png").convert("RGB")
H = max(a.height, b.height)
a = a.resize((round(a.width * H / a.height), H), Image.LANCZOS)
b = b.resize((round(b.width * H / b.height), H), Image.LANCZOS)
gap = 30
out = Image.new("RGB", (a.width + gap + b.width, H), (252, 252, 251))
out.paste(a, (0, 0)); out.paste(b, (a.width + gap, 0))
d = ImageDraw.Draw(out)
try:
    f = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 70)
except OSError:
    f = ImageFont.load_default()
d.text((25, 25), "a", font=f, fill=(11, 11, 11)); d.text((a.width + gap + 25, 25), "b", font=f, fill=(11, 11, 11))
out.save("figures/fig3_cost_release.png", dpi=(220, 220)); print(out.size)
