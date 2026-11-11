import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = pd.read_csv("summary.csv", index_col=0)
P = pd.read_csv("per_season.csv", index_col=0)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

def flip(row):  # interval for the negated quantity
    r = S.loc[row]
    return -r["mean"], (-r.whole_hi, -r.whole_lo), (-r.strat_hi, -r.strat_lo)

rows = [("Record only", *flip("d_margin_vs_record")),
        ("Uncapped margin", 0.0, (0, 0), (0, 0)),
        ("Capped margin (±14)", S.loc["d_capped_vs_uncapped", "mean"],
         (S.loc["d_capped_vs_uncapped", "whole_lo"], S.loc["d_capped_vs_uncapped", "whole_hi"]),
         (S.loc["d_capped_vs_uncapped", "strat_lo"], S.loc["d_capped_vs_uncapped", "strat_hi"]))]

fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.8), facecolor=SURF,
                           gridspec_kw={"width_ratios": [1, 1.25]})
for ax in (a, b):
    ax.set_facecolor(SURF)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]: ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.axhline(0, color=INK2, lw=1)

for i, (lab, m, w, s) in enumerate(rows):
    if lab == "Uncapped margin":
        a.plot(i, 0, "o", color=INK2, ms=8, mec=SURF, mew=2)
        a.annotate("reference (0 by definition)", (i, 0), xytext=(0, 10), textcoords="offset points",
                   ha="center", fontsize=8, color=INK2)
        continue
    a.errorbar(i - 0.08, m, yerr=[[m - w[0]], [w[1] - m]], fmt="o", color=BLUE, ms=8,
               mec=SURF, mew=2, lw=2, capsize=4, label="Whole-season bootstrap" if i == 0 else None)
    a.errorbar(i + 0.08, m, yerr=[[m - s[0]], [s[1] - m]], fmt="s", color=ORANGE, ms=8,
               mec=SURF, mew=2, lw=2, capsize=4, label="Games stratified by season" if i == 0 else None)
    a.annotate(f"{m:+.4f}", (i + 0.2, m), va="center", fontsize=9, color=INK)
a.set_xticks(range(3)); a.set_xticklabels([r[0] for r in rows], color=INK, fontsize=9)
a.set_xlim(-0.5, 2.6)
a.set_ylabel("Uncapped Brier − model Brier  (positive favors model)", color=INK2, fontsize=9)
a.set_title("Pooled validation, 2008–2014 (n = 1,322), 95% intervals",
            loc="left", fontsize=10, color=INK)
a.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")

x = P.index.values
b.plot(x, P["d_margin_vs_record"], "-o", color=BLUE, lw=2, ms=8, mec=SURF, mew=2,
       label="Uncapped margin vs record only")
b.plot(x, P["d_capped_vs_uncapped"], "-o", color=AQUA, lw=2, ms=8, mec=SURF, mew=2,
       label="Capped vs uncapped margin")
b.set_xticks(x)
b.set_ylabel("Paired Brier difference  (positive favors 2nd-named… first-named model)",
             color=INK2, fontsize=9)
b.set_ylabel("Brier improvement of first-named model", color=INK2, fontsize=9)
b.set_title("By validation season", loc="left", fontsize=10, color=INK)
b.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")
fig.suptitle("Baseline checkpoint: margin models vs record-only ridge logistic regression "
             "(test seasons 2015–2022 untouched)", x=0.01, ha="left", fontsize=11, color=INK)
fig.tight_layout()
fig.savefig("baseline.png", dpi=160, facecolor=SURF)
