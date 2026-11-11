"""Baseline checkpoint: How much should an NFL blowout count?

Rolling-origin validation, seasons 2008-2014 (train 2002..s-1, validate s).
Test seasons 2015-2022 are built (for counts only) but never scored.
"""
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

SEED = 2026
CAP = 14
LAMBDAS = [0.01, 0.1, 1, 10]
VAL_SEASONS = list(range(2008, 2015))
N_BOOT = 2000

path = sys.argv[1] if len(sys.argv) > 1 else "games.csv"
g = pd.read_csv(path)
g = g[(g.game_type == "REG") & g.season.between(2002, 2022)].copy()
g["kick"] = pd.to_datetime(g.gameday + " " + g.gametime.fillna("13:00"))
g = g.sort_values(["kick", "game_id"]).reset_index(drop=True)
print("regular-season games 2002-2022:", len(g))

# Long format: one row per team per game.
home = pd.DataFrame({"game_id": g.game_id, "season": g.season, "kick": g.kick,
                     "team": g.home_team, "margin": g.home_score - g.away_score})
away = pd.DataFrame({"game_id": g.game_id, "season": g.season, "kick": g.kick,
                     "team": g.away_team, "margin": g.away_score - g.home_score})
long = pd.concat([home, away]).sort_values(["team", "season", "kick"])
long["win"] = np.where(long.margin > 0, 1.0, np.where(long.margin == 0, 0.5, 0.0))
long["cmargin"] = long.margin.clip(-CAP, CAP)
grp = long.groupby(["team", "season"])
# shift(1) => only games strictly before this kickoff
long["n_prior"] = grp.cumcount()
for col, out in [("win", "wp4"), ("margin", "m4"), ("cmargin", "cm4")]:
    long[out] = grp[col].transform(lambda s: s.shift(1).rolling(4).mean())
feat = long.set_index(["game_id", "team"])[["n_prior", "wp4", "m4", "cm4"]]

def side(df, team_col, tag):
    f = feat.loc[list(zip(df.game_id, df[team_col]))].reset_index(drop=True)
    f.columns = [f"{tag}_{c}" for c in f.columns]
    return f

d = pd.concat([g.reset_index(drop=True), side(g, "home_team", "h"),
               side(g, "away_team", "a")], axis=1)
d["y"] = (d.home_score > d.away_score).astype(int)  # tie -> 0
d["rec"] = d.h_wp4 - d.a_wp4
d["mar"] = d.h_m4 - d.a_m4
d["cmar"] = d.h_cm4 - d.a_cm4
elig = d[(d.location != "Neutral") & (d.h_n_prior >= 4) & (d.a_n_prior >= 4)].copy()
print("eligible games:", len(elig),
      "| test 2015-2022:", (elig.season >= 2015).sum(),
      "| validation 2008-2014:", elig.season.between(2008, 2014).sum())
elig.to_csv("eligible_games.csv", index=False)

MODELS = {"record": ["rec"], "record+margin": ["rec", "mar"],
          "record+capped": ["rec", "cmar"]}

def fit_predict(tr, va, cols, lam):
    mu, sd = tr[cols].mean(), tr[cols].std(ddof=0)  # training-fold stats only
    Xtr, Xva = (tr[cols] - mu) / sd, (va[cols] - mu) / sd
    # objective: mean log loss + lam * ||w||^2 (intercept unpenalized)
    m = LogisticRegression(C=1.0 / (2 * len(tr) * lam), max_iter=1000)
    m.fit(Xtr, tr.y)
    return m.predict_proba(Xva)[:, 1]

# Out-of-fold predictions for every model x lambda, plus home-rate reference.
oof = []
for s in VAL_SEASONS:
    tr = elig[elig.season.between(2002, s - 1)]
    va = elig[elig.season == s]
    row = pd.DataFrame({"season": s, "y": va.y.values})
    row["p_homerate"] = tr.y.mean()
    for name, cols in MODELS.items():
        for lam in LAMBDAS:
            row[f"p_{name}_{lam}"] = fit_predict(tr, va, cols, lam)
    oof.append(row)
oof = pd.concat(oof, ignore_index=True)

def brier(p, y):
    return (p - y) ** 2

print("\nPooled validation Brier by lambda:")
chosen = {}
for name in MODELS:
    scores = {lam: brier(oof[f"p_{name}_{lam}"], oof.y).mean() for lam in LAMBDAS}
    chosen[name] = min(scores, key=scores.get)
    print(f"  {name:14s}", {k: round(v, 5) for k, v in scores.items()},
          "-> chosen", chosen[name])

B = pd.DataFrame({"season": oof.season, "homerate": brier(oof.p_homerate, oof.y)})
for name in MODELS:
    B[name] = brier(oof[f"p_{name}_{chosen[name]}"], oof.y)
B["d_capped_vs_uncapped"] = B["record+margin"] - B["record+capped"]  # >0 favors cap
B["d_margin_vs_record"] = B["record"] - B["record+margin"]
B["d_capped_vs_record"] = B["record"] - B["record+capped"]
B["d_record_vs_homerate"] = B["homerate"] - B["record"]

per_season = B.groupby("season").mean()
per_season["n"] = B.groupby("season").size()
print("\nPer-season validation Brier:")
print(per_season.round(4).to_string())

rng = np.random.default_rng(SEED)
seasons = np.array(VAL_SEASONS)
by_season = {s: B[B.season == s] for s in seasons}
cols = ["homerate", "record", "record+margin", "record+capped",
        "d_capped_vs_uncapped", "d_margin_vs_record", "d_capped_vs_record",
        "d_record_vs_homerate"]
sums = {s: by_season[s][cols].sum() for s in seasons}
ns = {s: len(by_season[s]) for s in seasons}
arrs = {s: by_season[s][cols].to_numpy() for s in seasons}

whole, strat = [], []
for _ in range(N_BOOT):
    pick = rng.choice(seasons, size=len(seasons), replace=True)
    whole.append(sum(sums[s] for s in pick) / sum(ns[s] for s in pick))
    tot = np.zeros(len(cols))
    for s in seasons:
        idx = rng.integers(0, ns[s], ns[s])
        tot += arrs[s][idx].sum(0)
    strat.append(tot / len(B))
whole = pd.DataFrame(whole)
strat = pd.DataFrame(strat, columns=cols)

summary = pd.DataFrame({
    "mean": B[cols].mean(),
    "season_sd": per_season[cols].std(),
    "whole_lo": whole.quantile(0.025), "whole_hi": whole.quantile(0.975),
    "strat_lo": strat.quantile(0.025), "strat_hi": strat.quantile(0.975),
})
print("\nPooled validation summary (95% bootstrap intervals, 2000 reps, seed 2026):")
print(summary.round(5).to_string())
summary.to_csv("summary.csv")
per_season.to_csv("per_season.csv")
B.to_csv("val_brier_by_game.csv", index=False)

# Spot check: 2007 NE @ IND
ex = d[(d.season == 2007) & (d.home_team == "IND") & (d.away_team == "NE")].iloc[0]
print("\n2007 NE@IND: NE avg margin", ex.a_m4, "capped", ex.a_cm4, "NE wp4", ex.a_wp4)
