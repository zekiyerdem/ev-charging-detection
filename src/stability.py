"""How much do the results depend on one clustering run? And a steadier EV decision.

detect.py and compare_baselines.py use a single setting: k = 7 clusters, random seed 0.
The baseline comparison showed results that swing a lot between K-Means and GMM
(p20: 30/30 every-night chargers with K-Means, 0/30 with the GMM), so one run is not
enough to pick a method.

For every baseline in features.BASELINES this script:
  1. builds the features once,
  2. repeats the clustering over k = 5..9 and seeds 0..4 (25 settings) with both
     K-Means and the GMM (50 runs), naming clusters with the same rules as detect.py,
  3. reports the spread of the results (min / median / max F1, every-night chargers
     found, homes flagged),
  4. gives each home an **EV score** = share of the runs that flag it, and flags
     the home when the score is at least 0.5 (a majority vote). The vote is done over
     all 50 runs, and over the 25 K-Means runs and the 25 GMM runs separately.

The EV score does not depend on one lucky or unlucky run. Labels are used only for scoring.

Writes reports/stability_runs.csv (every run), reports/stability_summary.csv (spread),
reports/stability_vote.csv (majority-vote result per baseline) and
reports/stability_home_scores.csv (EV score of every home for every baseline,
from all 50 runs).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from detect import FEATURES_FOR_CLUSTERING, cluster_homes, evaluate
from features import BASELINES, build_features

KS = range(5, 10)
SEEDS = range(5)


def scores_for(X: pd.DataFrame, ks=KS, seeds=SEEDS):
    """Run every (k, seed) setting with K-Means and GMM.

    Returns (runs, votes): runs is a list of (k, seed, algorithm, predictions) and
    votes is each home's share of runs that flagged it as an EV home."""
    runs = []
    for k in ks:
        for seed in seeds:
            c = cluster_homes(X, k, seed)
            runs.append((k, seed, "K-Means", c["pred_km"]))
            runs.append((k, seed, "GMM", c["pred_gmm"]))
    votes = np.mean([pred for *_, pred in runs], axis=0)
    return runs, pd.Series(votes, index=X.index, name="ev_score")


def describe(y, every_night, pred):
    """Scores for one set of predictions, including the every-night chargers found."""
    r = evaluate("", y, pred)
    r.pop("method")
    r["every_night_found"] = int(pred[every_night].sum())
    r["false_alarms"] = int(((pred == 1) & (y == 0)).sum())
    return r


def main(data="data", reports="reports", threshold=0.5):
    data, reports = Path(data), Path(reports)
    readings = pd.read_csv(data / "meter_readings.csv.gz")
    weather = pd.read_csv(data / "weather.csv")
    homes = pd.read_csv(data / "homes.csv").set_index("home_id")

    run_rows, vote_rows, home_scores = [], [], {}
    for baseline in BASELINES:
        X = build_features(readings, weather, baseline)[FEATURES_FOR_CLUSTERING]
        h = homes.loc[X.index]
        y = h["has_ev"].to_numpy()
        every_night = (h["charging_pattern"].fillna("") == "every_night").to_numpy()

        runs, score = scores_for(X)
        for k, seed, algo, pred in runs:
            run_rows.append({"baseline": baseline, "algorithm": algo, "k": k, "seed": seed,
                             **describe(y, every_night, pred)})

        for used in ["both", "K-Means", "GMM"]:
            preds = [pred for _, _, algo, pred in runs if used in ("both", algo)]
            score_u = pd.Series(np.mean(preds, axis=0), index=X.index)
            vote = (score_u.to_numpy() >= threshold).astype(int)
            v = {"baseline": baseline, "runs_used": used, "runs": len(preds),
                 **describe(y, every_night, vote),
                 "average_precision_of_score": round(average_precision_score(y, score_u), 3),
                 "homes_always_flagged": int((score_u == 1).sum()),
                 "homes_never_flagged": int((score_u == 0).sum()),
                 "homes_in_between": int(((score_u > 0) & (score_u < 1)).sum())}
            vote_rows.append(v)
            if used == "both":
                print(f"{baseline:13s} done: vote F1 {v['f1']:.3f}, "
                      f"every-night {v['every_night_found']}/{every_night.sum()}")
        home_scores[baseline] = score.round(3)

    runs = pd.DataFrame(run_rows)
    runs.to_csv(reports / "stability_runs.csv", index=False)

    n_every = int((homes["charging_pattern"].fillna("") == "every_night").sum())
    g = runs.groupby(["baseline", "algorithm"], sort=False)
    summary = pd.DataFrame({
        "runs": g.size(),
        "f1_min": g["f1"].min(), "f1_median": g["f1"].median(), "f1_max": g["f1"].max(),
        "every_night_min": g["every_night_found"].min(),
        "every_night_median": g["every_night_found"].median(),
        "every_night_max": g["every_night_found"].max(),
        "flagged_min": g["estimated_ev_homes"].min(),
        "flagged_median": g["estimated_ev_homes"].median(),
        "flagged_max": g["estimated_ev_homes"].max(),
    }).reset_index()
    summary.to_csv(reports / "stability_summary.csv", index=False)

    vote = pd.DataFrame(vote_rows)
    vote.to_csv(reports / "stability_vote.csv", index=False)

    scores = pd.DataFrame(home_scores)
    scores.index.name = "home_id"
    scores.to_csv(reports / "stability_home_scores.csv")

    print(f"\nSpread over {len(KS) * len(SEEDS)} settings (k = {KS.start}..{KS.stop - 1}, "
          f"seeds {SEEDS.start}..{SEEDS.stop - 1}); every-night chargers out of {n_every}:\n"
          + summary.to_string(index=False))
    print(f"\nMajority vote (EV score >= {threshold}):\n"
          + vote.to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data"); p.add_argument("--reports", default="reports")
    p.add_argument("--threshold", type=float, default=0.5, help="EV score needed to flag a home")
    a = p.parse_args()
    main(a.data, a.reports, a.threshold)
