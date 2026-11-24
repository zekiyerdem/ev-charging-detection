"""HDBSCAN (no k to choose) compared with the GMM.

K-Means and the GMM both need the number of clusters k. The GMM in particular swings a
lot with k and the seed (src/stability.py). HDBSCAN is a density-based method: it finds
as many clusters as the data supports and leaves homes that fit no dense group as
"noise" (label -1). Its main setting is min_cluster_size (the smallest group that counts
as a cluster), so instead of k we try several values of that.

For both typical-load baselines (the original `median` and the default `p20_edges`) this
script runs on the same standardised features as detect.py:
  * HDBSCAN with min_cluster_size = 5, 10, 15, 20, 30, 40,
  * one GMM run with k = 7, seed 0 (the original setup),
  * the GMM over k = 5..9 and seeds 0..4 (25 runs): spread and majority vote.
Clusters are named with the same physical rules as detect.py (label_cluster); noise
homes are never flagged as EV. Each row also shows the result after the charger power
check from detect.py, so it can be set next to the default decision.

Labels are used ONLY to score. Writes reports/hdbscan_comparison.csv and
reports/hdbscan_by_home_type.csv.
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler

from detect import FEATURES_FOR_CLUSTERING, cluster_homes, evaluate, home_type, pick_ev_clusters, power_check
from features import build_features

MIN_CLUSTER_SIZES = [5, 10, 15, 20, 30, 40]

# scikit-learn 1.9 warns that the default of HDBSCAN(copy=...) will change; it does not matter here
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.cluster._hdbscan")
KS = range(5, 10)
SEEDS = range(5)


def hdbscan_homes(X: pd.DataFrame, min_cluster_size: int):
    """HDBSCAN on standardised features. Returns (labels, EV flags); noise (-1) is never EV."""
    Xs = StandardScaler().fit_transform(X)
    labels = HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(Xs)
    ev_c, _ = pick_ev_clusters(X, labels)
    ev_c = [c for c in ev_c if c != -1]
    return labels, np.isin(labels, ev_c).astype(int)


def score(name, baseline, y, types, every_night, pred, feats, labels=None):
    r = evaluate(name, y, pred)
    checked = power_check(feats, pred)
    r.update({
        "baseline": baseline,
        "every_night_found": int(pred[every_night].sum()),
        "hot_tubs_flagged": int(pred[types == "hot_tub"].sum()),
        "other_false_alarms": int(pred[(y == 0) & (types != "hot_tub")].sum()),
        "f1_after_power_check": evaluate("", y, checked)["f1"],
        "flagged_after_power_check": int(checked.sum()),
    })
    if labels is not None:
        r["clusters"] = int(len(set(labels) - {-1}))
        r["noise_homes"] = int((labels == -1).sum())
        r["ev_homes_in_noise"] = int(((labels == -1) & (y == 1)).sum())
    return r


def main(data="data", reports="reports"):
    data, reports = Path(data), Path(reports)
    readings = pd.read_csv(data / "meter_readings.csv.gz")
    weather = pd.read_csv(data / "weather.csv")
    homes = pd.read_csv(data / "homes.csv").set_index("home_id")

    rows, by_type = [], {}
    for baseline in ["median", "p20_edges"]:
        feats = build_features(readings, weather, baseline)
        X = feats[FEATURES_FOR_CLUSTERING]
        h = homes.loc[X.index]
        y = h["has_ev"].to_numpy()
        types = home_type(h).to_numpy()
        every_night = (h["charging_pattern"].fillna("") == "every_night").to_numpy()

        for mcs in MIN_CLUSTER_SIZES:
            labels, pred = hdbscan_homes(X, mcs)
            rows.append(score(f"HDBSCAN (min_cluster_size={mcs})", baseline, y, types, every_night,
                              pred, feats, labels))
            if mcs in (5, 10):
                by_type[f"{baseline}: HDBSCAN (min_cluster_size={mcs})"] = pd.Series(pred, index=X.index)

        gmm_preds = []
        for k in KS:
            for seed in SEEDS:
                c = cluster_homes(X, k, seed)
                gmm_preds.append(c["pred_gmm"])
                if k == 7 and seed == 0:
                    rows.append(score("GMM (k=7, seed 0, one run)", baseline, y, types, every_night,
                                      c["pred_gmm"], feats, c["g_labels"]))
                    by_type[f"{baseline}: GMM (k=7, one run)"] = pd.Series(c["pred_gmm"], index=X.index)
        f1s = [evaluate("", y, p)["f1"] for p in gmm_preds]
        found = [int(p[every_night].sum()) for p in gmm_preds]
        vote = (np.mean(gmm_preds, axis=0) >= 0.5).astype(int)
        r = score(f"GMM vote (k={KS.start}-{KS.stop - 1}, seeds {SEEDS.start}-{SEEDS.stop - 1}, "
                  f"{len(gmm_preds)} runs)", baseline, y, types, every_night, vote, feats)
        r.update({"f1_min_over_runs": min(f1s), "f1_max_over_runs": max(f1s),
                  "every_night_min_over_runs": min(found), "every_night_max_over_runs": max(found)})
        rows.append(r)
        print(f"{baseline} done")

    cols = ["baseline", "method", "clusters", "noise_homes", "ev_homes_in_noise", "precision", "recall", "f1",
            "estimated_ev_homes", "true_ev_homes", "every_night_found", "hot_tubs_flagged", "other_false_alarms",
            "f1_after_power_check", "flagged_after_power_check", "f1_min_over_runs", "f1_max_over_runs",
            "every_night_min_over_runs", "every_night_max_over_runs"]
    res = pd.DataFrame(rows)[cols]
    for c in ["clusters", "noise_homes", "ev_homes_in_noise", "every_night_min_over_runs", "every_night_max_over_runs"]:
        res[c] = res[c].astype("Int64")
    res.to_csv(reports / "hdbscan_comparison.csv", index=False)

    t = pd.Series(home_type(homes).loc[X.index])
    bt = pd.DataFrame(by_type).groupby(t).sum()
    bt.insert(0, "homes", t.value_counts())
    bt.index.name = "type"
    bt.to_csv(reports / "hdbscan_by_home_type.csv")

    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(res.to_string(index=False))
        print("\nFlagged as EV by home type:\n", bt.to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data"); p.add_argument("--reports", default="reports")
    a = p.parse_args()
    main(a.data, a.reports)
