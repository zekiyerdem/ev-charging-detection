"""Compare ways of defining a home's "typical" load, focusing on every-night chargers.

Homes that charge on most nights are hard to find: the original method compares each
hour with the home's yearly median for that hour, and when charging happens on most
nights the median already contains the charging, so no block of extra load is left.

For each baseline in features.BASELINES this script builds the features, clusters
the homes with K-Means and a GMM (same settings as detect.py), and reports the
overall precision / recall / F1 and the recall for every-night and some-nights
chargers separately. Labels are used only for scoring.

It also checks the step before clustering: for every true charging session, is
any of its hours inside a detected block? This shows directly whether a baseline
makes the charging visible, independent of how the clusters fall.

Writes reports/baseline_comparison.csv, reports/baseline_comparison_by_type.csv and
reports/baseline_session_visibility.csv.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from detect import FEATURES_FOR_CLUSTERING, cluster_homes, evaluate, home_type
from features import BASELINES, build_features, charging_blocks, excess_series, to_cube


def session_visibility(readings, homes, sessions, baseline):
    """Share of true charging sessions (per EV home) with at least one hour inside a detected block."""
    home_ids, days, cube = to_cube(readings[readings["home_id"].isin(homes.index[homes["has_ev"] == 1])])
    t0 = days[0]
    out = {}
    for k, hid in enumerate(home_ids):
        big, small = charging_blocks(excess_series(cube[k], baseline))
        in_block = np.zeros(cube[k].size, dtype=bool)
        for s, length, _ in big + small:
            in_block[s:s + length] = True
        ses = sessions[sessions["home_id"] == hid]
        pos = ((ses["start"] - t0) / pd.Timedelta(hours=1)).astype(int).to_numpy()
        seen = [in_block[p:p + h].any() for p, h in zip(pos, ses["hours"].to_numpy())]
        out[hid] = float(np.mean(seen)) if seen else np.nan
    return pd.Series(out)


def main(data="data", reports="reports", k=7, seed=0):
    data, reports = Path(data), Path(reports)
    readings = pd.read_csv(data / "meter_readings.csv.gz")
    weather = pd.read_csv(data / "weather.csv")
    homes = pd.read_csv(data / "homes.csv").set_index("home_id")
    pattern = homes["charging_pattern"].fillna("")
    sessions = pd.read_csv(data / "ev_sessions.csv", parse_dates=["start"])
    vis_rows = []

    rows, by_type = [], []
    for baseline in BASELINES:
        feats = build_features(readings, weather, baseline)
        X = feats[FEATURES_FOR_CLUSTERING]
        h = homes.loc[X.index]
        y = h["has_ev"].to_numpy()
        types = home_type(h)
        c = cluster_homes(X, k, seed)
        vis = session_visibility(readings, homes, sessions, baseline)
        grp = homes.loc[vis.index, "charging_pattern"] + " / " + homes.loc[vis.index, "charge_timing"]
        vis_rows.append(vis.groupby(grp).mean().round(3).rename(baseline))
        for algo, pred in [("K-Means", c["pred_km"]), ("GMM", c["pred_gmm"])]:
            r = evaluate(f"{baseline} + {algo}", y, pred)
            r.update({"baseline": baseline, "algorithm": algo})
            for pat in ["every_night", "some_nights"]:
                m = (pattern.loc[X.index] == pat).to_numpy()
                r[f"recall_{pat}"] = round(float(pred[m].mean()), 3)
                r[f"found_{pat}"] = f"{int(pred[m].sum())}/{int(m.sum())}"
            r["false_alarms"] = int(((pred == 1) & (y == 0)).sum())
            rows.append(r)
            t = pd.Series(pred, index=X.index).groupby(types).sum().rename(f"{baseline} + {algo}")
            by_type.append(t)

    cols = ["baseline", "algorithm", "precision", "recall", "f1", "recall_every_night", "found_every_night",
            "recall_some_nights", "found_some_nights", "false_alarms", "estimated_ev_homes", "true_ev_homes"]
    res = pd.DataFrame(rows)[cols]
    res.to_csv(reports / "baseline_comparison.csv", index=False)
    types_all = home_type(homes)
    tab = pd.concat(by_type, axis=1)
    tab.insert(0, "homes", types_all.value_counts().reindex(tab.index))
    tab.index.name = "type"
    tab.to_csv(reports / "baseline_comparison_by_type.csv")

    vis_tab = pd.concat(vis_rows, axis=1)
    vis_tab.index.name = "pattern / timing"
    vis_tab.to_csv(reports / "baseline_session_visibility.csv")

    print("Baselines:")
    for b, d in BASELINES.items():
        print(f"  {b:13s} {d}")
    print("\n" + res.to_string(index=False))
    print("\nHomes flagged as EV, by true home type:\n" + tab.to_string())
    print("\nShare of true charging sessions that show up inside a detected block (mean per home):\n"
          + vis_tab.to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data"); p.add_argument("--reports", default="reports")
    p.add_argument("--k", type=int, default=7); p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    main(a.data, a.reports, a.k, a.seed)
