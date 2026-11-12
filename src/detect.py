"""Cluster homes to find likely EV owners, then check the result against the hidden labels.

Default decision (chosen in src/stability.py):
  * features built with the `p20_edges` typical-load baseline (catches homes that
    charge almost every night as well as some-nights chargers),
  * K-Means repeated over k = 5..9 and seeds 0..4 (25 runs),
  * each home gets an EV score = share of runs that flag it; a home is flagged as
    an EV home when the score is at least 0.5 (majority vote),
  * charger power check: a flagged home that has regular Level-2-like blocks must reach
    charger power in them (typical block peak >= 5.5 kW). This removes hot tubs, whose
    3-5 kW heaters make the same kind of night block but at lower power.

For comparison the script also runs:
  1. a baseline: K-Means on each home's *average daily load shape*,
  2. the original setup: `median` baseline features, one K-Means and one GMM run (k = 7),
  3. one K-Means run (k = 7) on the default features, used for the readable cluster table,
  4. the vote without the power check, and the vote with the three new block features
     (peak kW, kWh per block, night-to-night variability) added to the clustering instead,
  5. a rules-only detector: the same physical thresholds applied to each home, no clustering.

Clustering is unsupervised: the labels are used ONLY at the end to score the result.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, confusion_matrix, precision_score,
                             recall_score, f1_score, silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from features import average_profiles, build_features

FEATURES_FOR_CLUSTERING = [
    "big_block_rate", "big_block_kw", "big_block_hours", "big_block_evening_share", "big_block_flatness",
    "small_block_rate", "small_block_evening_share", "high_night_share",
    "night_temp_corr", "big_block_start_spread", "small_block_start_spread", "block_winter_share",
]

# per-block features added on 2026-11-10 to split Level 2 EVs from hot tubs; they are used
# by the power check, and tried as extra clustering features in a comparison row
BLOCK_FEATURES = ["big_block_peak_kw", "big_block_kwh", "big_block_night_kwh_cv"]

# Charger power check. Level 2 chargers on a 24-32 A circuit draw 5.8-7.7 kW; most hot tub
# heaters are 4-5.5 kW. A home counts as having "regular Level-2-like blocks" when they occur
# on at least 3% of days (about 11 a year); Level 1 homes almost never have them, so they are
# not affected. Known limit: a 16 A (3.8 kW) Level 2 charger would be removed as well.
CHARGER_MIN_KW = 5.5
L2_BLOCK_RATE = 0.03


def power_check(feats: pd.DataFrame, flagged: np.ndarray, min_kw: float = CHARGER_MIN_KW,
                min_rate: float = L2_BLOCK_RATE) -> np.ndarray:
    """Un-flag homes whose regular Level-2-like blocks never reach charger power."""
    low_power = ((feats["big_block_rate"] >= min_rate) & (feats["big_block_peak_kw"] < min_kw)).to_numpy()
    return (flagged.astype(bool) & ~low_power).astype(int)


def rules_only(feats: pd.DataFrame) -> np.ndarray:
    """No clustering: flag a home straight from thresholds on its own features, using the same
    rules of thumb that name the clusters. Level 2: Level-2-like blocks on >= 3% of days that
    reach charger power and start at a consistent time. Level 1: long low blocks on >= 8% of
    days, consistent start, not winter-only, night load not tracking the cold."""
    l2 = ((feats["big_block_rate"] >= L2_BLOCK_RATE) & (feats["big_block_peak_kw"] >= CHARGER_MIN_KW)
          & (feats["big_block_start_spread"] <= 2.5))
    l1 = ((feats["small_block_rate"] >= 0.08) & (feats["small_block_start_spread"] <= 3.5)
          & (feats["block_winter_share"] <= 0.6) & (feats["night_temp_corr"] > -0.8))
    return (l2 | l1).astype(int).to_numpy()


def describe_clusters(X: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Median of each feature per cluster, in original units, so a person can read it."""
    d = X.groupby(labels).median()
    d.insert(0, "homes", pd.Series(labels).value_counts().sort_index().values)
    return d


def label_cluster(row) -> str:
    """Name a cluster from its typical (median) home, using physical rules of thumb:
    - Level 2 charger: blocks of ~5-8 kW extra, lasting 2+ hours, starting at a consistent time
    - Level 1 charger: long (4+ h) blocks of ~1.3-2 kW extra, evening/overnight, all year
    - electric heating: blocks concentrated in winter and night load tracks cold weather
    """
    if row["big_block_rate"] >= 0.05 and row["big_block_kw"] >= 5.0 and row["big_block_start_spread"] <= 2.5:
        return "EV (Level 2 pattern)"
    if (row["small_block_rate"] >= 0.08 and row["small_block_start_spread"] <= 3.5
            and row["block_winter_share"] <= 0.6 and row["night_temp_corr"] > -0.8):
        return "EV (Level 1 pattern)"
    if row["block_winter_share"] > 0.6 or row["night_temp_corr"] <= -0.9:
        return "Electric heating"
    if row["big_block_rate"] >= 0.05:
        return "Other large appliance"
    return "Typical home"


def pick_ev_clusters(X: pd.DataFrame, labels: np.ndarray):
    desc = describe_clusters(X, labels)
    desc["name"] = desc.apply(label_cluster, axis=1)
    return [c for c in desc.index if desc.loc[c, "name"].startswith("EV")], desc


def cluster_homes(X: pd.DataFrame, k: int = 7, seed: int = 0, gmm: bool = True) -> dict:
    """Standardise features, run K-Means (and the GMM unless gmm=False), name the clusters, flag EV homes."""
    Xs = pd.DataFrame(StandardScaler().fit_transform(X), index=X.index, columns=X.columns)
    km = KMeans(n_clusters=k, n_init=50, random_state=seed).fit(Xs)
    ev_c, desc_km = pick_ev_clusters(X, km.labels_)
    out = {"Xs": Xs, "km_labels": km.labels_, "ev_c": ev_c, "desc_km": desc_km,
           "pred_km": np.isin(km.labels_, ev_c).astype(int)}
    if not gmm:
        return out
    gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=seed, n_init=5).fit(Xs)
    g_labels = gmm.predict(Xs)
    ev_g, desc_g = pick_ev_clusters(X, g_labels)
    prob_ev = gmm.predict_proba(Xs)[:, ev_g].sum(axis=1)
    return {**out, "g_labels": g_labels, "ev_g": ev_g, "desc_g": desc_g, "prob_ev": prob_ev,
            "pred_gmm": (prob_ev >= 0.5).astype(int)}


VOTE_KS = range(5, 10)
VOTE_SEEDS = range(5)


def kmeans_vote(X: pd.DataFrame, ks=VOTE_KS, seeds=VOTE_SEEDS, threshold: float = 0.5):
    """EV score of every home = share of K-Means runs (all k x seed settings) that flag it.

    Returns (score, flagged): flagged = score >= threshold (majority vote)."""
    preds = [cluster_homes(X, k, seed, gmm=False)["pred_km"] for k in ks for seed in seeds]
    score = pd.Series(np.mean(preds, axis=0), index=X.index, name="ev_score")
    return score, (score.to_numpy() >= threshold).astype(int)


def home_type(homes: pd.DataFrame) -> pd.Series:
    """Readable home type from the hidden labels, e.g. 'EV L2 every-night' or 'hot_tub'."""
    pattern = homes.get("charging_pattern", pd.Series("", index=homes.index)).fillna("")
    ev = "EV " + homes["charger_level"].astype(str) + np.where(pattern == "every_night", " every-night", "")
    look = homes["lookalike"].fillna("")
    return pd.Series(np.where(homes["has_ev"] == 1, ev, np.where(look != "", look, "no EV")),
                     index=homes.index)


def evaluate(name, y_true, y_pred, cluster_labels=None):
    out = {
        "method": name,
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 3),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 3),
        "estimated_ev_homes": int(y_pred.sum()),
        "true_ev_homes": int(y_true.sum()),
    }
    if cluster_labels is not None:
        out["adjusted_rand_index"] = round(adjusted_rand_score(y_true, cluster_labels), 3)
    return out


def main(data="data", reports="reports", k=7, seed=0, baseline="p20_edges", threshold=0.5):
    data, reports = Path(data), Path(reports)
    fig_dir = reports / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    readings = pd.read_csv(data / "meter_readings.csv.gz")
    weather = pd.read_csv(data / "weather.csv")
    homes = pd.read_csv(data / "homes.csv").set_index("home_id")

    results = []

    # ---- 1) baseline: average daily shape ----
    prof = average_profiles(readings)
    y = homes.loc[prof.index, "has_ev"].to_numpy()
    km0 = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(prof)
    # call the cluster with the most night-time share "EV"
    night_share = prof[[23, 0, 1, 2, 3, 4]].sum(axis=1).groupby(km0.labels_).mean()
    pred0 = np.isin(km0.labels_, [night_share.idxmax()]).astype(int)
    results.append(evaluate("Baseline: K-Means on average daily profile", y, pred0, km0.labels_))

    # ---- 2) original setup for comparison: median-baseline features, one run each ----
    X_med = build_features(readings, weather, "median")[FEATURES_FOR_CLUSTERING]
    y = homes.loc[X_med.index, "has_ev"].to_numpy()
    c_med = cluster_homes(X_med, k, seed)
    results.append(evaluate(f"Original: median features + K-Means (k={k})", y, c_med["pred_km"], c_med["km_labels"]))
    results.append(evaluate(f"Original: median features + GMM (k={k})", y, c_med["pred_gmm"], c_med["g_labels"]))
    c_med["desc_g"].round(2).to_csv(reports / "cluster_profiles_gmm.csv")

    # ---- 3) default: chosen baseline, one readable K-Means run + the K-Means vote ----
    feats = build_features(readings, weather, baseline)
    feats.to_csv(reports / "home_features.csv")
    X = feats[FEATURES_FOR_CLUSTERING]
    assert (X.index == X_med.index).all()
    c = cluster_homes(X, k, seed, gmm=False)
    Xs, km, desc_km = c["Xs"], c["km_labels"], c["desc_km"]
    results.append(evaluate(f"{baseline} features + K-Means (k={k}, one run)", y, c["pred_km"], km))
    desc_km.round(2).to_csv(reports / "cluster_profiles_kmeans.csv")

    score, pred_vote = kmeans_vote(X, threshold=threshold)
    n_runs = len(VOTE_KS) * len(VOTE_SEEDS)
    vote_name = f"{baseline} features + K-Means vote (k={VOTE_KS.start}-{VOTE_KS.stop - 1}, {n_runs} runs)"
    results.append(evaluate(vote_name, y, pred_vote))

    # the new block features as extra clustering features (instead of a separate check)
    _, pred_more = kmeans_vote(feats[FEATURES_FOR_CLUSTERING + BLOCK_FEATURES], threshold=threshold)
    results.append(evaluate(f"{vote_name}, + 3 block features in the clustering", y, pred_more))

    pred = power_check(feats, pred_vote)
    prefix = "DEFAULT: " if baseline == "p20_edges" else ""
    results.append(evaluate(f"{prefix}{vote_name} + power check (peak >= {CHARGER_MIN_KW} kW)", y, pred))

    pred_rules = rules_only(feats)
    results.append(evaluate("Rules only, no clustering (same thresholds)", y, pred_rules))

    # how much does the result depend on the 5.5 kW / 3% choice?
    types = home_type(homes.loc[X.index]).to_numpy()
    sens = []
    for min_kw in [4.5, 5.0, 5.5, 6.0, 6.5]:
        for min_rate in [0.01, 0.03, 0.05]:
            p_ = power_check(feats, pred_vote, min_kw, min_rate)
            r = evaluate("", y, p_); r.pop("method")
            sens.append({"min_peak_kw": min_kw, "min_l2_block_rate": min_rate, **r,
                         "ev_found": int(((p_ == 1) & (y == 1)).sum()),
                         "hot_tubs_flagged": int(p_[types == "hot_tub"].sum()),
                         "other_false_alarms": int(p_[(y == 0) & (types != "hot_tub")].sum())})
    pd.DataFrame(sens).to_csv(reports / "power_check_sensitivity.csv", index=False)

    # choose k by silhouette (for information)
    sil = {kk: round(silhouette_score(Xs, KMeans(n_clusters=kk, n_init=20, random_state=seed).fit_predict(Xs)), 3)
           for kk in range(2, 10)}

    # ---- save outputs ----
    out = homes.loc[Xs.index].copy()
    out["kmeans_cluster"] = km
    out["ev_score"] = score.round(3)
    out["predicted_ev"] = pred
    out["predicted_ev_vote_only"] = pred_vote
    out["predicted_ev_rules_only"] = pred_rules
    out["predicted_ev_original_gmm"] = c_med["pred_gmm"]
    out.to_csv(reports / "home_predictions.csv")

    res = pd.DataFrame(results)
    res.to_csv(reports / "results.csv", index=False)

    # who fooled the model? errors by home type (default decision and the original GMM)
    out["type"] = home_type(out)
    by_type = out.groupby("type").agg(homes=("predicted_ev", "size"), flagged_as_ev=("predicted_ev", "sum"),
                                      flagged_by_vote_only=("predicted_ev_vote_only", "sum"),
                                      flagged_by_rules_only=("predicted_ev_rules_only", "sum"),
                                      flagged_by_original_gmm=("predicted_ev_original_gmm", "sum"))
    by_type.to_csv(reports / "results_by_home_type.csv")

    # who is in each cluster of the readable k=7 run
    members = pd.crosstab(out["kmeans_cluster"], out["type"])
    members.insert(0, "name", desc_km["name"])
    members.to_csv(reports / "cluster_members_kmeans.csv")

    summary = {"baseline": baseline, "decision": f"K-Means majority vote, k={list(VOTE_KS)}, seeds={list(VOTE_SEEDS)}, "
                                                    f"then power check (block peak >= {CHARGER_MIN_KW} kW when "
                                                    f"Level-2-like blocks on >= {L2_BLOCK_RATE:.0%} of days)",
               "threshold": threshold, "silhouette_by_k": sil, "k_for_cluster_table": k,
               "ev_clusters_kmeans_k7": [int(c) for c in c["ev_c"]],
               "estimated_ev_homes": int(pred.sum()), "true_ev_homes": int(y.sum()),
               "confusion_matrix_default[[TN,FP],[FN,TP]]": confusion_matrix(y, pred).tolist()}
    (reports / "summary.json").write_text(json.dumps(summary, indent=2))

    # ---- figures ----
    pts = PCA(n_components=2, random_state=seed).fit_transform(Xs)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sc = ax[0].scatter(pts[:, 0], pts[:, 1], c=score, cmap="viridis", s=14, vmin=0, vmax=1)
    fig.colorbar(sc, ax=ax[0], label="EV score (share of K-Means runs)")
    ax[0].set_title("Homes coloured by EV score")
    colors = np.where(out["has_ev"] == 1, "tab:red", np.where(out["type"].isin(["water_heater", "hot_tub", "baseboard_heat"]), "tab:orange", "tab:gray"))
    ax[1].scatter(pts[:, 0], pts[:, 1], c=colors, s=14)
    ax[1].set_title("True labels: red = EV, orange = look-alike, grey = other")
    for a in ax:
        a.set_xlabel("PC1"); a.set_ylabel("PC2")
    fig.tight_layout(); fig.savefig(fig_dir / "clusters_pca.png", dpi=120); plt.close(fig)

    # example weeks: L2, every-night L2, L1, hot tub, normal home, with the EV score of each
    readings["timestamp"] = pd.to_datetime(readings["timestamp"])
    examples = []
    for label, mask in [("EV Level 2", out["type"] == "EV L2"),
                        ("EV Level 2, charges every night", out["type"] == "EV L2 every-night"),
                        ("EV Level 1", out["type"] == "EV L1"),
                        ("Hot tub (look-alike)", out["type"] == "hot_tub"), ("No EV", out["type"] == "no EV")]:
        if mask.any():
            examples.append((label, out.index[mask][0]))
    fig, axes = plt.subplots(len(examples), 1, figsize=(12, 2.4 * len(examples)), sharex=True)
    week = (readings["timestamp"] >= "2025-03-03") & (readings["timestamp"] < "2025-03-17")
    for a, (label, hid) in zip(np.atleast_1d(axes), examples):
        s = readings[week & (readings["home_id"] == hid)]
        a.plot(s["timestamp"], s["kwh"], lw=1)
        verdict = ("flagged as EV" if out.loc[hid, "predicted_ev"] else
                   "removed by the power check" if out.loc[hid, "predicted_ev_vote_only"] else "not flagged")
        a.set_title(f"{label} ({hid}): EV score {out.loc[hid, 'ev_score']:.2f}, {verdict}", fontsize=10)
        a.set_ylabel("kWh/h")
    fig.tight_layout(); fig.savefig(fig_dir / "example_homes_two_weeks.png", dpi=120); plt.close(fig)

    print(res.to_string(index=False))
    print(f"\nK-Means k={k} cluster profiles ({baseline}, medians):\n", desc_km[["homes", "big_block_rate", "big_block_kw",
          "big_block_start_spread", "small_block_rate", "small_block_start_spread", "block_winter_share",
          "night_temp_corr", "name"]].round(2).to_string())
    print("\nWho is in each cluster:\n", members.to_string())
    print("\nFlagged as EV by home type:\n", by_type.to_string())
    print("\nPower check sensitivity:\n", pd.DataFrame(sens).to_string(index=False))
    print("\nExample homes:", examples)
    print("\nSilhouette by k:", sil)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data"); p.add_argument("--reports", default="reports")
    p.add_argument("--k", type=int, default=7); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--baseline", default="p20_edges", help="typical-load method, see features.BASELINES")
    p.add_argument("--threshold", type=float, default=0.5, help="EV score needed to flag a home")
    a = p.parse_args()
    main(a.data, a.reports, a.k, a.seed, a.baseline, a.threshold)
