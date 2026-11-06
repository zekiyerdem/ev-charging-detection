"""Cluster homes to find likely EV owners, then check the result against the hidden labels.

Two approaches are compared:
  1. Baseline: K-Means on each home's *average daily load shape*.
     Simple, but averaging over the year blurs out charging that only happens
     some nights, so it struggles.
  2. Feature-based: K-Means and a Gaussian Mixture Model on engineered features
     (flat high blocks, evening/night timing, on/off nights, temperature link).

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


def main(data="data", reports="reports", k=7, seed=0):
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

    # ---- 2) engineered features ----
    feats = build_features(readings, weather)
    feats.to_csv(reports / "home_features.csv")
    X = feats[FEATURES_FOR_CLUSTERING]
    Xs = pd.DataFrame(StandardScaler().fit_transform(X), index=X.index, columns=X.columns)
    y = homes.loc[Xs.index, "has_ev"].to_numpy()

    # choose k by silhouette (for information)
    sil = {kk: round(silhouette_score(Xs, KMeans(n_clusters=kk, n_init=20, random_state=seed).fit_predict(Xs)), 3)
           for kk in range(2, 8)}

    km = KMeans(n_clusters=k, n_init=50, random_state=seed).fit(Xs)
    ev_c, desc_km = pick_ev_clusters(X, km.labels_)
    pred_km = np.isin(km.labels_, ev_c).astype(int)
    results.append(evaluate(f"Features + K-Means (k={k})", y, pred_km, km.labels_))

    gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=seed, n_init=5).fit(Xs)
    g_labels = gmm.predict(Xs)
    ev_g, desc_g = pick_ev_clusters(X, g_labels)
    desc_g.round(2).to_csv(reports / "cluster_profiles_gmm.csv")
    desc_km.round(2).to_csv(reports / "cluster_profiles_kmeans.csv")
    prob_ev = gmm.predict_proba(Xs)[:, ev_g].sum(axis=1)
    pred_gmm = (prob_ev >= 0.5).astype(int)
    results.append(evaluate(f"Features + Gaussian Mixture (k={k})", y, pred_gmm, g_labels))

    # ---- save outputs ----
    out = homes.loc[Xs.index].copy()
    out["kmeans_cluster"] = km.labels_
    out["gmm_cluster"] = g_labels
    out["ev_probability_gmm"] = np.round(prob_ev, 3)
    out["predicted_ev"] = pred_gmm
    out.to_csv(reports / "home_predictions.csv")

    res = pd.DataFrame(results)
    res.to_csv(reports / "results.csv", index=False)

    # who fooled the model? errors by home type
    out["type"] = np.where(out["has_ev"] == 1, "EV " + out["charger_level"].astype(str),
                           np.where(out["lookalike"].fillna("") != "", out["lookalike"].fillna(""), "no EV"))
    by_type = out.groupby("type")["predicted_ev"].agg(["count", "sum"]).rename(
        columns={"count": "homes", "sum": "flagged_as_ev"})
    by_type.to_csv(reports / "results_by_home_type.csv")

    summary = {"silhouette_by_k": sil, "k_used": k,
               "ev_clusters_kmeans": [int(c) for c in ev_c], "ev_clusters_gmm": [int(c) for c in ev_g],
               "confusion_matrix_gmm[[TN,FP],[FN,TP]]": confusion_matrix(y, pred_gmm).tolist()}
    (reports / "summary.json").write_text(json.dumps(summary, indent=2))

    # ---- figures ----
    pts = PCA(n_components=2, random_state=seed).fit_transform(Xs)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].scatter(pts[:, 0], pts[:, 1], c=g_labels, cmap="tab10", s=14)
    ax[0].set_title("Homes coloured by GMM cluster (PCA view)")
    colors = np.where(out["has_ev"] == 1, "tab:red", np.where(out["type"].isin(["water_heater", "hot_tub", "baseboard_heat"]), "tab:orange", "tab:gray"))
    ax[1].scatter(pts[:, 0], pts[:, 1], c=colors, s=14)
    ax[1].set_title("True labels: red = EV, orange = look-alike, grey = other")
    for a in ax:
        a.set_xlabel("PC1"); a.set_ylabel("PC2")
    fig.tight_layout(); fig.savefig(fig_dir / "clusters_pca.png", dpi=120); plt.close(fig)

    # example weeks: an L2 home, an L1 home, a hot tub home, a normal home
    readings["timestamp"] = pd.to_datetime(readings["timestamp"])
    examples = []
    for label, mask in [("EV Level 2", out["type"] == "EV L2"), ("EV Level 1", out["type"] == "EV L1"),
                        ("Hot tub (look-alike)", out["type"] == "hot_tub"), ("No EV", out["type"] == "no EV")]:
        if mask.any():
            examples.append((label, out.index[mask][0]))
    fig, axes = plt.subplots(len(examples), 1, figsize=(12, 2.4 * len(examples)), sharex=True)
    week = (readings["timestamp"] >= "2025-03-03") & (readings["timestamp"] < "2025-03-17")
    for a, (label, hid) in zip(np.atleast_1d(axes), examples):
        s = readings[week & (readings["home_id"] == hid)]
        a.plot(s["timestamp"], s["kwh"], lw=1)
        a.set_title(f"{label} ({hid})", fontsize=10); a.set_ylabel("kWh/h")
    fig.tight_layout(); fig.savefig(fig_dir / "example_homes_two_weeks.png", dpi=120); plt.close(fig)

    print(res.to_string(index=False))
    print("\nGMM cluster profiles (medians):\n", desc_g[["homes", "big_block_rate", "big_block_kw", "big_block_start_spread",
          "small_block_rate", "small_block_start_spread", "block_winter_share", "night_temp_corr", "name"]].round(2).to_string())
    print("\nFlagged as EV by home type (GMM):\n", by_type.to_string())
    print("\nSilhouette by k:", sil)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data"); p.add_argument("--reports", default="reports")
    p.add_argument("--k", type=int, default=7); p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    main(a.data, a.reports, a.k, a.seed)
