"""From "which homes charge an EV" to "when and how much": session-level EV load.

For every home that detect.py flags as an EV home (reports/home_predictions.csv):
  1. take the extra load above the home's own normal (the default `p20_edges` excess),
  2. keep the charging blocks: Level-2-like blocks (>= 3.5 kW for >= 2 h) for homes that
     have them regularly, otherwise Level-1-like blocks (>= 1.1 kW for >= 4 h),
  3. each block is one estimated charging session; its excess load is the estimated EV load.

The estimate is compared with the hidden session table (data/ev_sessions.csv). That table
stores the charger power and the number of hours of each session; the true EV load per hour
is taken as the charger power for every hour of the session, so it slightly over-states the
last, partly used hour.

Writes reports/ev_load_by_hour.csv, reports/session_summary.json and
reports/figures/ev_load_by_hour.png.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from detect import L2_BLOCK_RATE
from features import charging_blocks, excess_series, to_cube


def estimated_ev_load(home_cube: np.ndarray, baseline: str = "p20_edges"):
    """Hourly estimated EV load (kWh, flat series) and the list of sessions (start, hours, kWh)."""
    excess = excess_series(home_cube, baseline)
    big, small = charging_blocks(excess)
    n_days = home_cube.shape[0]
    blocks = big if len(big) / n_days >= L2_BLOCK_RATE else small
    load = np.zeros_like(excess)
    sessions = []
    for s, length, _ in blocks:
        seg = np.clip(excess[s:s + length], 0, None)
        load[s:s + length] = seg
        sessions.append((s, length, float(seg.sum())))
    return load, sessions


def true_ev_load(sessions: pd.DataFrame, index: pd.DatetimeIndex, home_ids) -> np.ndarray:
    """Hourly true EV load per home, shape (homes, hours), from the hidden session table."""
    pos = {h: i for i, h in enumerate(home_ids)}
    out = np.zeros((len(home_ids), len(index)))
    starts = index.searchsorted(pd.to_datetime(sessions["start"]))
    for h, st, hrs, kw in zip(sessions["home_id"], starts, sessions["hours"], sessions["kw"]):
        out[pos[h], st:st + hrs] += kw
    return out


def main(data="data", reports="reports"):
    data, reports = Path(data), Path(reports)
    readings = pd.read_csv(data / "meter_readings.csv.gz")
    true_sessions = pd.read_csv(data / "ev_sessions.csv")
    pred = pd.read_csv(reports / "home_predictions.csv").set_index("home_id")

    home_ids, days, cube = to_cube(readings)
    index = pd.date_range(days[0], periods=len(days) * 24, freq="h")
    flagged = pred.index[pred["predicted_ev"] == 1]
    ev_homes = pred.index[pred["has_ev"] == 1]

    est = np.zeros((len(home_ids), len(index)))
    rows = []
    for k, hid in enumerate(home_ids):
        if hid not in flagged:
            continue
        est[k], sess = estimated_ev_load(cube[k])
        rows.append({"home_id": hid, "estimated_sessions": len(sess),
                     "estimated_kwh": round(sum(e for *_, e in sess), 1)})
    per_home = pd.DataFrame(rows).set_index("home_id")
    truth = true_ev_load(true_sessions, index, home_ids)
    t_home = pd.DataFrame({"true_sessions": true_sessions.groupby("home_id").size(),
                           "true_kwh": true_sessions.assign(e=true_sessions.hours * true_sessions.kw)
                                                    .groupby("home_id")["e"].sum().round(1)})
    per_home = per_home.join(t_home, how="outer").fillna(0)
    per_home["type"] = pred.loc[per_home.index, "charger_level"].fillna("") + " " + \
        pred.loc[per_home.index, "charging_pattern"].fillna("")
    per_home.to_csv(reports / "ev_sessions_per_home.csv")

    # hour-of-day profile of total EV load, average day (kW summed over all homes)
    hod = index.hour
    prof = pd.DataFrame({"hour": range(24),
                         "estimated_ev_kw": [est[:, hod == h].sum(axis=0).mean() for h in range(24)],
                         "true_ev_kw": [truth[:, hod == h].sum(axis=0).mean() for h in range(24)]})
    prof = prof.round(1)
    prof.to_csv(reports / "ev_load_by_hour.csv", index=False)

    est_total, true_total = est.sum(), truth.sum()
    # the evening with the largest total EV load (coincident peak of all homes)
    est_series, true_series = est.sum(axis=0), truth.sum(axis=0)
    both = per_home.loc[per_home.index.isin(flagged) & per_home.index.isin(ev_homes)]
    by_type = (both.groupby("type")[["estimated_sessions", "true_sessions", "estimated_kwh", "true_kwh"]]
                   .sum().astype(int))
    summary = {
        "flagged_homes": int(len(flagged)), "true_ev_homes": int(len(ev_homes)),
        "estimated_sessions": int(per_home["estimated_sessions"].sum()),
        "true_sessions": int(len(true_sessions)),
        "estimated_ev_mwh_per_year": round(est_total / 1000, 1),
        "true_ev_mwh_per_year_upper": round(true_total / 1000, 1),
        "estimated_share_of_energy_in_23h": round(prof.loc[23, "estimated_ev_kw"] / prof["estimated_ev_kw"].sum(), 3),
        "true_share_of_energy_in_23h": round(prof.loc[23, "true_ev_kw"] / prof["true_ev_kw"].sum(), 3),
        "estimated_peak_hour_of_day": int(prof["estimated_ev_kw"].idxmax()),
        "true_peak_hour_of_day": int(prof["true_ev_kw"].idxmax()),
        "estimated_avg_kw_at_23h": float(prof.loc[23, "estimated_ev_kw"]),
        "true_avg_kw_at_23h": float(prof.loc[23, "true_ev_kw"]),
        "estimated_max_coincident_kw": round(float(est_series.max()), 1),
        "estimated_max_coincident_at": str(index[est_series.argmax()]),
        "true_max_coincident_kw": round(float(true_series.max()), 1),
        "true_max_coincident_at": str(index[true_series.argmax()]),
        "by_type": by_type.to_dict(orient="index"),
    }
    (reports / "session_summary.json").write_text(json.dumps(summary, indent=2))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(prof["hour"] - 0.2, prof["true_ev_kw"], width=0.4, label="true (hidden sessions)", color="tab:gray")
    ax.bar(prof["hour"] + 0.2, prof["estimated_ev_kw"], width=0.4, label="estimated from the meter", color="tab:blue")
    ax.set_xticks(range(24)); ax.set_xlabel("hour of day"); ax.set_ylabel("average EV load, all homes (kW)")
    ax.set_title(f"EV charging load by hour, {len(ev_homes)} EV homes out of {len(home_ids)}")
    ax.legend(); fig.tight_layout()
    fig.savefig(reports / "figures" / "ev_load_by_hour.png", dpi=120); plt.close(fig)

    print(prof.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data"); p.add_argument("--reports", default="reports")
    a = p.parse_args()
    main(a.data, a.reports)
