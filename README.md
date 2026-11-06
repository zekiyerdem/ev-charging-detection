# Detecting Home EV Charging from Smart Meter Data

A practice project: can we find which households charge an electric vehicle (EV) at home, and how many there are, using **only hourly smart meter readings** and **unsupervised clustering**?

Real smart meter data is private, so this project **generates realistic synthetic data** for 600 Ontario-style homes over one year, hides the true answers, then tries to recover them.

## The idea in one picture

![Two weeks of load for four kinds of homes](reports/figures/example_homes_two_weeks.png)

Home EV charging leaves a clear fingerprint in the meter data:

| Signal | Level 2 charger (240 V) | Level 1 charger (120 V) |
|---|---|---|
| Extra power | ~6–7.7 kW | ~1.3–1.9 kW |
| Session length | 2–5 hours | 5–14 hours |
| Timing | evening, or 11 pm on an overnight-rate timer | evening/overnight |
| Shape | flat "block" that switches on and off sharply | long, low, flat block |
| Frequency | some nights, not others | some nights, not others |
| Season | all year | all year |

The hard part is **look-alikes**: hot tubs, electric water heaters and electric baseboard heating also add big blocks of load.

## Data (synthetic)

`src/generate_data.py` simulates, for each home:

* a daily routine (morning and evening peaks, weekday vs weekend),
* heating and air conditioning driven by a synthetic Toronto temperature series,
* random appliance spikes (oven, dryer),
* **EV charging for ~15% of homes** (70% Level 2, 30% Level 1; ~55% use an 11 pm timer for the Ultra-Low Overnight price),
* **look-alike loads for ~15% of homes** (water heater, hot tub, baseboard heating).

Outputs (in `data/`, not committed):

| File | Contents |
|---|---|
| `meter_readings.csv.gz` | `home_id, timestamp, kwh` (hourly, 5.26 M rows) |
| `homes.csv` | hidden truth: `has_ev`, `charger_level`, `lookalike` |
| `ev_sessions.csv` | hidden truth: every charging session |
| `weather.csv` | hourly temperature |

## Method

1. **Compare every hour with the home's own normal.** For each home, compute its median load for each hour of the day, then the *excess* above that.
2. **Find blocks.** Runs where the excess stays ≥ 3.5 kW for ≥ 2 h ("big", Level-2-like) or ≥ 1.1 kW for ≥ 4 h ("small", Level-1-like).
3. **Describe each home with 12 features** (`src/features.py`): how often blocks happen, their power and length, how flat they are, whether they start in the evening/night, how *consistent* the start time is, whether they cluster in winter, and how strongly night-time use tracks cold weather.
4. **Cluster homes** (`src/detect.py`) with K-Means and a Gaussian Mixture Model (GMM) on the standardised features.
5. **Name each cluster** from its typical (median) home using physical rules of thumb (e.g. "~6 kW blocks starting at a consistent time → Level 2 EV").
6. **Score against the hidden truth** (labels are used only here, never for training).

A **baseline** clusters each home's *average daily load shape*. It is the obvious first idea, but averaging a year of data blurs out charging that only happens some nights.

## Results (600 homes, 90 with an EV)

| Method | Precision | Recall | F1 | Homes flagged | True EV homes |
|---|---|---|---|---|---|
| Baseline: K-Means on average daily profile | 1.00 | 0.44 | 0.62 | 40 | 90 |
| Features + K-Means (k = 7) | 0.82 | 0.98 | 0.89 | 107 | 90 |
| Features + Gaussian Mixture (k = 7) | 0.88 | 0.82 | 0.85 | 84 | 90 |

What the GMM found:

| Cluster (median home) | Homes | Named as |
|---|---|---|
| ~6.1 kW blocks, start time spread 0.7 h, 24% in winter | 46 | EV (Level 2) |
| long ~1.5 kW blocks, start spread 2.5 h, all year | 38 | EV (Level 1) |
| ~4.7 kW blocks, consistent night start | 32 | Other large appliance (hot tubs **and 16 Level-2 EVs**) |
| blocks mostly in winter, night load tracks cold (r = −0.96) | 34 | Electric heating |
| no blocks | 450 | Typical homes (3 clusters) |

![Clusters vs true labels](reports/figures/clusters_pca.png)

**Takeaways**

* Engineered, physically meaningful features beat clustering raw daily shapes (F1 0.85–0.89 vs 0.62).
* Every Level 1 home was found; 46 of 62 Level 2 homes were found with zero false alarms from heating.
* The main confusion is **lower-power Level 2 chargers vs hot tubs** (both ~4–6 kW night blocks). Features that could separate them: hot tubs cycle in short bursts every night all year; EVs charge on some nights in one continuous block whose energy varies with driving.
* Choosing *k* is a judgement call: silhouette scores favour very few clusters, but more clusters separate the look-alikes better. Interpreting cluster profiles matters as much as the algorithm.

## Run it

```bash
pip install -r requirements.txt
./run_all.sh          # ~30 seconds: generate data, cluster, write reports/
```

Outputs in `reports/`: `results.csv`, `results_by_home_type.csv`, `cluster_profiles_gmm.csv`, `home_features.csv`, `home_predictions.csv`, figures.

## Next steps

* **Session-level detection**: count charging sessions and kWh per home (useful for grid planning, e.g. how much EV load lands at 11 pm).
* Add features for **energy per block** and **night-to-night variability** to split EVs from hot tubs.
* Try **DBSCAN/HDBSCAN** (no need to choose *k*) and **semi-supervised** labelling with a few known EV owners.
* Test on **15-minute** data, where charger on/off edges are much sharper.
* Real-data issues: missing reads, daylight-saving shifts, solar panels (negative net load), multiple EVs per home.
