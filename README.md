# Detecting Home EV Charging from Smart Meter Data

A practice project: can we find which households charge an electric vehicle (EV) at home, and how many there are, using **only hourly smart meter readings** and **unsupervised clustering**?

Real smart meter data is private, so this project **generates realistic synthetic data** for 630 Ontario-style homes over one year (600 general homes plus 30 extra homes that charge an EV almost every night), hides the true answers, then tries to recover them.

## The idea in one picture

![Two weeks of load for four kinds of homes](reports/figures/example_homes_two_weeks.png)

Home EV charging leaves a clear fingerprint in the meter data:

| Signal | Level 2 charger (240 V) | Level 1 charger (120 V) |
|---|---|---|
| Extra power | ~6–7.7 kW | ~1.3–1.9 kW |
| Session length | 2–5 hours | 5–14 hours |
| Timing | evening, or 11 pm on an overnight-rate timer | evening/overnight |
| Shape | flat "block" that switches on and off sharply | long, low, flat block |
| Frequency | some nights, not others (or every night, see below) | some nights, not others |
| Season | all year | all year |

The hard part is **look-alikes**: hot tubs, electric water heaters and electric baseboard heating also add big blocks of load.

## Data (synthetic)

`src/generate_data.py` simulates, for each home:

* a daily routine (morning and evening peaks, weekday vs weekend),
* heating and air conditioning driven by a synthetic Toronto temperature series,
* random appliance spikes (oven, dryer),
* **EV charging for ~15% of homes** (70% Level 2, 30% Level 1; ~55% use an 11 pm timer for the Ultra-Low Overnight price),
* **look-alike loads for ~15% of homes** (water heater, hot tub, baseboard heating),
* **30 extra every-night chargers** (23 Level 2, 7 Level 1): they plug in on 85–100% of nights and top up only what that day's driving used (Level 2: usually 1–3 h), either starting at 11 pm on a timer or finishing at a set departure time (6 or 7 am). They come from their own random stream, so the first 600 homes are identical with or without them (`--every-night-homes 0` reproduces the original 600-home data).

Outputs (in `data/`, not committed):

| File | Contents |
|---|---|
| `meter_readings.csv.gz` | `home_id, timestamp, kwh` (hourly, 5.52 M rows) |
| `homes.csv` | hidden truth: `has_ev`, `charger_level`, `charging_pattern` (some_nights / every_night), `charge_timing`, `lookalike` |
| `ev_sessions.csv` | hidden truth: every charging session |
| `weather.csv` | hourly temperature |

## Method

1. **Compare every hour with the home's own normal.** For each home, compute its typical load for each hour of the day, then the *excess* above that. The default (`p20_edges`) takes, hour by hour, the larger of (a) the excess over the home's 20th-percentile load for that hour and (b) the excess after a sudden jump of ≥ 1.1 kW, until the load falls back. The first version used the yearly median, which hides homes that charge almost every night; see [Every-night chargers](#every-night-chargers-a-blind-spot-of-the-typical-load-baseline).
2. **Find blocks.** Runs where the excess stays ≥ 3.5 kW for ≥ 2 h ("big", Level-2-like) or ≥ 1.1 kW for ≥ 4 h ("small", Level-1-like).
3. **Describe each home with 12 features** (`src/features.py`): how often blocks happen, their power and length, how flat they are, whether they start in the evening/night, how *consistent* the start time is, whether they cluster in winter, and how strongly night-time use tracks cold weather.
4. **Cluster homes** (`src/detect.py`) with K-Means on the standardised features.
5. **Name each cluster** from its typical (median) home using physical rules of thumb (e.g. "~6 kW blocks starting at a consistent time → Level 2 EV").
6. **Vote instead of trusting one run.** Steps 4–5 are repeated for k = 5–9 clusters and 5 random seeds (25 runs). Each home's **EV score** is the share of runs that put it in an EV cluster; a home is flagged when the score is at least 0.5. The estimated number of EV homes is the number flagged.
7. **Score against the hidden truth** (labels are used only here, never for training).

A **baseline** clusters each home's *average daily load shape*. It is the obvious first idea, but averaging a year of data blurs out charging that only happens some nights. The first version of this project (median baseline, one K-Means or GMM run with k = 7) is kept as a comparison.

## Results (630 homes, 120 with an EV)

| Method | Precision | Recall | F1 | Homes flagged | True EV homes |
|---|---|---|---|---|---|
| Baseline: K-Means on average daily profile | 1.00 | 0.34 | 0.51 | 41 | 120 |
| First version: median features + K-Means (k = 7) | 0.84 | 0.85 | 0.85 | 121 | 120 |
| First version: median features + GMM (k = 7) | 0.77 | 0.93 | 0.84 | 146 | 120 |
| p20_edges features + K-Means (k = 7, one run) | 0.86 | 0.98 | 0.92 | 138 | 120 |
| **Default: p20_edges features + K-Means vote (k = 5–9, 25 runs)** | **0.85** | **0.98** | **0.91** | **139** | **120** |

The default finds 118 of the 120 EV homes. Flagged by home type (default vs the first-version GMM):

| Home type | Homes | Flagged (default) | Flagged (first-version GMM) |
|---|---|---|---|
| EV Level 2, some nights | 62 | 62 | 62 |
| EV Level 2, every night | 23 | 21 | 15 |
| EV Level 1, some nights | 28 | 28 | 28 |
| EV Level 1, every night | 7 | 7 | 7 |
| Hot tub | 22 | **21** | 21 |
| Electric water heater | 17 | 0 | 0 |
| Baseboard heating | 34 | 0 | 0 |
| No EV, no look-alike | 437 | 0 | 13 |

The EV score is clear-cut for most homes: 119 homes are flagged in all 25 runs, 486 in none, and only 25 fall in between (mostly Level 1 chargers at 0.8, flagged in every run except k = 5, and four water heaters at 0.04–0.2).

What one K-Means run (k = 7) on the default features finds:

| Cluster (median home) | Homes | Named as | Who is really in it |
|---|---|---|---|
| ~6.3 kW blocks on 42% of days, 3.2 h long, start spread 0.5 h | 103 | EV (Level 2) | 62 Level 2 EVs, 21 every-night Level 2 EVs, **20 hot tubs** |
| long ~1.1–2 kW blocks on 53% of days, start spread 1.5 h | 35 | EV (Level 1) | 35 Level 1 EVs (incl. 7 every-night), nothing else |
| blocks of ~45 h, night load tracks cold (r = −0.96) | 34 | Electric heating | 34 baseboard heating |
| few blocks, mostly in winter (74–75%) | 178 | Electric heating (2 clusters) | 176 homes without an EV, 1 water heater, 1 every-night Level 2 EV |
| no or rare blocks | 280 | Typical homes (2 clusters) | 261 homes without an EV, 16 water heaters, 2 hot tubs, 1 every-night Level 2 EV |

The naming rules call two clusters of ordinary homes "electric heating" because their few blocks fall in winter. That label is wrong but harmless here: those clusters are not EV clusters, so it does not change who is flagged. Full tables: `reports/cluster_profiles_kmeans.csv` and `reports/cluster_members_kmeans.csv`.

![Homes coloured by EV score vs true labels](reports/figures/clusters_pca.png)

**Takeaways**

* Engineered, physically meaningful features beat clustering raw daily shapes (F1 0.91 vs 0.51).
* The new default finds 28 of 30 every-night chargers (the first-version K-Means found 14) and all 90 some-nights chargers, and the Level 1 cluster is now pure: no false alarms from homes without an EV.
* All 21 false alarms are **hot tubs** (21 of 22): like a Level 2 charger they add a flat ~4–6 kW block at night. So the estimated count (139) is too high by about the number of hot tubs (true: 120). Separating EVs from hot tubs is the main open problem.
* The two missed homes are every-night Level 2 chargers whose top-ups are mostly 1 hour long, shorter than the 2-hour block rule.
* Choosing *k* is a judgement call (silhouette is highest at k = 7 among k = 2–9, 0.40, but close for 2, 6 and 8). The vote over k = 5–9 removes the need to pick one.

## Every-night chargers: a blind spot of the "typical load" baseline

Many EV owners on the Ultra-Low Overnight rate plug in **every night**. Then charging is part of the home's normal night, so the yearly median for 11 pm or 5 am already contains it and the extra-load blocks disappear. `src/compare_baselines.py` measures this and tries five alternatives to the yearly median (`features.BASELINES`):

| Baseline | How "typical load" is defined |
|---|---|
| `median` (original) | median of each hour of day over the year |
| `p20` | 20th percentile of each hour of day over the year |
| `quiet_nights` | median of each hour of day over quiet nights only (largest night hour within 1 kW of the home's 10th percentile) |
| `rolling` | rolling 29-day median of each hour of day |
| `edges` | no baseline: a block starts with a jump of ≥ 1.1 kW and ends when load falls back by half the jump |
| `p20_edges` | hour by hour, the larger of the `p20` excess and the `edges` excess (keeps both kinds of block) |

**Step 1: does the charging show up at all?** Share of true charging sessions with at least one hour inside a detected block (mean per home):

| Charging pattern | median | p20 | quiet_nights | rolling | edges | p20_edges |
|---|---|---|---|---|---|---|
| every night, 11 pm timer | 0.06 | 0.14 | 0.16 | 0.06 | 0.32 | 0.38 |
| every night, finish by departure | 0.05 | 0.14 | 0.22 | 0.05 | 0.29 | 0.35 |
| some nights, 11 pm timer | 0.62 | 0.95 | 0.89 | 0.63 | 0.83 | 0.96 |
| some nights, evening | 0.76 | 0.97 | 0.90 | 0.77 | 0.86 | 0.99 |

With the original median, **94–95% of every-night sessions are invisible.** Part of the gap cannot be closed with hourly data: about 40% of every-night Level 2 top-ups last only one hour, shorter than the 2-hour block rule.

**Step 2: are the homes found?** Same features and clustering as above (k = 7):

| Baseline + clustering | Every-night EVs found | Some-nights EVs found | False alarms | F1 (all 120 EVs) |
|---|---|---|---|---|
| median + K-Means (original) | 14 / 30 | 88 / 90 | 19 | 0.85 |
| median + GMM (original) | 22 / 30 | 90 / 90 | 34 | 0.84 |
| p20 + K-Means | **30 / 30** | 90 / 90 | 22 | **0.92** |
| p20 + GMM | 0 / 30 | 61 / 90 | 0 | 0.67 |
| quiet_nights + K-Means | 19 / 30 | 86 / 90 | 18 | 0.86 |
| quiet_nights + GMM | 4 / 30 | 88 / 90 | 3 | 0.86 |
| rolling + K-Means | 10 / 30 | 88 / 90 | 19 | 0.83 |
| rolling + GMM | 5 / 30 | 62 / 90 | 13 | 0.67 |
| edges + K-Means | 29 / 30 | 78 / 90 | 23 | 0.86 |
| edges + GMM | 8 / 30 | 73 / 90 | 13 | 0.76 |
| p20_edges + K-Means | 28 / 30 | 90 / 90 | 20 | 0.92 |
| p20_edges + GMM | 22 / 30 | 78 / 90 | 21 | 0.83 |

What this says:

* The original method finds every-night chargers only through their few long (heavy-driving) nights: K-Means misses 16 of 30, the GMM misses 8 of 30.
* **Edge detection** makes the most every-night sessions visible (about 30%, vs 6%), because it does not care how often a block happens. The cost is weaker Level 1 detection (slow 1.3–1.9 kW starts are harder to tell from normal evening ramps).
* A **20th-percentile baseline** gave the best single result (p20 + K-Means: all 30 every-night chargers, F1 0.92), but the same features with a GMM fell apart (0 of 30). Combining it with edge detection (`p20_edges`) makes the most sessions visible of all baselines (35–38% of every-night sessions, 96–99% of some-nights sessions). The full tables are in `reports/baseline_comparison*.csv` and `reports/baseline_session_visibility.csv`.
* Most of the remaining false alarms are hot tubs, which, like every-night chargers, add a block of load almost every night.

### Is one clustering run enough? (stability check)

The tables above come from **one** setting (k = 7, seed 0), and the K-Means/GMM gap suggested that one run can be lucky or unlucky. `src/stability.py` repeats the clustering for every baseline over **k = 5–9 and seeds 0–4** (25 settings × K-Means and GMM = 50 runs) and gives each home an **EV score**: the share of runs that flag it. A home is flagged when its score is at least 0.5 (majority vote).

Spread over the 25 settings (every-night chargers found, out of 30):

| Baseline + clustering | F1 min – median – max | Every-night found min – median – max | Homes flagged min – max |
|---|---|---|---|
| median + K-Means | 0.83 – 0.84 – 0.86 | 12 – 14 – 19 | 117 – 129 |
| median + GMM | 0.67 – 0.79 – 0.92 | 5 – 12 – 30 | 80 – 146 |
| p20 + K-Means | 0.76 – 0.90 – 0.92 | 13 – 27 – 30 | 110 – 142 |
| p20 + GMM | 0.66 – 0.76 – 0.92 | 0 – 10 – 30 | 61 – 147 |
| quiet_nights + K-Means | 0.75 – 0.86 – 0.89 | 18 – 20 – 24 | 100 – 129 |
| quiet_nights + GMM | 0.66 – 0.83 – 0.89 | 0 – 22 – 30 | 63 – 168 |
| rolling + K-Means | 0.68 – 0.82 – 0.86 | 5 – 10 – 18 | 81 – 127 |
| rolling + GMM | 0.67 – 0.70 – 0.77 | 0 – 5 – 23 | 62 – 100 |
| edges + K-Means | 0.83 – 0.86 – 0.89 | 19 – 29 – 29 | 113 – 137 |
| edges + GMM | 0.68 – 0.76 – 0.88 | 0 – 9 – 30 | 62 – 152 |
| **p20_edges + K-Means** | **0.83 – 0.91 – 0.92** | **21 – 28 – 29** | 120 – 144 |
| p20_edges + GMM | 0.61 – 0.83 – 0.88 | 0 – 21 – 28 | 68 – 134 |

Majority vote on the EV score:

| Baseline | Vote over all 50 runs: F1 / every-night | K-Means runs only: F1 / every-night | GMM runs only: F1 / every-night |
|---|---|---|---|
| median | 0.85 / 13 | 0.85 / 14 | 0.85 / 14 |
| p20 | 0.86 / 16 | **0.92 / 30** | 0.76 / 10 |
| quiet_nights | 0.86 / 19 | 0.86 / 19 | 0.88 / 23 |
| rolling | 0.68 / 5 | 0.83 / 10 | 0.67 / 5 |
| edges | 0.84 / 19 | 0.86 / 29 | 0.76 / 9 |
| p20_edges | 0.84 / 23 | **0.91 / 28** | 0.83 / 21 |

What this says:

* **The GMM is the unstable part.** For every baseline, the number of every-night chargers it finds ranges from 0 (5 for the median baseline) to 23–30 depending on k and the seed. K-Means with 50 restarts hardly depends on the seed (F1 changes by at most 0.03 at a fixed k); its spread comes from the choice of k.
* **p20_edges + K-Means is the steadiest strong option**: F1 0.83–0.92 across all settings (median 0.91) and at least 21 of 30 every-night chargers in every run. p20 + K-Means has the best single result but drops to F1 0.76 and 13 every-night chargers at k = 5. Its K-Means vote finds 28 of 30 every-night chargers and all 90 some-nights chargers, with 21 false alarms (F1 0.91).
* Voting over K-Means **and** GMM runs does not help: the unstable GMM runs pull the vote down. The vote is useful within one algorithm, as a way to stop depending on one k.
* The p20_edges K-Means vote flags 139 homes for 120 real EV homes: it finds 118 of the 120, and all 21 false alarms are hot tubs (21 of 22). So the **count** of EV homes is over-estimated by about the number of hot tubs; separating EVs from hot tubs is the main remaining problem.
* Chosen default: **`p20_edges` features, K-Means, majority vote over k = 5–9**. `detect.py` now uses it (see [Results](#results-630-homes-120-with-an-ev)); the first-version median runs are kept there as comparison rows.

Outputs: `reports/stability_summary.csv`, `reports/stability_vote.csv`, `reports/stability_runs.csv` (every run) and `reports/stability_home_scores.csv` (EV score of every home).

## Run it

```bash
pip install -r requirements.txt
./run_all.sh          # ~2 minutes: generate data, cluster, compare baselines, check stability, write reports/
```

Outputs in `reports/`: `results.csv`, `results_by_home_type.csv`, `home_predictions.csv` (EV score and flag for every home), `cluster_profiles_kmeans.csv` and `cluster_members_kmeans.csv` (default features, k = 7), `cluster_profiles_gmm.csv` (first-version GMM), `home_features.csv`, `baseline_comparison.csv`, `baseline_comparison_by_type.csv`, `baseline_session_visibility.csv`, `stability_summary.csv`, `stability_vote.csv`, `stability_runs.csv`, `stability_home_scores.csv`, figures.

`python3 src/detect.py --baseline median` runs the main pipeline with another baseline; `--threshold 0.8` makes the vote stricter.

## Next steps

* **Session-level detection**: count charging sessions and kWh per home (useful for grid planning, e.g. how much EV load lands at 11 pm).
* Add features for **energy per block** and **night-to-night variability** to split EVs from hot tubs (all 21 remaining false alarms).
* Try **DBSCAN/HDBSCAN** (no need to choose *k*) and **semi-supervised** labelling with a few known EV owners.
* Test on **15-minute** data, where charger on/off edges are much sharper.
* Real-data issues: missing reads, daylight-saving shifts, solar panels (negative net load), multiple EVs per home.
