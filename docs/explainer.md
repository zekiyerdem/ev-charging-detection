# Plain-language explainer (interview prep)

## What problem is this?
Utilities and grid operators want to know **where EVs are charging at home**, because many cars charging at the same time (for example right at 11 pm when the overnight price starts) can overload local transformers and change province-wide demand. Customers don't report owning an EV, so we try to infer it from the meter.

## Why clustering (unsupervised)?
We usually **don't have labels** (who owns an EV). Clustering groups homes with similar behaviour, then an analyst looks at each group's typical home and decides what it is. If a few labels exist (e.g. from a rebate program), they can be used to check or improve the clusters.

## Why not just cluster the daily load curve?
Averaging a year of data hides charging that happens only on some nights, and the timing varies. That's why the baseline only found 34% of EV homes. Building **features that describe the physical event** (a flat block of extra power) works much better.

## The features, in one line each
* `big_block_rate`: how often a ≥3.5 kW extra block lasting ≥2 h happens (Level 2).
* `small_block_rate`: how often a ≥1.1 kW block lasting ≥4 h happens (Level 1).
* `big_block_kw`, `big_block_hours`: typical power and length of those blocks.
* `*_evening_share`: share of blocks starting 4 pm–2 am.
* `*_start_spread`: how consistent the start time is (timers and commute habits = consistent).
* `big_block_flatness`: chargers draw constant power, so the block is flat.
* `block_winter_share`: heating is winter-only; EV charging is all year.
* `night_temp_corr`: if night use rises when it's cold, it's probably heating.
* `high_night_share`: share of nights with a big extra chunk of energy.
* `big_block_peak_kw`, `big_block_kwh`, `big_block_night_kwh_cv`: the highest hour and energy of a typical Level-2-like block, and how much block energy changes from night to night. Used for the charger power check, not the clustering.

## How I chose and checked the model
* Standardised features so no single unit dominates.
* Compared K-Means (hard, round clusters) with a Gaussian Mixture Model (soft clusters + a probability per home).
* Looked at silhouette scores, but chose k by **whether clusters were interpretable**, then validated against the hidden labels: precision, recall, F1, adjusted Rand index.
* Final decision: K-Means on the combined low-percentile + edge features, repeated over k = 5–9 and 5 seeds; a home is an EV home when at least half the runs say so. That finds 118 of 120 EV homes and flags 139; all 21 false alarms are hot tubs. A charger power check then un-flags homes whose regular Level-2-like blocks never reach 5.5 kW: all 21 hot tubs are removed and no EV is lost (118 flagged, precision 1.00, F1 0.99).

## Talking points
* "I'd start with physically meaningful features, not raw curves, because charging is intermittent."
* "The hard part is look-alikes like hot tubs and electric heating, so I add features on season, temperature and start-time consistency."
* "Clustering gives candidate groups; domain rules or a small labelled sample confirm them."
* "With 15-minute data, the on/off edges are much sharper and Level 2 detection gets easier."
* "Comparing each hour with the home's own median has a blind spot: someone who charges every night on the overnight rate has the charging *inside* their median. In my synthetic test 94–95% of those sessions became invisible. Edge detection (a sudden jump of several kW and a matching drop) or a low-percentile baseline brings many of them back, but the clustering step then needs to be made more stable."
* "A single clustering run can be lucky. I repeated it over k = 5–9 and five seeds: the GMM found anywhere from 0 to 30 of the every-night chargers depending on the setting, while K-Means on the combined low-percentile + edge features stayed at F1 0.83–0.92. A majority vote over those runs gives each home an EV score instead of depending on one k."
* "I also tried HDBSCAN, which needs no k. It swaps k for a minimum cluster size that moves F1 just as much (0.59–0.89), and it parks 18–51 EV homes in its noise group, because EV homes are spread out rather than one dense cluster. So it under-counts EVs; I kept K-Means with a vote over k."
* "Hot tubs look like Level 2 chargers to the clustering. What separates them is power: a charger draws 6–7.7 kW for whole hours, a hot tub heater 3–5 kW. Night-to-night variability did not help, because a thermostat cycles at random too. On real data I'd expect overlap (3.8 kW chargers, bigger hot tubs), so I'd check that with a few labelled homes."
* "Next, I'd estimate sessions and kWh per home, which is what grid planning needs: how much EV load lands at what hour."
