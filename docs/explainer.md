# Plain-language explainer (interview prep)

## What problem is this?
Utilities and grid operators want to know **where EVs are charging at home**, because many cars charging at the same time (for example right at 11 pm when the overnight price starts) can overload local transformers and change province-wide demand. Customers don't report owning an EV, so we try to infer it from the meter.

## Why clustering (unsupervised)?
We usually **don't have labels** (who owns an EV). Clustering groups homes with similar behaviour, then an analyst looks at each group's typical home and decides what it is. If a few labels exist (e.g. from a rebate program), they can be used to check or improve the clusters.

## Why not just cluster the daily load curve?
Averaging a year of data hides charging that happens only on some nights, and the timing varies. That's why the baseline only found 44% of EV homes. Building **features that describe the physical event** (a flat block of extra power) works much better.

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

## How I chose and checked the model
* Standardised features so no single unit dominates.
* Compared K-Means (hard, round clusters) with a Gaussian Mixture Model (soft clusters + a probability per home).
* Looked at silhouette scores, but chose k by **whether clusters were interpretable**, then validated against the hidden labels: precision, recall, F1, adjusted Rand index.

## Talking points
* "I'd start with physically meaningful features, not raw curves, because charging is intermittent."
* "The hard part is look-alikes like hot tubs and electric heating, so I add features on season, temperature and start-time consistency."
* "Clustering gives candidate groups; domain rules or a small labelled sample confirm them."
* "With 15-minute data, the on/off edges are much sharper and Level 2 detection gets easier."
* "Next, I'd estimate sessions and kWh per home, which is what grid planning needs: how much EV load lands at what hour."
