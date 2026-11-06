"""Generate synthetic hourly smart meter data for Ontario-style households.

Each home gets a realistic base load (daily routine, weekday/weekend, seasonal
heating and cooling driven by a synthetic Toronto temperature series, appliance
spikes). About 15% of homes own an electric vehicle and charge it at home with a
Level 1 (~1.4-1.9 kW, long sessions) or Level 2 (~6-7.7 kW, short sessions)
charger. Some homes without an EV get "look-alike" loads (electric water heater,
hot tub, baseboard heating) so the problem is not too easy.

Outputs (in data/):
  meter_readings.csv.gz  home_id, timestamp, kwh   (one row per home per hour)
  homes.csv              ground-truth labels per home (has_ev, charger_level, look-alikes)
  ev_sessions.csv        ground-truth charging sessions (home_id, start, hours, kw)
  weather.csv            hourly temperature used to drive heating/cooling
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make_temperature(index, rng):
    """Toronto-like hourly temperature: seasonal cycle + daily cycle + weather noise."""
    doy = index.dayofyear.values
    hour = index.hour.values
    seasonal = 8.5 - 14.0 * np.cos(2 * np.pi * (doy - 20) / 365.25)   # ~ -5.5C Jan, ~22C Jul
    daily = 4.0 * np.sin(2 * np.pi * (hour - 9) / 24)                # warmest mid-afternoon
    # slow-moving weather fronts (AR(1) noise)
    noise = np.zeros(len(index))
    eps = rng.normal(0, 0.6, len(index))
    for i in range(1, len(index)):
        noise[i] = 0.97 * noise[i - 1] + eps[i]
    return seasonal + daily + noise


def base_profile(rng):
    """24-hour shape of a household's routine (kW), with morning and evening peaks."""
    h = np.arange(24)
    morning_peak = rng.uniform(6, 8.5)
    evening_peak = rng.uniform(17.5, 20.5)
    shape = (
        0.35
        + rng.uniform(0.3, 0.8) * np.exp(-0.5 * ((h - morning_peak) / 1.2) ** 2)
        + rng.uniform(0.7, 1.6) * np.exp(-0.5 * ((h - evening_peak) / 2.0) ** 2)
    )
    return shape * rng.uniform(0.7, 1.4)


def simulate_home(home_id, index, temp, rng, ev_share, lookalike_share):
    n = len(index)
    hour = index.hour.values
    weekend = (index.dayofweek.values >= 5)

    prof = base_profile(rng)
    load = prof[hour].copy()
    # weekends: later mornings and more daytime use
    load[weekend] = np.roll(prof, 1)[hour[weekend]] * 1.1 + 0.2 * ((hour[weekend] >= 10) & (hour[weekend] <= 16))

    # heating (gas furnace fan or heat pump) and air conditioning
    heat_kw_per_deg = rng.uniform(0.0, 0.05)
    cool_kw_per_deg = rng.uniform(0.05, 0.18) if rng.random() < 0.8 else 0.0
    load += heat_kw_per_deg * np.clip(15 - temp, 0, None)
    load += cool_kw_per_deg * np.clip(temp - 22, 0, None) * ((hour >= 11) & (hour <= 22))

    # random appliance spikes (oven, dryer, dishwasher): short 1-hour bursts
    spikes = rng.random(n) < 0.04 * ((hour >= 7) & (hour <= 22))
    load += spikes * rng.uniform(1.0, 3.5, n)

    # measurement / behaviour noise
    load *= rng.lognormal(0, 0.12, n)

    info = {"home_id": home_id, "has_ev": 0, "charger_level": "", "lookalike": ""}
    sessions = []

    if rng.random() < ev_share:
        info["has_ev"] = 1
        level = "L2" if rng.random() < 0.7 else "L1"
        info["charger_level"] = level
        kw = rng.uniform(6.0, 7.7) if level == "L2" else rng.uniform(1.3, 1.9)
        sessions_per_week = rng.uniform(1.5, 5.5)
        # many Ontario drivers set the charger to start at 11 pm (Ultra-Low Overnight price)
        uses_timer = rng.random() < 0.55
        days = pd.DatetimeIndex(np.unique(index.normalize()))
        for d in days:
            if rng.random() > sessions_per_week / 7:
                continue
            if uses_timer:
                start = d + pd.Timedelta(hours=23)
            else:
                start = d + pd.Timedelta(hours=int(np.clip(rng.normal(18.5, 1.5), 16, 23)))
            energy = rng.uniform(8, 32)                    # kWh needed that night
            hours = int(np.clip(np.ceil(energy / kw), 1, 14))
            sessions.append({"home_id": home_id, "start": start, "hours": hours, "kw": round(kw, 2)})
            pos = index.searchsorted(start)
            end = min(pos + hours, n)
            load[pos:end] += kw
            # the last hour is usually partial (battery finishes)
            if end - 1 >= pos and hours > 1:
                load[end - 1] -= kw * rng.uniform(0.0, 0.6)

    elif rng.random() < lookalike_share / (1 - ev_share):
        kind = rng.choice(["water_heater", "hot_tub", "baseboard_heat"])
        info["lookalike"] = kind
        if kind == "water_heater":      # 3-4.5 kW element, 1-2 h after morning and evening use
            kw = rng.uniform(3.0, 4.5)
            mask = np.isin(hour, [7, 8, 19, 20]) & (rng.random(n) < 0.6)
            load += mask * kw
        elif kind == "hot_tub":         # 3-5 kW heater cycling at night, all year
            kw = rng.uniform(3.0, 5.0)
            mask = ((hour >= 0) & (hour <= 5)) & (rng.random(n) < 0.35)
            load += mask * kw
        else:                           # electric baseboard: big, temperature-driven, winter only
            load += rng.uniform(0.25, 0.45) * np.clip(12 - temp, 0, None)

    return np.round(np.clip(load, 0.05, None), 3), info, sessions


def main(n_homes=600, days=365, ev_share=0.15, lookalike_share=0.15, seed=42, out="data"):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2025-01-01", periods=days * 24, freq="h")
    temp = make_temperature(index, rng)

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    frames, infos, all_sessions = [], [], []
    for i in range(n_homes):
        home_id = f"H{i:04d}"
        kwh, info, sessions = simulate_home(home_id, index, temp, rng, ev_share, lookalike_share)
        frames.append(pd.DataFrame({"home_id": home_id, "timestamp": index, "kwh": kwh}))
        infos.append(info)
        all_sessions.extend(sessions)

    readings = pd.concat(frames, ignore_index=True)
    readings.to_csv(out / "meter_readings.csv.gz", index=False, compression="gzip")
    pd.DataFrame(infos).to_csv(out / "homes.csv", index=False)
    pd.DataFrame(all_sessions).to_csv(out / "ev_sessions.csv", index=False)
    pd.DataFrame({"timestamp": index, "temp_c": np.round(temp, 1)}).to_csv(out / "weather.csv", index=False)

    homes = pd.DataFrame(infos)
    print(f"Homes: {n_homes} | with EV: {homes.has_ev.sum()} "
          f"(L2 {int((homes.charger_level == 'L2').sum())}, L1 {int((homes.charger_level == 'L1').sum())}) "
          f"| look-alikes: {int((homes.lookalike != '').sum())} | rows: {len(readings):,}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--homes", type=int, default=600)
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--ev-share", type=float, default=0.15)
    p.add_argument("--lookalike-share", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="data")
    a = p.parse_args()
    main(a.homes, a.days, a.ev_share, a.lookalike_share, a.seed, a.out)
