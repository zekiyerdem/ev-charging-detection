"""Generate synthetic hourly smart meter data for Ontario-style households.

Each home gets a realistic base load (daily routine, weekday/weekend, seasonal
heating and cooling driven by a synthetic Toronto temperature series, appliance
spikes). About 15% of homes own an electric vehicle and charge it at home with a
Level 1 (~1.4-1.9 kW, long sessions) or Level 2 (~6-7.7 kW, short sessions)
charger. Some homes without an EV get "look-alike" loads (electric water heater,
hot tub, baseboard heating) so the problem is not too easy.

On top of the main population, a set of extra "every-night" charging homes is
added (default 30). They plug in on most nights and top up only what the day's
driving used (typically 1-3 hours on Level 2), starting at 11 pm on a timer or
finishing at a set departure time. Because they charge on most nights, their
charging becomes part of their own "typical" night, which makes them hard to
find. They use their own random stream, so the first --homes homes are the same
with or without them.

Outputs (in data/):
  meter_readings.csv.gz  home_id, timestamp, kwh   (one row per home per hour)
  homes.csv              ground-truth labels per home (has_ev, charger_level,
                         charging_pattern, look-alikes)
  ev_sessions.csv        ground-truth charging sessions (home_id, start, hours, kw)
  weather.csv            hourly temperature used to drive heating/cooling

--hard makes the hot tub look-alikes harder (stress test): every third Level 2 charger
(by home number) becomes a 16 A unit drawing 3.3-3.8 kW instead of 6-7.7 kW, and hot tub
heaters draw 4.5-6.5 kW instead of 3-5 kW. The same random draws are reused (only
rescaled), so every home is otherwise identical to the normal data.
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


def low_power_l2(home_id, kw, hard):
    """--hard: every third Level 2 home (by number) gets a 16 A charger, 6-7.7 kW rescaled to 3.3-3.8 kW."""
    if hard and int(home_id[1:]) % 3 == 0:
        return 3.3 + (kw - 6.0) / 1.7 * 0.5
    return kw


def simulate_home(home_id, index, temp, rng, ev_share, lookalike_share, hard=False):
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

    info = {"home_id": home_id, "has_ev": 0, "charger_level": "", "lookalike": "", "charge_timing": ""}
    sessions = []

    if rng.random() < ev_share:
        info["has_ev"] = 1
        level = "L2" if rng.random() < 0.7 else "L1"
        info["charger_level"] = level
        kw = rng.uniform(6.0, 7.7) if level == "L2" else rng.uniform(1.3, 1.9)
        if level == "L2":
            kw = low_power_l2(home_id, kw, hard)
        sessions_per_week = rng.uniform(1.5, 5.5)
        # many Ontario drivers set the charger to start at 11 pm (Ultra-Low Overnight price)
        uses_timer = rng.random() < 0.55
        info["charge_timing"] = "timer_23h" if uses_timer else "evening"
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
            if hard:                    # bigger heaters: 3-5 kW rescaled to 4.5-6.5 kW
                kw += 1.5
            mask = ((hour >= 0) & (hour <= 5)) & (rng.random(n) < 0.35)
            load += mask * kw
        else:                           # electric baseboard: big, temperature-driven, winter only
            load += rng.uniform(0.25, 0.45) * np.clip(12 - temp, 0, None)

    return np.round(np.clip(load, 0.05, None), 3), info, sessions


def add_every_night_charging(home_id, index, load, rng, hard=False):
    """Every-night (or most-nights) charging: short top-ups sized by the day's driving.

    Modifies load in place and returns (info updates, sessions)."""
    n = len(index)
    level = "L2" if rng.random() < 0.8 else "L1"
    kw = rng.uniform(6.0, 7.7) if level == "L2" else rng.uniform(1.3, 1.9)
    if level == "L2":
        kw = low_power_l2(home_id, kw, hard)
    nights_share = rng.uniform(0.85, 1.0)                # plugs in on 85-100% of nights
    daily_kwh = rng.uniform(5, 12)                       # typical daily driving (~30-75 km)
    mode = "timer_23h" if rng.random() < 0.5 else "departure"
    departure = int(rng.choice([6, 7]))                  # car ready by 6 or 7 am
    sessions = []
    days = pd.DatetimeIndex(np.unique(index.normalize()))
    for d in days:
        if rng.random() > nights_share:
            continue
        energy = float(np.clip(daily_kwh * rng.lognormal(0, 0.45), 1.5, 30))  # varies with the day's driving
        energy = min(energy, kw * 14)                    # a session never runs past 14 h
        full_hours, rest = int(energy // kw), energy % kw
        hours = full_hours + (1 if rest > 0.05 else 0)
        if mode == "timer_23h":
            start = d + pd.Timedelta(hours=23)
        else:                                           # finish at departure the next morning
            start = d + pd.Timedelta(days=1, hours=departure) - pd.Timedelta(hours=hours)
        pos = index.searchsorted(start)
        if pos >= n:
            continue
        profile = [kw] * full_hours + ([rest] if rest > 0.05 else [])
        if mode == "departure":                         # the partial hour comes first
            profile = profile[::-1]
        for j, e in enumerate(profile):
            if pos + j < n:
                load[pos + j] += e
        sessions.append({"home_id": home_id, "start": start, "hours": hours, "kw": round(kw, 2)})
    info = {"has_ev": 1, "charger_level": level, "charging_pattern": "every_night", "charge_timing": mode}
    return info, sessions


def main(n_homes=600, days=365, ev_share=0.15, lookalike_share=0.15, seed=42, out="data",
         every_night_homes=30, hard=False):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2025-01-01", periods=days * 24, freq="h")
    temp = make_temperature(index, rng)

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    frames, infos, all_sessions = [], [], []
    for i in range(n_homes):
        home_id = f"H{i:04d}"
        kwh, info, sessions = simulate_home(home_id, index, temp, rng, ev_share, lookalike_share, hard)
        info["charging_pattern"] = "some_nights" if info["has_ev"] else ""
        frames.append(pd.DataFrame({"home_id": home_id, "timestamp": index, "kwh": kwh}))
        infos.append(info)
        all_sessions.extend(sessions)

    # extra every-night chargers, from their own random stream (main homes stay unchanged)
    rng_night = np.random.default_rng(seed + 1)
    for i in range(n_homes, n_homes + every_night_homes):
        home_id = f"H{i:04d}"
        load, info, _ = simulate_home(home_id, index, temp, rng_night, ev_share=0.0, lookalike_share=0.0)
        extra, sessions = add_every_night_charging(home_id, index, load, rng_night, hard)
        info.update(extra)
        frames.append(pd.DataFrame({"home_id": home_id, "timestamp": index, "kwh": np.round(load, 3)}))
        infos.append(info)
        all_sessions.extend(sessions)

    readings = pd.concat(frames, ignore_index=True)
    readings.to_csv(out / "meter_readings.csv.gz", index=False, compression="gzip")
    pd.DataFrame(infos).to_csv(out / "homes.csv", index=False)
    pd.DataFrame(all_sessions).to_csv(out / "ev_sessions.csv", index=False)
    pd.DataFrame({"timestamp": index, "temp_c": np.round(temp, 1)}).to_csv(out / "weather.csv", index=False)

    homes = pd.DataFrame(infos)
    print(f"Homes: {len(homes)} | with EV: {homes.has_ev.sum()} "
          f"(every-night chargers {int((homes.charging_pattern == 'every_night').sum())}) "
          f"(L2 {int((homes.charger_level == 'L2').sum())}, L1 {int((homes.charger_level == 'L1').sum())}) "
          f"| look-alikes: {int((homes.lookalike != '').sum())} | rows: {len(readings):,}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--homes", type=int, default=600)
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--ev-share", type=float, default=0.15)
    p.add_argument("--lookalike-share", type=float, default=0.15)
    p.add_argument("--every-night-homes", type=int, default=30,
                   help="extra homes that charge on most nights (added after --homes)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="data")
    p.add_argument("--hard", action="store_true",
                   help="stress test: 16 A (3.3-3.8 kW) Level 2 chargers in every third Level 2 home, 4.5-6.5 kW hot tubs")
    a = p.parse_args()
    main(a.homes, a.days, a.ev_share, a.lookalike_share, a.seed, a.out, a.every_night_homes, a.hard)
