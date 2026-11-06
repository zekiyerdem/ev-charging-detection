"""Turn hourly meter readings into per-home features that describe EV-like behaviour.

Key idea: home EV charging shows up as a *flat, high block* of extra power that
(1) appears suddenly, (2) lasts several hours, (3) mostly happens in the evening
or overnight, and (4) happens on some nights but not others. Normal household
use, heating and air conditioning do not look like that.

We compare each hour with the same home's *typical* load for that hour of day
(the median over the year), so every home is judged against its own baseline.
"""
import numpy as np
import pandas as pd


def to_cube(readings: pd.DataFrame):
    """Return (home_ids, days, cube) where cube has shape (homes, days, 24) in kWh."""
    df = readings.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["day"] = df["timestamp"].dt.normalize()
    df["hour"] = df["timestamp"].dt.hour
    home_ids = np.sort(df["home_id"].unique())
    days = np.sort(df["day"].unique())
    cube = (df.set_index(["home_id", "day", "hour"])["kwh"]
              .unstack("hour")
              .reindex(pd.MultiIndex.from_product([home_ids, days]))
              .to_numpy()
              .reshape(len(home_ids), len(days), 24))
    return home_ids, pd.DatetimeIndex(days), cube


def find_blocks(excess_series: np.ndarray, min_kw: float, min_hours: int):
    """Find runs where excess load stays >= min_kw for at least min_hours.

    Returns a list of (start_index, length, mean_kw)."""
    above = excess_series >= min_kw
    blocks, i, n = [], 0, len(above)
    while i < n:
        if above[i]:
            j = i
            while j < n and above[j]:
                j += 1
            if j - i >= min_hours:
                blocks.append((i, j - i, float(excess_series[i:j].mean())))
            i = j
        else:
            i += 1
    return blocks


def home_features(home_cube: np.ndarray, days: pd.DatetimeIndex, temp_by_day: np.ndarray):
    """Features for one home. home_cube: (days, 24) kWh."""
    flat = home_cube.reshape(-1)                         # continuous hourly series
    typical = np.nanmedian(home_cube, axis=0)            # typical kWh for each hour of day
    excess = (home_cube - typical).reshape(-1)           # extra load above the home's own normal
    hours_of_day = np.tile(np.arange(24), len(days))
    n_days = len(days)

    # Level-2-like blocks: >= 3.5 kW extra for >= 2 hours
    big = find_blocks(excess, 3.5, 2)
    # Level-1-like blocks: >= 1.1 kW extra for >= 4 hours
    small = find_blocks(excess, 1.1, 4)

    def evening_night_share(blocks):
        if not blocks:
            return 0.0
        starts = [hours_of_day[s] for s, _, _ in blocks]
        return float(np.mean([(h >= 16) or (h <= 2) for h in starts]))

    def flatness(blocks):
        """How flat the blocks are: low coefficient of variation inside a block = charger-like."""
        if not blocks:
            return 0.0
        cvs = []
        for s, length, _ in blocks:
            seg = excess[s:s + length - 1] if length > 2 else excess[s:s + length]
            cvs.append(np.std(seg) / (np.mean(seg) + 1e-6))
        return float(1.0 / (1.0 + np.median(cvs)))

    def start_spread(blocks):
        """Spread (std, hours) of block start times. Chargers on a timer or after the
        commute start at a consistent time; cycling appliances do not."""
        if len(blocks) < 3:
            return 6.0
        starts = np.array([hours_of_day[s] for s, _, _ in blocks])
        ang = 2 * np.pi * starts / 24                      # circular: 23:00 and 00:00 are close
        r = np.hypot(np.cos(ang).mean(), np.sin(ang).mean())
        return float(np.sqrt(-2 * np.log(max(r, 1e-6))) * 24 / (2 * np.pi))

    months = days.month.values
    def winter_share(blocks):
        """Share of blocks in Dec-Feb. EV charging happens all year; electric heating is winter-only."""
        if not blocks:
            return 0.0
        m = np.array([months[s // 24] for s, _, _ in blocks])
        return float(np.isin(m, [12, 1, 2]).mean())

    # nightly energy (6 pm to 6 am) and how "on/off" it is from night to night
    night = np.concatenate([home_cube[:, 18:], np.roll(home_cube, -1, axis=0)[:, :6]], axis=1).sum(axis=1)
    night_med = np.median(night)
    high_nights = float(np.mean(night > night_med + 6.0))         # nights with a big extra chunk of energy
    temp_corr = float(np.corrcoef(night, temp_by_day)[0, 1]) if np.std(night) > 0 else 0.0

    return {
        "big_block_rate": len(big) / n_days,              # L2-style blocks per day
        "big_block_hours": np.mean([b[1] for b in big]) if big else 0.0,
        "big_block_kw": np.mean([b[2] for b in big]) if big else 0.0,
        "big_block_evening_share": evening_night_share(big),
        "big_block_flatness": flatness(big),
        "small_block_rate": len(small) / n_days,          # L1-style blocks per day
        "small_block_evening_share": evening_night_share(small),
        "big_block_start_spread": start_spread(big),
        "small_block_start_spread": start_spread(small),
        "block_winter_share": winter_share(big + small),
        "high_night_share": high_nights,                  # intermittent extra night energy
        "night_temp_corr": temp_corr,                     # heating loads correlate strongly with cold
        "p99_kw": float(np.nanpercentile(flat, 99)),
        "mean_kw": float(np.nanmean(flat)),
    }


def build_features(readings: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    home_ids, days, cube = to_cube(readings)
    w = weather.copy()
    w["day"] = pd.to_datetime(w["timestamp"]).dt.normalize()
    temp_by_day = w.groupby("day")["temp_c"].mean().reindex(days).to_numpy()
    rows = []
    for k, hid in enumerate(home_ids):
        f = home_features(cube[k], days, temp_by_day)
        f["home_id"] = hid
        rows.append(f)
    return pd.DataFrame(rows).set_index("home_id")


def average_profiles(readings: pd.DataFrame) -> pd.DataFrame:
    """Baseline representation: each home's average 24-hour shape, normalised to sum to 1."""
    df = readings.copy()
    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
    prof = df.groupby(["home_id", "hour"])["kwh"].mean().unstack("hour")
    return prof.div(prof.sum(axis=1), axis=0)
