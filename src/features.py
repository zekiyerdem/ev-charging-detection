"""Turn hourly meter readings into per-home features that describe EV-like behaviour.

Key idea: home EV charging shows up as a *flat, high block* of extra power that
(1) appears suddenly, (2) lasts several hours, (3) mostly happens in the evening
or overnight, and (4) happens on some nights but not others. Normal household
use, heating and air conditioning do not look like that.

We compare each hour with the same home's *typical* load for that hour of day,
so every home is judged against its own baseline. The default typical load is the
median over the year for each hour of day. That fails for homes that charge on
most nights: the charging becomes part of the median and the blocks vanish.
`BASELINES` lists the alternatives compared in compare_baselines.py.
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


BASELINES = {
    "median": "median of each hour of day over the year (original method)",
    "p20": "20th percentile of each hour of day over the year",
    "quiet_nights": "median of each hour of day over quiet nights only (no big night load)",
    "rolling": "rolling 29-day median of each hour of day (follows the seasons)",
    "edges": "no baseline: blocks start with a jump and end with a matching drop",
}


def typical_load(home_cube: np.ndarray, method: str = "median") -> np.ndarray:
    """Typical kWh for every day and hour, shape (days, 24), for one home."""
    n_days = home_cube.shape[0]
    if method == "median":
        typ = np.nanmedian(home_cube, axis=0)
    elif method == "p20":
        typ = np.nanpercentile(home_cube, 20, axis=0)
    elif method == "quiet_nights":
        # a night is "quiet" when its largest hour (22:00-07:00) is close to the home's
        # quietest nights; charging nights are left out of the baseline
        night_max = np.nanmax(home_cube[:, list(range(0, 8)) + [22, 23]], axis=1)
        quiet = night_max <= np.nanpercentile(night_max, 10) + 1.0
        if quiet.sum() >= 10:
            typ = np.nanmedian(home_cube[quiet], axis=0)
        else:                                            # too few quiet nights: fall back
            typ = np.nanpercentile(home_cube, 10, axis=0)
    elif method == "rolling":
        typ = (pd.DataFrame(home_cube)
                 .rolling(29, center=True, min_periods=8).median()
                 .to_numpy())
        return typ
    else:
        raise ValueError(f"unknown baseline {method!r}; choose from {list(BASELINES)}")
    return np.broadcast_to(typ, (n_days, 24))


def edge_excess(flat: np.ndarray, min_jump: float = 1.1, max_hours: int = 14) -> np.ndarray:
    """Excess load from on/off edges, with no baseline at all.

    A block starts where load jumps up by >= min_jump kW from one hour to the next and
    ends at the first hour (within max_hours) where load falls back below the level
    before the jump plus half the jump. Inside the block, excess = load minus the level
    before the jump. Everything else is 0."""
    ex = np.zeros_like(flat, dtype=float)
    n, t = len(flat), 1
    while t < n:
        jump = flat[t] - flat[t - 1]
        if jump >= min_jump:
            level0 = flat[t - 1]
            end = next((t2 for t2 in range(t + 1, min(t + max_hours + 1, n))
                        if flat[t2] < level0 + 0.5 * jump), None)
            if end is not None:
                ex[t:end] = flat[t:end] - level0
                t = end
                continue
        t += 1
    return ex


def excess_series(home_cube: np.ndarray, baseline: str = "median") -> np.ndarray:
    """Hourly extra load (kWh) above the home's typical load, as one flat series."""
    if baseline == "edges":
        return edge_excess(home_cube.reshape(-1))
    return (home_cube - typical_load(home_cube, baseline)).reshape(-1)


def charging_blocks(excess: np.ndarray):
    """Level-2-like blocks (>= 3.5 kW for >= 2 h) and Level-1-like blocks (>= 1.1 kW for >= 4 h)."""
    return find_blocks(excess, 3.5, 2), find_blocks(excess, 1.1, 4)


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


def home_features(home_cube: np.ndarray, days: pd.DatetimeIndex, temp_by_day: np.ndarray,
                  baseline: str = "median"):
    """Features for one home. home_cube: (days, 24) kWh. baseline: a key of BASELINES."""
    flat = home_cube.reshape(-1)                         # continuous hourly series
    excess = excess_series(home_cube, baseline)          # extra load above the home's own normal
    hours_of_day = np.tile(np.arange(24), len(days))
    n_days = len(days)

    # Level-2-like blocks (>= 3.5 kW for >= 2 h) and Level-1-like blocks (>= 1.1 kW for >= 4 h)
    big, small = charging_blocks(excess)

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
        r = min(max(r, 1e-6), 1.0)                         # all-identical starts can give r = 1 + rounding
        return float(np.sqrt(-2 * np.log(r)) * 24 / (2 * np.pi))

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


def build_features(readings: pd.DataFrame, weather: pd.DataFrame, baseline: str = "median") -> pd.DataFrame:
    home_ids, days, cube = to_cube(readings)
    w = weather.copy()
    w["day"] = pd.to_datetime(w["timestamp"]).dt.normalize()
    temp_by_day = w.groupby("day")["temp_c"].mean().reindex(days).to_numpy()
    rows = []
    for k, hid in enumerate(home_ids):
        f = home_features(cube[k], days, temp_by_day, baseline)
        f["home_id"] = hid
        rows.append(f)
    return pd.DataFrame(rows).set_index("home_id")


def average_profiles(readings: pd.DataFrame) -> pd.DataFrame:
    """Baseline representation: each home's average 24-hour shape, normalised to sum to 1."""
    df = readings.copy()
    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
    prof = df.groupby(["home_id", "hour"])["kwh"].mean().unstack("hour")
    return prof.div(prof.sum(axis=1), axis=0)
