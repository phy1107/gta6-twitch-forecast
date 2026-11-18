"""gta6-twitch-ccu-forecast-01 entry (Python 3.13, standard library only).

Model: scale-regression launch-shape forecast.
  - Every comparable launch in the case gives a full release-day curve.
    For hour h, its log-ratio to its own hour-1 print, log(v_h / v_1),
    is that launch's intraday shape.
  - The shape depends strongly on scale: log(hour-1 viewers) correlates
    about -0.73 with the log evening ratio log(v_19 / v_1) across
    comparables -- huge launches are already near their peak at hour 1
    (their evening ratio is near 1), small ones climb several-fold into
    the evening, when the US unlocks (05:00 / 08:00 UTC) and the
    European evening arrive.  A nearest-neighbour median can only repeat
    the shapes of launches already seen; it cannot say what happens at
    a scale beyond them, and GTA VI will very likely be beyond every
    comparable (the largest hour-1 print in the kit is ~875k).
    So, for each hour h separately, fit a ridge regression of the
    log-ratio on log hour-1 viewers across the comparable launches:
        log(v_h / v_1) ~ a_h + b_h * (log v_1 - mean log v_1)
    and predict it at this case's log hour-1 print.  The slope b_h is
    negative in the evening hours and near zero overnight, which is
    exactly the scale effect above, now estimated from all launches
    instead of the 10 nearest.
  - Training pool: comparables with a complete 24-hour release day,
    all prints > 0, and hour 1 at least 15% of that day's peak.  The
    15% cut drops simultaneous-global unlocks whose hour 1 is
    essentially zero (their shapes are a different regime: ratios of
    hundreds) and slow-burn launches whose hour 1 is a small fraction
    of the peak -- GTA VI unlocks rolling at midnight local, with the
    UK unlocking at 00:00 UTC and central Europe already live, so its
    hour 1 will be a large share of its peak.
  - Extrapolation: beyond the largest log hour-1 print in the training
    pool, the slope is damped to half.  Held-out-giant tests (train on
    launches with hour 1 <= 200k/400k/550k, predict the giants above)
    show free linear extrapolation overshoots -- the evening ratio
    saturates near 1 for the biggest launches instead of falling
    forever -- while half-slope extrapolation beats both free
    extrapolation and the nearest-neighbour median on every cut.
  - Spread: the per-hour residual standard deviation of the
    regression, floored at 0.18 log units.  The bands are honest about
    how much launches of a given size still disagree with each other.
  - Quantiles: q_tau(h) = p1 * exp(pred_h + z_tau * spread_h),
    z = (-1.645, -0.674, 0, 0.674, 1.645), log-ratio predictions
    clamped to [-1.5, 1.5], rounded to 2 decimals half-up, clipped to
    (0, 1e9] and non-decreasing.

Leave-one-out cross-validation on the comparables in the kit
(29 rolling-unlock-like launches, each predicted from the others):
mean pinball loss 0.257, typical miss 54.5%, 90% coverage 0.83 --
against 0.280 / 61.8% / 0.86 for the nearest-neighbour median it
replaces, and ~0.36 for the house persistence baseline.  On the
16 largest launches alone: 0.210 against 0.222.  A paired bootstrap
over the 29 targets puts P(no improvement) at about 0.006.

Reads every <id>.in in $MAKEFUN_INPUT_DIR (default /input) and
writes <id>.out to $MAKEFUN_OUTPUT_DIR (default /output):
{"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]},
23 rows, hours 2..24.  Deterministic, no network, no model calls.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
MIN_HOUR1_FRACTION = 0.15  # hour 1 must be >= 15% of the day's peak
RIDGE = 1.0                # ridge penalty on the per-hour slope
DAMP = 0.5                 # slope multiplier beyond the training range
SPREAD_FLOOR = 0.18        # minimum log-spread per hour
LOG_RATIO_CLAMP = 1.5      # predictions clamped to [-1.5, 1.5]


def _parse(t):
    return datetime.fromisoformat(t.replace("Z", "+00:00"))


def round2(x):
    """Round to 2 decimals, half up (as the starter does)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value == int(value) else value


def release_day_rows(comp):
    """Hour -> viewers for the comparable's UTC release day, or None."""
    try:
        day_start = _parse(comp["day_start"])
    except (KeyError, ValueError):
        return None
    rows = {}
    for point in comp.get("hourly", []):
        try:
            t = _parse(point["from"])
            viewers = point["viewers"]
        except (KeyError, ValueError):
            continue
        if viewers is None:
            continue
        if day_start <= t < day_start + timedelta(days=1):
            rows[(t - day_start).seconds // 3600 + 1] = viewers
    if len(rows) < 24:
        return None
    if any(v is None or v <= 0 for v in rows.values()):
        return None
    return rows


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def fit_shape(pool, log_p1):
    """Per-hour (predicted log-ratio, log-spread) at log hour-1 print.

    pool: list of release-day rows dicts (complete days, hour 1 not
    tiny).  One ridge regression per hour of log(v_h / v_1) on
    log(v_1), centred on the training mean; slope damped past the
    largest training log hour-1 print.  Spread is the residual
    standard deviation around that hour's fit.
    """
    xs = [math.log(rows[1]) for rows in pool]
    mean_x = sum(xs) / len(xs)
    max_x = max(xs)
    sxx = sum((x - mean_x) ** 2 for x in xs) + RIDGE
    shape = {}
    spread = {}
    for h in range(2, 25):
        ys = [math.log(rows[h] / rows[1]) for rows in pool]
        mean_y = sum(ys) / len(ys)
        slope = sum((x - mean_x) * (y - mean_y)
                    for x, y in zip(xs, ys)) / sxx
        if log_p1 > max_x:
            # Half-slope extrapolation: the scale trend saturates.
            pred = (mean_y + slope * (max_x - mean_x)
                    + slope * DAMP * (log_p1 - max_x))
        else:
            pred = mean_y + slope * (log_p1 - mean_x)
        residuals = [y - (mean_y + slope * (x - mean_x))
                     for x, y in zip(xs, ys)]
        dof = max(1, len(residuals) - 2)
        sigma = math.sqrt(sum(r * r for r in residuals) / dof)
        shape[h] = max(-LOG_RATIO_CLAMP, min(LOG_RATIO_CLAMP, pred))
        spread[h] = max(SPREAD_FLOOR, sigma)
    return shape, spread


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])
    log_p1 = math.log(p1)

    # Training pool: complete release days whose hour 1 is a real
    # share of the peak (rolling-midnight regime, like GTA VI).
    pool = []
    for comp in case.get("comparables", []):
        rows = release_day_rows(comp)
        if rows is None:
            continue
        if rows[1] < MIN_HOUR1_FRACTION * max(rows.values()):
            continue
        pool.append(rows)

    if len(pool) < 5:
        # Fall back: any complete release day, then persistence.
        pool = []
        for comp in case.get("comparables", []):
            rows = release_day_rows(comp)
            if rows is not None:
                pool.append(rows)
    if len(pool) < 5:
        shape = {h: 0.0 for h in range(2, 25)}
        spread = {h: 0.15 * math.sqrt(h - 1) for h in range(2, 25)}
    else:
        shape, spread = fit_shape(pool, log_p1)

    hours = []
    for row in case["forecast"]:
        h = row["hour"]
        log_med = log_p1 + shape[h]
        sigma = spread[h]
        quantiles = []
        prev = 0.0
        for z in Z:
            q = math.exp(log_med + z * sigma)
            q = round2(min(1e9, max(1.0, q)))
            if q < prev:
                q = prev
            prev = q
            quantiles.append(q)
        hours.append({"hour": h, "q": quantiles})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
