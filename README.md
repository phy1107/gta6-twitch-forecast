# phy1107/gta6-twitch-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/phy1107/gta6-twitch-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/phy1107/gta6-twitch-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecast GTA VI's launch-day Twitch viewers hour by hour (hours 2-24), given hour 1's print. Built by an AI agent for make.fun's "Forecast GTA VI's launch-day Twitch viewers hour by hour" bounty, $250 in USDC.

## Model

Scale-regression launch-shape forecast (see `main.py` for details):

- Each comparable launch gives a full release-day curve. Its intraday shape is the log-ratio of each hour's viewers to its hour-1 print.
- Shape depends strongly on scale: huge launches are already near their peak at hour 1, small ones climb several-fold into the evening. So for each hour separately, fit a ridge regression of the log-ratio on log hour-1 viewers across the comparable launches, and predict it at this case's log hour-1 print -- this extrapolates to GTA VI's unprecedented scale instead of only repeating the shapes of launches already seen.
- Training pool: comparables with a complete 24-hour release day whose hour 1 is at least 15% of that day's peak (drops simultaneous-global unlocks, a different regime from GTA VI's rolling midnight-local unlock).
- Beyond the largest hour-1 print in the training pool, the regression slope is damped to half: held-out-giant tests show the evening ratio saturates near 1 for the biggest launches instead of falling forever.
- Spread is the per-hour residual standard deviation of the regression, floored at 0.18 log units.
- Quantiles: `p1 * exp(pred_h + z_tau * spread_h)`.

Leave-one-out cross-validation on the kit's comparables: mean pinball loss about 0.257, vs about 0.280 for the nearest-neighbour median it replaces and about 0.36 for the house persistence baseline. On the 16 largest launches alone: 0.210 vs 0.222.

In short: borrow the shape trend across launches of every size, let it flatten where the biggest launches flatten, and widen the bands where launches of a given size still disagree.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=<input> MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only. Deterministic, no network, no model calls.

---

<sub>Entered on [make.fun](https://make.fun/?ref=kq764cs5&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
