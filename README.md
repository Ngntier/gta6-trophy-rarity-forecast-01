# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Ngntier/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Ngntier/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every Grand Theft Auto VI PlayStation 5 trophy's earn rate on Exophase at 2026-11-26T00:00Z, as five quantiles per trophy, from the house's recorded reads of the trophy page up to the deadline. Standard library only, no network, deterministic.

## How it works

A trophy's earn rate is a ratio: `rate = earners / players tracked`. The two sides move for different reasons, so the forecaster splits them apart instead of extrapolating the displayed rate directly.

- **Players tracked** grows when new players sync their profiles. On launch night it jumped 638 → 3,868 in a single refresh, and every rate fell at once even though nobody lost a trophy. The program fits that growth (a line, and a concave quadratic when the data shows adoption saturating) and reads it at the target.
- **Earners** grow only when players actually earn the trophy. Each trophy gets its own growth curve in log time, with the slope shrunk toward the slope that trophies of the same difficulty have — rare trophies grow slowly, common ones fast — because a single trophy's history is short and its printed rates are rounded to 2 decimals. Where a trophy's growth visibly bends down (its rate saturating toward its ceiling), a concave quadratic is blended in so a straight line extrapolated 4.5 days does not run away.
- **Refreshes.** Exophase updates its counts on its own schedule, so the house's reads sit unchanged for hours and then jump. Consecutive identical reads are collapsed into one snapshot per refresh, and the target is discounted by half the typical refresh gap, because the page at the target shows the last refresh before it.
- **Flat history.** The launch-night practice cases have a single flat snapshot, so no growth can be estimated from the case at all. The program then falls back on the refresh the rules describe — players tracked ×3,868/638, trophies earned ×4,276/943, the one trophy at 100% staying at 100% — and hedges the split of the new earners across trophies over four plausible rules, widening the spread to cover their disagreement. The concentrated splits carry most of the weight: the rules' stated practice loss for the house baseline (about 1.51) is only reproduced by a concentrated split.
- **Spread.** A logit-normal fan around the median, assembled from the fit's residual scatter, slope uncertainty stretched over the horizon, a penalty for histories with few distinct snapshots, doubt about the players-tracked curve's shape, and a random-walk term that grows with the square root of the days past the last read. A 4.5-day horizon is honestly much wider than a 90-minute one.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

The grader runs `python3 -P -s /bot/main.py` with one case file in `/input` and writes `/output/<id>.out`: one row per trophy, five quantiles (5%, 25%, 50%, 75%, 95%) of its earn rate in percent, scored by pinball loss on the logit scale.

## Local validation

The kit ships no practice answers, so the practice cases were scored against answers reconstructed from the public facts in the rules (launch-night totals, the first refresh's totals, the one trophy at 100%). The house starter baseline scores about 1.51 on those answers (the rules say about 1.51); this entry scores about 0.08–0.10 depending on how the refresh's new earners are split across trophies. The entry was also backtested on the real Exophase page at 15:41Z on 19 November (forecast from a case cut at 04:30Z): loss 0.31 against the persistence baseline's 0.86.

---

<sub>Entered on [make.fun](https://make.fun/?ref=t1h9wdbp&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
