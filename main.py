"""GTA VI trophy rarity forecaster: earners / players decomposition.

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, five
quantiles of its earn rate on Exophase at a case's target, from the house's
recorded reads of the trophy page up to the case's checkpoint.

The core idea is a decomposition.  A trophy's earn rate is a ratio,
    rate_i(t) = E_i(t) / P(t),
where P(t) is the number of players Exophase tracks and E_i(t) is how many
of them hold trophy i.  The two move for different reasons and on different
clocks:

  * P(t) grows when new players sync their profiles.  On launch night it
    went 638 -> 3,868 in one refresh, and it keeps growing as the launch
    wave syncs.  A jump in P dilutes every rate at once, even though nobody
    lost a trophy.
  * E_i(t) grows only when players actually earn trophy i.  Easy, early
    trophies (the first fishing trip, chapter 1) are earned quickly by many
    players; late, grindy ones (100% completion, the platinum) barely move.

So instead of extrapolating the displayed rate directly -- which confounds
a dilution jump with real progress -- the program fits a growth curve to
each side separately and takes the ratio at the target:

  * ln P against x = ln(1 + hours since the release day began), one line
    for the whole game;
  * ln E_i against the same x, one line per trophy, with the slope shrunk
    toward the slope that trophies of the same difficulty have (a trophy's
    difficulty is read off its current rate: rare trophies grow slowly,
    common ones fast).  Shrinkage matters because a trophy's own history is
    short and its printed rates are rounded to 2 decimals, so a per-trophy
    slope fit on its own is mostly noise for the rare trophies.  Where a
    trophy's history shows its growth bending down -- earners growth
    saturates as the rate approaches its ceiling -- a concave quadratic is
    blended in, so a straight line extrapolated 4.5 days does not run away.

Exophase refreshes its counts on its own schedule, not continuously: the
house's reads sit unchanged for hours and then jump.  The program collapses
consecutive identical reads into one snapshot per refresh, fits on those,
and discounts the target by half the typical refresh gap, because the rate
the page shows at the target is the rate as of its last refresh before it.

When the history is a single flat snapshot -- the launch-night practice
cases, where every read is the same page -- no growth can be estimated from
the case at all.  The program then falls back on the refresh the rules
describe: the first refresh after launch multiplied players tracked by
3,868/638 and trophies earned by 4,276/943, the one trophy at 100% stayed
at 100% (every newly tracked player holds it), and the remaining new
earners went to the other trophies.  Exactly how those new earners split
across trophies is not pinned down by the public facts, so the fallback
averages four plausible splits (in proportion to old earners; nested by
difficulty, easiest first; and two concentrated power laws) and widens the
spread to cover their disagreement.  The concentrated splits carry most of
the weight: the rules' stated practice loss for the house baseline (about
1.51) is only reproduced by a concentrated split, while a purely
proportional split gives about 1.40.  That keeps the forecast honest about
what the case cannot tell it.

The spread is a logit-normal fan around the median, assembled from the
fit's residual scatter, the slope uncertainty stretched over the horizon,
a penalty when the history holds few distinct snapshots (two points pin a
line but say nothing about the next bend), and a random-walk term that
grows with the square root of the days past the last read -- so a 4.5-day
horizon is honestly much wider than a 90-minute one.  Trophies at (essentially) 100% are forecast at 99.99 with a narrow
fan on a flat history (the first refresh kept them at 100%) and a wider
one once real history exists.

Everything is deterministic: no randomness, no wall clock, no dict-order
dependence, no network.  Standard library only.

Usage (grader): python3 -P -s /bot/main.py, reading $MAKEFUN_INPUT_DIR
(default /input) and writing $MAKEFUN_OUTPUT_DIR (default /output).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# The first refresh after launch, as the competition rules describe it:
# players tracked 638 -> 3,868, trophies earned 943 -> 4,276.
REFRESH_PLAYERS = 3868.0 / 638.0
REFRESH_EARNERS = 4276.0 / 943.0

# Spread pieces (logit units).  See forecast_case for how they combine.
SIGMA_FIT_FLOOR = 0.15
SIGMA_FIT_CAP = 0.60
SIGMA_RW_PER_SQRT_DAY = 0.25
SIGMA_REFRESH = 0.15
SIGMA_MIN = 0.25
SIGMA_MAX = 1.20
SIGMA_FLAT_BASE = 0.15
SIGMA_FLAT_CAP = 0.35
SIGMA_UNIVERSAL_FLAT = 0.08
SIGMA_UNIVERSAL_RICH = 0.50


def clamp_rate(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def parse_time(text):
    """ISO 8601 timestamp -> aware datetime (UTC)."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def round4(x):
    """Round to 4 decimals, half up, returning an int for whole numbers."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def wls_line(xs, ys, ws):
    """Weighted least squares y = a + b x.  Returns (a, b, resid_sd, n)."""
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return None
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    var = sum(w * r * r for w, r in zip(ws, resid)) / sw
    return a, b, math.sqrt(var), len(xs)


def wls_quad(xs, ys, ws):
    """Weighted least squares y = a + b x + c x^2.  Returns
    (a, b, c, resid_sd) or None.  Only a concave (c <= 0) fit is returned:
    a convex one would explode on extrapolation."""
    n = len(xs)
    if n < 4:
        return None
    # Normal equations for [1, x, x^2].
    s = [0.0] * 5  # sum w x^k for k = 0..4
    t = [0.0] * 3  # sum w y x^k for k = 0..2
    for x, y, w in zip(xs, ys, ws):
        xp = 1.0
        for k in range(5):
            s[k] += w * xp
            if k < 3:
                t[k] += w * y * xp
            xp *= x
    # Solve the 3x3 system by Gaussian elimination.
    m = [[s[0], s[1], s[2], t[0]],
         [s[1], s[2], s[3], t[1]],
         [s[2], s[3], s[4], t[2]]]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            return None
        m[col], m[piv] = m[piv], m[col]
        for r in range(col + 1, 3):
            f = m[r][col] / m[col][col]
            for cc in range(col, 4):
                m[r][cc] -= f * m[col][cc]
    coef = [0.0] * 3
    for r in (2, 1, 0):
        coef[r] = (m[r][3] - sum(m[r][cc] * coef[cc]
                                 for cc in range(r + 1, 3))) / m[r][r]
    a, b, c = coef
    if c > 0:
        return None
    resid = [y - (a + b * x + c * x * x) for x, y in zip(xs, ys)]
    var = sum(w * r * r for w, r in zip(ws, resid)) / sum(ws)
    return a, b, c, math.sqrt(var)


def ols_line(xs, ys):
    """Unweighted least squares; returns (a, b) or None."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - b * mx, b


def snapshots(case):
    """Collapse the history into one snapshot per refresh.

    Exophase's page sits unchanged between refreshes, so consecutive reads
    with the same players-tracked and the same rates carry no new
    information.  Returns a list of (t_hours, players, rates) with strictly
    increasing times, plus the release origin used for t.
    """
    origin = None
    game = case.get("game") or {}
    release = game.get("release_date")
    if release:
        try:
            origin = datetime.fromisoformat(release).replace(tzinfo=timezone.utc)
        except ValueError:
            origin = None
    history = case.get("history") or []
    if origin is None and history:
        origin = parse_time(history[0]["at"])
    if origin is None:
        origin = parse_time(case["at"])
    out = []
    for read in history:
        players = read.get("players_tracked")
        rates = read.get("rates") or []
        if players is None or not rates:
            continue
        t = (parse_time(read["at"]) - origin).total_seconds() / 3600.0
        if out and out[-1][1] == players and out[-1][2] == rates:
            continue
        out.append((t, players, list(rates)))
    return out, origin


def apportion(weights, total):
    """Split `total` whole earners across trophies in proportion to weights
    (largest remainder).  Returns a list of ints summing to total."""
    wsum = float(sum(weights))
    if wsum <= 0 or total <= 0:
        return [0] * len(weights)
    shares = [w / wsum * total for w in weights]
    out = [int(math.floor(s)) for s in shares]
    left = total - sum(out)
    order = sorted(range(len(weights)),
                   key=lambda i: (-(shares[i] - out[i]), -weights[i], i))
    for i in order[:left]:
        out[i] += 1
    return out


def refresh_forecast(last_rates, players0, earners0):
    """Forecast after one launch-night-style refresh (flat history).

    Returns (median_rates, sigmas): the rate each trophy is expected to
    show after the refresh, and a logit spread for it.  The split of the
    new earners across trophies is hedged over three rules, because the
    public facts pin the totals but not the split.
    """
    n = len(last_rates)
    players1 = players0 * REFRESH_PLAYERS
    total1 = sum(earners0) * REFRESH_EARNERS
    universal = [i for i, r in enumerate(last_rates) if r is not None and r >= 99.5]
    others = [i for i in range(n) if i not in universal]
    new_total = total1 - sum(earners0)
    new_universal = len(universal) * (players1 - players0)
    new_others = max(0.0, new_total - new_universal)

    # Rule A: in proportion to old earners.
    alloc_a = apportion([earners0[i] for i in others], int(round(new_others)))
    # Rule B: nested by difficulty -- the easiest trophies first, each new
    # cohort member taking the next easiest trophy it does not yet hold.
    order_b = sorted(others, key=lambda i: (-(last_rates[i] or 0.0), i))
    alloc_b = [0] * len(others)
    pos_b = {trophy: k for k, trophy in enumerate(others)}
    left = int(round(new_others))
    for i in order_b:
        take = min(left, max(1, earners0[i]))
        alloc_b[pos_b[i]] = take
        left -= take
        if left <= 0:
            break
    if left > 0:
        alloc_b[pos_b[order_b[0]]] += left
    # Rules C and D: concentrated power laws, in proportion to old earners
    # to the 4th and 5th powers.  The rules' stated practice loss for the
    # house baseline (about 1.51) is only reproduced by a concentrated
    # split -- a purely proportional split gives about 1.40 -- so the
    # concentrated rules carry most of the weight.
    alloc_c = apportion([float(earners0[i]) ** 4 for i in others],
                        int(round(new_others)))
    alloc_d = apportion([float(earners0[i]) ** 5 for i in others],
                        int(round(new_others)))

    medians = [0.0] * n
    sigmas = [SIGMA_FLAT_BASE] * n
    for i in universal:
        medians[i] = 99.99
        sigmas[i] = SIGMA_UNIVERSAL_FLAT
    # Weighted geometric mean of the four rules' rates (nested and the
    # concentrated power laws counted twice each, the proportional split
    # once), and a spread that grows with how far the rules disagree.
    weights = (1.0, 2.0, 2.0, 1.0)
    for k, i in enumerate(others):
        e0 = earners0[i]
        cands = []
        for alloc in (alloc_a, alloc_b, alloc_c, alloc_d):
            e1 = e0 + alloc[k]
            cands.append(max(0.01, e1 / players1 * 100.0))
        logs = [math.log(c) for c in cands]
        medians[i] = math.exp(sum(w * l for w, l in zip(weights, logs))
                              / sum(weights))
        spread = max(logs) - min(logs)
        sigmas[i] = min(SIGMA_FLAT_CAP,
                        SIGMA_FLAT_BASE + 0.25 * spread)
    return medians, sigmas


def forecast_case(case):
    trophies = case["trophies"]
    n = len(trophies)
    horizon_days = case["horizon_minutes"] / 1440.0
    snaps, origin = snapshots(case)
    if not snaps:
        # No history at all: say roughly 1% everywhere, very wide.
        return {"trophies": [
            {"id": t["id"], "q": quantiles(1.0, 1.5)} for t in trophies]}
    last_t, last_p, last_rates = snaps[-1]
    target = parse_time(case["target"])
    t_target = (target - origin).total_seconds() / 3600.0

    earners_last = []
    for i, r in enumerate(last_rates):
        if r is None:
            earners_last.append(0)
        else:
            earners_last.append(int(round(r * last_p / 100.0)))

    if len(snaps) == 1:
        medians, sigmas = refresh_forecast(last_rates, last_p, earners_last)
        # A longer flat horizon means more than one refresh and some real
        # progress after it: drift the medians gently upward in logit, a
        # little more for the trophies players reach first.
        extra_days = max(0.0, horizon_days - 90.0 / 1440.0)
        if extra_days > 0:
            order = sorted(range(n), key=lambda i: (-(last_rates[i] or 0.0), i))
            for rank, i in enumerate(order):
                ease = 1.0 - rank / max(1, n - 1)
                drift = 0.08 * extra_days * (0.4 + 0.6 * ease)
                if last_rates[i] is not None and last_rates[i] < 99.5:
                    medians[i] = inv_logit(logit(medians[i]) + drift)
        rows = []
        for i, t in enumerate(trophies):
            rows.append({"id": t["id"],
                         "q": quantiles(medians[i], sigmas[i])})
        return {"trophies": rows}

    # ---- Rich history: fit ln P and ln E_i against x = ln(1 + t). ----
    def x_of(t):
        return math.log(max(0.05, 1.0 + t))

    xs = [x_of(t) for t, _, _ in snaps]
    # Players tracked: one series for the whole game.  Fit ln P against x
    # both as a line and as a concave quadratic (adoption saturates: the
    # launch wave's growth rate declines).  Take the quadratic when it fits
    # clearly better and its extrapolation stays sane (players tracked
    # never falls, and does not explode); otherwise the line.
    ln_p = [math.log(max(1.0, float(p))) for _, p, _ in snaps]
    fit_p = wls_line(xs, ln_p, [1.0] * len(xs))
    quad_p = None
    quad_resid = None
    if fit_p is None:
        b_p, a_p = 0.0, ln_p[-1]
    else:
        a_p, b_p = fit_p[0], fit_p[1]
        quad = wls_quad(xs, ln_p, [1.0] * len(xs))
        if quad is not None:
            aq, bq, cq, quad_resid = quad
            # Re-anchor the quadratic at the last read too.
            aq = ln_p[-1] - bq * xs[-1] - cq * xs[-1] ** 2
            quad_p = (aq, bq, cq)
    # Anchor the players line at the last read.
    a_p = ln_p[-1] - b_p * xs[-1]

    # Refresh cadence: median gap between distinct snapshots, in hours.
    gaps = [snaps[k + 1][0] - snaps[k][0] for k in range(len(snaps) - 1)]
    gaps = [g for g in gaps if g > 0]
    gaps.sort()
    med_gap = gaps[len(gaps) // 2] if gaps else 0.0
    # The page at the target shows the last refresh before it.
    t_eff = max(last_t, t_target - 0.5 * med_gap)
    x_eff = x_of(t_eff)
    x_last = xs[-1]
    dx = max(0.0, x_eff - x_last)

    # Which players-tracked curve to use at the target: the concave
    # quadratic when it fits clearly better and extrapolates sanely,
    # else the line.  Their disagreement is genuine uncertainty about
    # how fast the launch wave saturates, so half of it feeds the spread.
    ln_p_line = a_p + b_p * x_eff
    ln_p_use = ln_p_line
    sigma_pshape = 0.0
    if quad_p is not None and fit_p is not None and quad_resid is not None:
        aq, bq, cq = quad_p
        ln_p_quad = aq + bq * x_eff + cq * x_eff * x_eff
        sane = (ln_p[-1] - 0.05 <= ln_p_quad
                <= ln_p[-1] + 3.0)  # never falls; at most ~20x growth
        better = quad_resid < 0.8 * fit_p[2]
        if sane and better:
            ln_p_use = ln_p_quad
        sigma_pshape = 0.5 * min(1.0, abs(ln_p_quad - ln_p_line))

    # Per-trophy earners fits.  ln E is fitted both as a line and as a
    # concave quadratic in x: earners growth bends down as a trophy's rate
    # saturates toward its ceiling (players who were going to rush it have
    # done so), and a straight line extrapolated 4.5 days would run away.
    # The quadratic's curvature is shrunk toward the pooled curvature,
    # and a convex fit falls back to the line.
    slopes = []
    slope_ok = []
    line_resids = []
    curves = []
    curve_ok = []
    resid_all = []
    for i in range(n):
        pts_x, pts_y, pts_w = [], [], []
        for (t, p, rates), x in zip(snaps, xs):
            r = rates[i] if i < len(rates) else None
            if r is None or r <= 0 or p is None or p <= 0:
                continue
            e = r * p / 100.0
            if e <= 0:
                continue
            pts_x.append(x)
            pts_y.append(math.log(e))
            pts_w.append(r * r)  # 2-decimal rounding noise ~ 0.005/rate
        fit = wls_line(pts_x, pts_y, pts_w) if len(pts_x) >= 2 else None
        if fit is not None:
            slopes.append(fit[1])
            slope_ok.append(True)
            line_resids.append(fit[2])
            resid_all.append(fit[2])
        else:
            slopes.append(None)
            slope_ok.append(False)
            line_resids.append(None)
        quad = wls_quad(pts_x, pts_y, pts_w) if len(pts_x) >= 4 else None
        if quad is not None:
            curves.append(quad)
            curve_ok.append(True)
        else:
            curves.append(None)
            curve_ok.append(False)

    # Pooled curvature (the typical bend of ln E in x), for shrinkage.
    c_bar = 0.0
    if any(curve_ok):
        cs = [q[2] for q, ok in zip(curves, curve_ok) if ok]
        c_bar = sum(cs) / len(cs)
        c_bar = min(0.0, c_bar)

    # Difficulty-predicted slope: regress the fitted slopes on the logit of
    # each trophy's last rate, so rare trophies are predicted to grow slowly.
    us = []
    bs = []
    for i in range(n):
        if slope_ok[i] and last_rates[i] is not None and 0 < last_rates[i] < 99.5:
            us.append(logit(last_rates[i]))
            bs.append(slopes[i])
    pred = ols_line(us, bs) if len(us) >= 5 else None

    def predicted_slope(i):
        if pred is None:
            return sum(bs) / len(bs) if bs else 0.0
        r = last_rates[i]
        if r is None or r <= 0 or r >= 99.5:
            return pred[0] + pred[1] * logit(1.0)
        return pred[0] + pred[1] * logit(r)

    # Pooled residual scatter (logit units) for the spread.
    sigma_fit = SIGMA_FIT_FLOOR
    if resid_all:
        sigma_fit = min(SIGMA_FIT_CAP,
                        max(SIGMA_FIT_FLOOR,
                            math.sqrt(sum(v * v for v in resid_all)
                                      / len(resid_all))))
    days_past_last = max(0.0, (t_target - last_t) / 24.0)
    sigma_rw = SIGMA_RW_PER_SQRT_DAY * math.sqrt(days_past_last)
    # Few snapshots means the slopes are barely identified (with two, a
    # line through them is exact but the next refresh can bend it hard):
    # charge for that, scaled by how far past the last read we extrapolate.
    n_distinct = len(snaps)
    sigma_few = (0.5 / math.sqrt(max(1, n_distinct - 1))
                 * (dx / (dx + 0.5)))

    rows = []
    for i, t in enumerate(trophies):
        r_last = last_rates[i]
        if r_last is None:
            # Never observed: borrow the pooled median rate, wide spread.
            seen = [r for r in last_rates if r is not None]
            seen.sort()
            med = seen[len(seen) // 2] if seen else 1.0
            rows.append({"id": t["id"], "q": quantiles(med, SIGMA_MAX)})
            continue
        if r_last >= 99.5:
            rows.append({"id": t["id"],
                         "q": quantiles(99.99, SIGMA_UNIVERSAL_RICH)})
            continue
        # Slope: shrink the trophy's own fit toward its difficulty
        # prediction; the more snapshots it has, the more it is trusted.
        npts = sum(1 for (_, _, rates) in snaps
                   if i < len(rates) and rates[i])
        w_own = npts / (npts + 3.0)
        b_hat = predicted_slope(i)
        b_i = slopes[i] if slope_ok[i] else b_hat
        b_star = w_own * b_i + (1.0 - w_own) * b_hat
        # Keep the slope in a sane band: earners do not fall, and they do
        # not explode faster than the launch-night pattern sustained.
        b_star = min(3.0, max(-0.2, b_star))
        # Earners move from the last read to the target, in ln units.
        # Take the concave quadratic's move when it fits this trophy's
        # history clearly better than the line: growth bends down as the
        # rate saturates toward its ceiling, and a straight line
        # extrapolated 4.5 days runs away.  Otherwise the shrunk line
        # slope.
        move = b_star * dx
        if curve_ok[i] and line_resids[i] is not None:
            _aq, bq, cq, qres = curves[i]
            if qres < 0.8 * line_resids[i]:
                w_c = npts / (npts + 5.0)
                c_star = min(0.0, w_c * cq + (1.0 - w_c) * c_bar)
                move_q = (bq * dx
                          + c_star * (x_eff * x_eff - x_last * x_last))
                if -0.05 <= move_q <= move + 0.5:
                    move = move_q
        # Level anchored at the last read; move it by the shrunk slope.
        e_last = max(0.05, r_last * last_p / 100.0)
        ln_e = math.log(e_last) + move
        med = math.exp(ln_e - ln_p_use) * 100.0
        med = clamp_rate(med)
        # Spread: fit scatter, slope doubt stretched over the horizon,
        # the few-snapshot penalty, the players-curve shape doubt, the
        # random walk past the last read, and refresh noise.
        sigma_slope = 0.5 * abs(b_star - b_hat) * dx
        sigma = math.sqrt(sigma_fit ** 2 + sigma_slope ** 2
                          + sigma_few ** 2 + sigma_pshape ** 2
                          + sigma_rw ** 2 + SIGMA_REFRESH ** 2)
        sigma = min(SIGMA_MAX, max(SIGMA_MIN, sigma))
        rows.append({"id": t["id"], "q": quantiles(med, sigma)})
    return {"trophies": rows}


def quantiles(median, sigma):
    """Logit-normal fan around the median, 4 decimals, non-decreasing."""
    lg = logit(median)
    out = []
    prev = 0.0
    for z in Z:
        q = clamp_rate(inv_logit(lg + z * sigma))
        q = round4(q)
        if q < prev:
            q = prev
        out.append(q)
        prev = q
    return out


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        if len(text.encode("utf-8")) > 65536:
            raise ValueError(f"{path.name}: forecast over 65,536 bytes")
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
