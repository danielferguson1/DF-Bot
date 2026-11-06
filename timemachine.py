"""Time machine: before a stock BUY, how would this exact plan have played out on the symbol's own recent days?

Two answers, both from the symbol's last TM_DAYS full sessions of 5-minute bars:
- replays: TM_REPLAYS simulated rest-of-days. Each bar is drawn from real bars at about the same time of day
  (within +-window slots), as ratios to the bar before it, chained from today's entry price.
- history: every real past session, entered at the close of the same time-of-day bar.
Each run ends at the target, the stop, or the bot's end-of-day flatten, whichever comes first. If a bar
reaches both the stop and the target, the stop counts (conservative, like backtest.py).

It is a simulation of the last ~3 months, not a prediction: earnings, halts and news can do things those
days never did. Pure functions (no API calls), so it is tested offline.

    python timemachine.py AAPL --days 60     # odds for a BUY right now with the bot's ATR stop/target
"""
import numpy as np
import pandas as pd

import config

BARS_PER_DAY = 390 // config.BAR_MINUTES                                   # 78 five-minute bars
LAST_SLOT = BARS_PER_DAY - config.FLATTEN_MINUTES_BEFORE_CLOSE // config.BAR_MINUTES - 1  # last bar held
MIN_SESSIONS = 10


def slot(ts) -> int:
    """Bar number in the session: 0 = 9:30 bar. ts: tz-aware timestamp."""
    et = pd.Timestamp(ts).tz_convert("America/New_York")
    return (et.hour * 60 + et.minute - 570) // config.BAR_MINUTES


def full_sessions(df: pd.DataFrame, before) -> np.ndarray:
    """(sessions, BARS_PER_DAY, 4) array of open/high/low/close for complete regular-hours days before the
    `before` date (New York). Half days and days with a missing bar are left out."""
    et = df.index.tz_convert("America/New_York")
    days = []
    for d, g in df.groupby(et.date):
        if d < before and len(g) == BARS_PER_DAY:
            days.append(g[["open", "high", "low", "close"]].to_numpy(dtype=float))
    return np.array(days).reshape(-1, BARS_PER_DAY, 4)


def _walk(o, h, l, c, entry, stop, target):
    """Vectorised over runs: arrays (runs, bars). Returns R per run and outcome codes 1 target, -1 stop, 0 open."""
    risk = entry - stop
    runs, n = c.shape
    r = np.full(runs, np.nan)
    outcome = np.zeros(runs, dtype=int)
    for j in range(n):
        live = np.isnan(r)
        hit_stop = live & (l[:, j] <= stop)
        hit_tp = live & ~hit_stop & (h[:, j] >= target)
        r[hit_stop] = (np.minimum(o[hit_stop, j], stop) - entry) / risk       # a gap down fills at the open
        r[hit_tp] = (np.maximum(o[hit_tp, j], target) - entry) / risk
        outcome[hit_stop], outcome[hit_tp] = -1, 1
    still_open = np.isnan(r)
    r[still_open] = (c[still_open, -1] - entry) / risk if n else 0.0
    return r, outcome


def _summary(r, outcome) -> dict:
    n = len(r)
    return {"n": n, "target": float(np.mean(outcome == 1)), "stop": float(np.mean(outcome == -1)),
            "open": float(np.mean(outcome == 0)), "avg_r": float(np.mean(r))}


def history(days: np.ndarray, k: int, stop_pct: float, target_pct: float, last: int = LAST_SLOT) -> dict:
    """Enter each past session at the close of bar k, same % stop and target."""
    entry = days[:, k, 3]
    after = days[:, k + 1:last + 1, :] / entry[:, None, None]  # scale every day to an entry of 1.0
    o, h, l, c = (after[:, :, i] for i in range(4))
    return _summary(*_walk(o, h, l, c, 1.0, 1 - stop_pct, 1 + target_pct))


def replay(days: np.ndarray, k: int, stop_pct: float, target_pct: float, last: int = LAST_SLOT,
           replays: int = 10_000, seed: int = 19, window: int = 2) -> dict:
    """Monte Carlo rest-of-day from bar ratios at about the same time of day."""
    prev_close = days[:, :-1, 3]
    ratios = days[:, 1:, :] / prev_close[:, :, None]   # bar s (s >= 1) relative to bar s-1's close
    rng = np.random.default_rng(seed)
    steps = last - k
    sampled = np.empty((replays, steps, 4))
    for j in range(steps):
        s = k + 1 + j
        lo, hi = max(1, s - window), min(BARS_PER_DAY - 1, s + window)
        pool = ratios[:, lo - 1:hi, :].reshape(-1, 4)      # ratios[:, s-1] is bar s
        sampled[:, j, :] = pool[rng.integers(0, len(pool), replays)]
    close = np.cumprod(sampled[:, :, 3], axis=1)
    start = np.concatenate([np.ones((replays, 1)), close[:, :-1]], axis=1)  # previous close of each bar
    o, h, l = (start * sampled[:, :, i] for i in range(3))
    return _summary(*_walk(o, h, l, close, 1.0, 1 - stop_pct, 1 + target_pct))


def run(df: pd.DataFrame, now, entry: float, stop: float, target: float, **kw) -> dict | None:
    """df: the symbol's regular-hours 5-minute bars (any length). now: when the order is sent.
    Returns None if there isn't enough history or no time left in the day."""
    now = pd.Timestamp(now)
    k = slot(now) - 1                                  # the bar that just closed
    if not 0 <= k < LAST_SLOT or entry <= stop or target <= entry:
        return None
    days = full_sessions(df, now.tz_convert("America/New_York").date())[-config.TM_DAYS:]
    if len(days) < MIN_SESSIONS:
        return None
    stop_pct, target_pct = (entry - stop) / entry, (target - entry) / entry
    return {"mc": replay(days, k, stop_pct, target_pct, **kw), "history": history(days, k, stop_pct, target_pct)}


def describe(tm: dict | None) -> str:
    if tm is None:
        return "TM: not enough history"
    mc, hist = tm["mc"], tm["history"]
    return (f"TM {hist['n']}d: replays target {mc['target']:.0%} / stop {mc['stop']:.0%} / "
            f"open {mc['open']:.0%}, avg {mc['avg_r']:+.2f}R; history {hist['target']:.0%} / "
            f"{hist['stop']:.0%} / {hist['open']:.0%}, avg {hist['avg_r']:+.2f}R")


def main():
    import argparse
    import os
    from datetime import datetime, timedelta, timezone
    from dotenv import load_dotenv
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    import strategy

    ap = argparse.ArgumentParser()
    ap.add_argument("symbol")
    ap.add_argument("--days", type=int, default=config.TM_DAYS)
    args = ap.parse_args()
    load_dotenv()
    client = StockHistoricalDataClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"))
    now = datetime.now(timezone.utc)
    raw = client.get_stock_bars(StockBarsRequest(
        symbol_or_symbols=args.symbol, timeframe=TimeFrame(config.BAR_MINUTES, TimeFrameUnit.Minute),
        start=now - timedelta(days=int(args.days * 1.5) + 7), end=now - timedelta(minutes=16),
        feed=config.DATA_FEED)).df.loc[args.symbol]
    df = strategy.add_indicators(strategy.regular_hours(raw))
    row = df.iloc[-1]
    stop, target = strategy.bracket_prices(row.close, row.atr_14)
    at = row.name + pd.Timedelta(minutes=config.BAR_MINUTES)
    print(f"{args.symbol}: entry {row.close:.2f} at {at.tz_convert('America/New_York'):%Y-%m-%d %H:%M} ET, "
          f"stop {stop:.2f}, target {target:.2f}")
    print(describe(run(df, at, row.close, stop, target)))


if __name__ == "__main__":
    main()
