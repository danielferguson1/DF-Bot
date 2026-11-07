"""Intraday momentum on one ETF (Zarattini, Aziz & Barbon 2024, "Beat the Market: An Effective Intraday Momentum
Strategy for S&P500 ETF (SPY)"). Pure functions, no API calls: backtest.py and the bot run the same code.

Each session:
1. Noise area: sigma(m) = average over the last MOMENTUM_LOOKBACK sessions of |price m minutes after the open / that
   session's open - 1|. Upper band = max(today's open, yesterday's close) x (1 + sigma(m)); lower band =
   min(today's open, yesterday's close) x (1 - sigma(m)).
2. Every MOMENTUM_CHECK_MINUTES from 30 minutes after the open (10:00, 10:30, ...) until the close-out: long while the
   price is above both the upper band and today's VWAP, short while it is below both the lower band and VWAP, flat
   otherwise. The band and VWAP are the trailing stops, checked only at these times, as in the paper.
3. Flat at the bot's close-out (FLATTEN_MINUTES_BEFORE_CLOSE before the close). Size: MOMENTUM_SIZE x yesterday's
   equity / today's open, in whole shares.
Arrays are per minute of the session: index k = the price k+1 minutes after the open (the close of bar k).
"""
import math

import numpy as np
import pandas as pd

import config

NY = "America/New_York"


def session_bounds(cal_entry) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Alpaca calendar entry -> (open, close) in New York time."""
    def ts(x):
        x = pd.Timestamp(x)
        return x.tz_localize(NY) if x.tzinfo is None else x.tz_convert(NY)
    return ts(cal_entry.open), ts(cal_entry.close)


def check_minutes(session_minutes: int) -> list[int]:
    """Minutes after the open of each decision: 30, 60, ... before the close-out (15:30 on a full day, 12:30 on a half
    day)."""
    return list(range(30, session_minutes - config.FLATTEN_MINUTES_BEFORE_CLOSE, config.MOMENTUM_CHECK_MINUTES))


def _minute_index(bars: pd.DataFrame, open_time: pd.Timestamp, session_minutes: int):
    k = ((bars.index - open_time).total_seconds() // 60).to_numpy().astype(int)
    keep = (k >= 0) & (k < session_minutes)
    return k[keep], keep


def minute_prices(bars: pd.DataFrame, open_time: pd.Timestamp, session_minutes: int) -> np.ndarray:
    """1-minute bars of one session (indexed by bar start) -> price after each minute, carried forward over minutes
    without a trade; NaN before the first one."""
    k, keep = _minute_index(bars, open_time, session_minutes)
    p = np.full(session_minutes, np.nan)
    p[k] = bars["close"].to_numpy(float)[keep]
    return pd.Series(p).ffill().to_numpy()


def vwap_by_minute(bars: pd.DataFrame, open_time: pd.Timestamp, session_minutes: int) -> np.ndarray:
    """Session VWAP through each minute: each bar's own VWAP (its close if missing) weighted by its volume."""
    k, keep = _minute_index(bars, open_time, session_minutes)
    v = np.zeros(session_minutes)
    pv = np.zeros(session_minutes)
    vol = bars["volume"].to_numpy(float)[keep]
    px = (bars["vwap"].fillna(bars["close"]) if "vwap" in bars else bars["close"]).to_numpy(float)[keep]
    np.add.at(v, k, vol)
    np.add.at(pv, k, px * vol)
    cv = np.cumsum(v)
    return np.where(cv > 0, np.cumsum(pv) / np.where(cv > 0, cv, 1.0), np.nan)


def session_open(bars: pd.DataFrame, open_time: pd.Timestamp) -> float:
    """The opening trade: the first bar's open if it is the opening minute, else that bar's close."""
    first = bars.iloc[0]
    return float(first["open"] if bars.index[0] <= open_time else first["close"])


def sigma(past: list[tuple[float, np.ndarray]]) -> np.ndarray:
    """past: (open, minute prices) of earlier sessions, oldest first. Average |price / open - 1| at each minute over
    the last MOMENTUM_LOOKBACK of them; NaN where fewer than LOOKBACK - 4 have that minute (half days end at 13:00)."""
    n = config.MOMENTUM_LOOKBACK
    moves = np.full((n, 390), np.nan)
    for i, (o, p) in enumerate(past[-n:]):
        moves[i, :len(p)] = np.abs(p / o - 1)
    count = np.sum(~np.isnan(moves), axis=0)
    total = np.nansum(moves, axis=0)
    return np.where(count >= n - 4, total / np.maximum(count, 1), np.nan)


def bands(open_: float, prev_close: float, sig: float) -> tuple[float, float]:
    """(upper, lower) edges of the noise area."""
    return max(open_, prev_close) * (1 + sig), min(open_, prev_close) * (1 - sig)


def target(price: float, vwap: float, upper: float, lower: float) -> int:
    """+1 long, -1 short, 0 flat."""
    if price > max(upper, vwap):
        return 1
    if price < min(lower, vwap):
        return -1
    return 0


def shares(equity: float, open_: float) -> int:
    """Whole shares for the day: MOMENTUM_SIZE x equity at yesterday's close / today's open."""
    return math.floor(config.MOMENTUM_SIZE * equity / open_) if open_ > 0 and equity > 0 else 0
