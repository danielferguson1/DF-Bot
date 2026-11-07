"""Swing rules on index ETFs: positions held for days to months. Pure functions, no API calls, so backtest.py and the
bot run the same code.

Decided once a day a few minutes before the close, from past daily closes plus the live price as today's close:
1. Trend (Siegel 1994, Faber 2007): hold SWING_TREND_WEIGHT of equity in SWING_TREND_SYMBOL while its close is above
   its SWING_TREND_SMA-day average; none of it below.
2. Dips (Connors & Alvarez 2008, RSI(2)): for each of SWING_DIP_SYMBOLS, hold SWING_DIP_WEIGHT / N of equity from a
   close with RSI(2) < SWING_DIP_RSI_BELOW above its 200-day average, until the first close above its
   SWING_DIP_EXIT_SMA-day average. Bought on the entry day only (see targets()).
A symbol in both sleeves holds the sum. No stops: the trend exit and the dip exit are the rules as published.
"""
import math

import numpy as np
import pandas as pd

import config


def symbols() -> list[str]:
    return list(dict.fromkeys([config.SWING_TREND_SYMBOL, *config.SWING_DIP_SYMBOLS]))


def dip_slot() -> float:
    """Fraction of equity per dip position."""
    return config.SWING_DIP_WEIGHT / len(config.SWING_DIP_SYMBOLS) if config.SWING_DIP_SYMBOLS else 0.0


def rsi(close: pd.Series, n: int = 2) -> pd.Series:
    """Wilder's RSI (exponential smoothing, alpha = 1/n), as published for RSI(2). Not strategy.py's rolling-mean RSI."""
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / down)


def trend_on(close: pd.Series) -> pd.Series:
    """Close above its SWING_TREND_SMA-day average (False while the average isn't defined yet)."""
    return close > close.rolling(config.SWING_TREND_SMA).mean()


def dip_on(close: pd.Series) -> pd.Series:
    """Whether the RSI(2) rule holds the symbol after each close. Path-dependent, so it is replayed from the first
    close; a year of history settles it (a dip trade lasts days)."""
    r = rsi(close).to_numpy()
    up = trend_on(close).to_numpy()
    out = (close > close.rolling(config.SWING_DIP_EXIT_SMA).mean()).to_numpy()
    held, pos = np.zeros(len(close), dtype=bool), False
    for i in range(len(close)):
        if not pos and r[i] < config.SWING_DIP_RSI_BELOW and up[i]:
            pos = True
        elif pos and out[i]:
            pos = False
        held[i] = pos
    return pd.Series(held, index=close.index)


def rule_states(closes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """closes: date x symbol daily closes. (trend weight, dip rule holding: bool) per symbol after each close.
    Symbols missing from `closes` are left out (the caller doesn't trade them)."""
    cols = [s for s in symbols() if s in closes]
    trend = pd.DataFrame(0.0, index=closes.index, columns=cols)
    dips = pd.DataFrame(False, index=closes.index, columns=cols)
    t = config.SWING_TREND_SYMBOL
    if t in closes:
        trend[t] = trend_on(closes[t]) * config.SWING_TREND_WEIGHT
    for s in config.SWING_DIP_SYMBOLS:
        if s in closes:
            dips[s] = dip_on(closes[s])
    return trend, dips


def targets(trend_now: dict, trend_before: dict, dip_now: dict, dip_before: dict, equity: float,
            held_value: dict) -> dict:
    """Today's target fraction of equity per symbol. *_now / *_before: rule states after today's / the previous close.
    A dip trade is bought on its entry day only. One that is running but not held (the bot was off that day, or just
    started) is skipped: the rule bought at that day's close, and the price may since have fallen below the 200-day
    average. Held = the position beyond yesterday's trend weight is at least half a dip slot."""
    out, slot = {}, dip_slot()
    for s in dict.fromkeys([*trend_now, *dip_now]):
        dip = 0.0
        if dip_now.get(s):
            new = not dip_before.get(s, False)
            held = equity > 0 and (held_value.get(s, 0.0) / equity - trend_before.get(s, 0.0)) >= slot / 2
            dip = slot if new or held else 0.0
        out[s] = trend_now.get(s, 0.0) + dip
    return out


def today_targets(closes: pd.DataFrame, equity: float, held_value: dict) -> dict:
    """targets() for the last row of `closes` (today, with the live price as its close)."""
    trend, dips = rule_states(closes)
    before = (trend.iloc[-2].to_dict(), dips.iloc[-2].to_dict()) if len(closes) > 1 else ({}, {})
    return targets(trend.iloc[-1].to_dict(), before[0], dips.iloc[-1].to_dict(), before[1], equity, held_value)


def explain(close: pd.Series) -> dict:
    """The indicator values behind the latest target, for the log."""
    return {"price": float(close.iloc[-1]),
            "sma_trend": float(close.rolling(config.SWING_TREND_SMA).mean().iloc[-1]),
            "sma_exit": float(close.rolling(config.SWING_DIP_EXIT_SMA).mean().iloc[-1]),
            "rsi2": float(rsi(close).iloc[-1]),
            "trend": bool(trend_on(close).iloc[-1]), "dip": bool(dip_on(close).iloc[-1])}


def min_trade() -> float:
    """Smallest target change worth trading, as a fraction of equity: half the smallest slot. Entering or leaving a
    slot always trades; equity drift alone rarely does."""
    slots = [x for x in (config.SWING_TREND_WEIGHT, dip_slot()) if x > 0]
    return min(slots) / 2 if slots else 0.0


def orders(target: dict, equity: float, prices: dict, held: dict, fractional: dict) -> list[tuple[str, float]]:
    """[(symbol, signed qty)] that move `held` (signed shares) toward `target` (fraction of equity), sells first so
    they free the cash for the buys. A target of 0 closes the whole position. fractional[symbol] False = whole
    shares. Quantities are rounded toward zero, so an order never trades more than the gap."""
    out = []
    for s, w in target.items():
        have = held.get(s, 0.0)
        if w <= 0:
            if have:
                out.append((s, -have))
            continue
        gap = w * equity / prices[s] - have
        if abs(gap) * prices[s] < min_trade() * equity:
            continue
        qty = math.trunc(round(gap * 1e4, 6)) / 1e4 if fractional.get(s) else int(gap)
        if qty:
            out.append((s, qty))
    return sorted(out, key=lambda x: x[1])
