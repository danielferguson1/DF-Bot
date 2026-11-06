"""Pure strategy logic: indicators, signal, sizing. No API calls, so it can be tested and backtested."""
import math
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR

import pandas as pd

import config

SIGNAL_FIELDS = ("sma_5", "sma_20", "rsi_14", "rel_volume", "atr_14")


def regular_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Keep 9:30-16:00 ET bars. Index must be tz-aware timestamps."""
    et = df.index.tz_convert("America/New_York")
    minutes = et.hour * 60 + et.minute
    return df[(minutes >= 570) & (minutes < 960)]


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """df: open/high/low/close/volume, tz-aware time index, regular hours only."""
    df = df.copy()
    close = df["close"]
    df["sma_5"] = close.rolling(5).mean()
    df["sma_20"] = close.rolling(20).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi_14"] = (100 - 100 / (1 + gain / loss)).fillna(50)  # 0/0 (flat) -> neutral; x/0 -> 100

    prev_close = close.shift()
    true_range = pd.concat([df["high"] - df["low"],
                            (df["high"] - prev_close).abs(),
                            (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    df["atr_14"] = true_range.rolling(14).mean()

    day = df.index.tz_convert("America/New_York").date
    typical = (df["high"] + df["low"] + close) / 3
    df["vwap"] = (typical * df["volume"]).groupby(day).cumsum() / df["volume"].groupby(day).cumsum()
    df["rel_volume"] = df["volume"] / df["volume"].rolling(20).mean()
    return df


def bar_is_fresh(bar_start, now, max_age_bars: int = 2) -> bool:
    """False if the bar ended more than max_age_bars intervals ago (e.g. yesterday's last bar at the open)."""
    bar = pd.Timedelta(minutes=config.BAR_MINUTES)
    return now - (bar_start + bar) <= max_age_bars * bar


def regime_up(spy_row) -> bool:
    return bool(spy_row.close > spy_row.vwap)  # NaN vwap -> False


def signal(prev, row, held: bool, regime_ok: bool) -> str:
    """BUY / SELL / HOLD from the previous and latest completed bar."""
    if any(_nan(getattr(r, f)) for r in (prev, row) for f in SIGNAL_FIELDS):
        return "HOLD"
    if held:
        return "SELL" if row.sma_5 < row.sma_20 else "HOLD"   # momentum failed
    if (regime_ok
            and row.sma_5 > row.sma_20                        # short-term uptrend
            and prev.rsi_14 < 40 and row.rsi_14 > prev.rsi_14  # RSI turning up from oversold
            and row.rel_volume >= 1.0):                        # volume confirms
        return "BUY"
    return "HOLD"


def position_size(equity: float, price: float, atr: float, qty_increment: float = 1):
    """Size so a stop-out loses ~RISK_PCT of equity, capped by MAX_POSITION_USD.
    qty_increment=1 -> whole shares (int); e.g. 1e-9 for fractional crypto."""
    stop_dist = config.STOP_ATR * atr
    if _nan(stop_dist) or stop_dist <= 0 or price <= 0:
        return 0
    raw = min(equity * config.RISK_PCT / stop_dist, config.MAX_POSITION_USD / price)
    return int(raw) if qty_increment == 1 else round_to_increment(raw, qty_increment)


def round_to_increment(x: float, inc: float, up: bool = False) -> float:
    """Round x down (or up) to a multiple of inc, without float noise like 0.30000000000000004."""
    steps = (Decimal(str(x)) / Decimal(str(inc))).to_integral_value(ROUND_CEILING if up else ROUND_FLOOR)
    return float(steps * Decimal(str(inc)))


def bracket_prices(entry: float, atr: float) -> tuple[float, float]:
    """(stop_price, take_profit_price)"""
    return round(entry - config.STOP_ATR * atr, 2), round(entry + config.TAKE_PROFIT_ATR * atr, 2)


def _nan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))
