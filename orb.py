"""Opening-range breakout on stocks in play (Zarattini, Barbon & Aziz 2023). Pure functions, no API calls.

Each trading day:
1. Before the open: the universe = stocks whose prior close > ORB_MIN_PRICE, 14-day average daily volume
   >= ORB_MIN_AVG_VOLUME and 14-day daily ATR > ORB_MIN_ATR (eligible()).
2. After the 9:30-9:35 bar: rank by that bar's volume relative to its 14-day average, keep the top ORB_TOP with
   rvol > ORB_MIN_RVOL (select()). If the bar closed up: buy stop 1 cent above its high; down: sell-short stop
   1 cent below its low. Protective stop ORB_STOP_ATR x ATR beyond the trigger.
3. Hold until the stop or the bot's end-of-day flatten. No target.
"""
import math
from dataclasses import dataclass

import pandas as pd

import config


@dataclass
class Plan:
    symbol: str
    side: str          # "long" / "short"
    trigger: float     # entry stop price
    stop: float        # protective stop price
    rvol: float
    atr: float
    qty: int = 0

    @property
    def risk_per_share(self) -> float:
        return round(abs(self.trigger - self.stop), 6)  # 50.00 - 49.80 = 0.2000000000000028 otherwise


def eligible(daily: pd.DataFrame, today) -> pd.DataFrame:
    """daily: (symbol, timestamp) x open/high/low/close/volume daily bars. Only days before `today` (New York
    date) are used. Returns one row per eligible symbol: prevclose, avgvol14, atr14."""
    d = daily.reset_index()
    d["date"] = d["timestamp"].dt.tz_convert("America/New_York").dt.date
    d = d[d["date"] < today].sort_values(["symbol", "date"])
    pc = d.groupby("symbol")["close"].shift()
    d["tr"] = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(), (d["low"] - pc).abs()], axis=1).max(axis=1)
    last = d.groupby("symbol").tail(14)
    stats = last.groupby("symbol").agg(n=("close", "size"), prevclose=("close", "last"),
                                       avgvol14=("volume", "mean"), atr14=("tr", "mean"))
    stats = stats[stats["n"] == 14]
    return stats[(stats["prevclose"] > config.ORB_MIN_PRICE) & (stats["avgvol14"] >= config.ORB_MIN_AVG_VOLUME)
                 & (stats["atr14"] > config.ORB_MIN_ATR)].drop(columns="n")


def avg_open_volume(open_bars: dict) -> pd.Series:
    """open_bars: {date: DataFrame indexed by symbol with a 'volume' column} of past 9:30-9:35 bars.
    Average per symbol over the days it traded; at least 10 of them are required."""
    vols = pd.DataFrame({d: df["volume"] for d, df in open_bars.items()})
    counts = vols.notna().sum(axis=1)
    return vols.mean(axis=1)[counts >= 10]


def _cents(x: float, up: bool) -> float:
    return (math.ceil if up else math.floor)(round(x * 100, 6)) / 100


def select(today_open: pd.DataFrame, avg_vol: pd.Series, elig: pd.DataFrame, shortable: set) -> list[Plan]:
    """today_open: today's 9:30-9:35 bar per symbol (open/high/low/close/volume). Best rvol first."""
    df = today_open.join(avg_vol.rename("avg"), how="inner").join(elig, how="inner")
    df = df[(df["avg"] > 0) & (df["open"] > config.ORB_MIN_PRICE)]
    df["rvol"] = df["volume"] / df["avg"]
    df = df[df["rvol"] > config.ORB_MIN_RVOL].sort_values("rvol", ascending=False).head(config.ORB_TOP)
    plans = []
    for sym, r in df.iterrows():
        dist = config.ORB_STOP_ATR * r["atr14"]
        if r["close"] > r["open"] and "long" in config.ORB_SIDES:
            trigger = _cents(r["high"] + 0.01, up=True)
            plans.append(Plan(sym, "long", trigger, _cents(trigger - dist, up=False), r["rvol"], r["atr14"]))
        elif r["close"] < r["open"] and "short" in config.ORB_SIDES and sym in shortable:
            trigger = _cents(r["low"] - 0.01, up=False)
            plans.append(Plan(sym, "short", trigger, _cents(trigger + dist, up=True), r["rvol"], r["atr14"]))
    return plans


def size(plan: Plan, equity: float) -> int:
    """Whole shares: risk ORB_RISK_PCT of equity at the stop, capped at ORB_POSITION_USD."""
    if plan.risk_per_share <= 0 or plan.trigger <= 0:
        return 0
    return int(min(equity * config.ORB_RISK_PCT / plan.risk_per_share, config.ORB_POSITION_USD / plan.trigger))


def already_through(plan: Plan, price: float) -> bool:
    """The price has already crossed the trigger, so a stop order would fill at once at a worse price."""
    return price >= plan.trigger if plan.side == "long" else price <= plan.trigger


def trim_to_budget(plans: list[Plan], budget: float) -> list[Plan]:
    """Keep plans in rank order while their total risk at the stop fits the budget."""
    out, used = [], 0.0
    for p in plans:
        r = p.qty * p.risk_per_share
        if used + r <= budget:
            out.append(p)
            used += r
    return out
