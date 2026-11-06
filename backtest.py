"""Replay the bot's rules on historical 5-minute bars.

    python backtest.py --days 60 --cost-bps 5

Simplifications (results will look better than live):
- each symbol is simulated independently: no MAX_POSITIONS / exposure / sector / daily-loss caps
- no spread check and no Qwen news veto
- if a bar touches both stop and target, the stop is assumed to hit first (conservative)
"""
import argparse
import os
from datetime import datetime, timedelta, timezone

import pandas as pd
from dotenv import load_dotenv
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

import config
import metrics
import strategy

START_EQUITY = 100_000.0


def simulate(df: pd.DataFrame, cost: float, equity: float = START_EQUITY, crypto: bool = False) -> list[tuple]:
    """df: indicator columns + 'regime_ok'. Returns [(exit_time, pnl)].
    Signals are computed at a bar's close and filled at the next bar's open.
    crypto=True mirrors the live crypto path: fractional size, held overnight (no daily flatten),
    take-profit checked at bar close (the bot checks it in code), stop still intrabar."""
    trades = []
    pos = None            # dict(qty, entry, stop, tp)
    pending = None        # ("BUY", atr) or ("SELL", None)
    rows = list(df.itertuples())
    et_times = df.index.tz_convert("America/New_York")

    def close_pos(price, when):
        nonlocal pos
        exit_price = price * (1 - cost)
        trades.append((when, pos["qty"] * (exit_price - pos["entry"])))
        pos = None

    for i in range(1, len(rows)):
        prev, row, t = rows[i - 1], rows[i], et_times[i]

        # 1. fill whatever the previous bar decided (the last bar of a day never sets one)
        if pending:
            action, atr = pending
            if action == "SELL" and pos:
                close_pos(row.open, row.Index)
            elif action == "BUY" and not pos:
                entry = row.open * (1 + cost)
                qty = strategy.position_size(equity, entry, atr, qty_increment=1e-9 if crypto else 1)
                if qty > 0:
                    stop, tp = strategy.bracket_prices(entry, atr)
                    pos = {"qty": qty, "entry": entry, "stop": stop, "tp": tp}
        pending = None

        # 2. stop intrabar (gaps fill at the open); target intrabar for stocks, at the close for crypto
        if pos and row.low <= pos["stop"]:
            close_pos(min(pos["stop"], row.open), row.Index)
        elif pos and not crypto and row.high >= pos["tp"]:
            close_pos(max(pos["tp"], row.open), row.Index)
        elif pos and crypto and row.close >= pos["tp"]:
            close_pos(row.close, row.Index)

        # 3. stocks: flatten on the last bar of the day
        if not crypto:
            last_bar = i == len(rows) - 1 or et_times[i + 1].date() != t.date()
            if last_bar:
                if pos:
                    close_pos(row.close, row.Index)
                continue

        # 4. decide for the next bar
        action = strategy.signal(prev, row, pos is not None, row.regime_ok)
        minutes_to_close = 960 - (t.hour * 60 + t.minute + config.BAR_MINUTES)
        if not crypto and action == "BUY" and minutes_to_close <= config.NO_ENTRY_MINUTES_BEFORE_CLOSE:
            action = "HOLD"
        if action in ("BUY", "SELL"):
            pending = (action, row.atr_14)
    return trades


def equity_curve(trades: list[tuple], bar_index: pd.DatetimeIndex, start_equity: float = START_EQUITY) -> pd.Series:
    """Account value at every bar (not just at exits), so daily returns include days with no trades."""
    pnl = pd.Series([p for _, p in trades], index=[t for t, _ in trades], dtype=float)
    pnl = pnl.groupby(level=0).sum()
    pnl = pnl.reindex(bar_index.union(pnl.index), fill_value=0.0)  # union: never drop an exit
    return start_equity + pnl.cumsum()


def load_bars(client, symbols, days):
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame(config.BAR_MINUTES, TimeFrameUnit.Minute),
        start=datetime.now(timezone.utc) - timedelta(days=days),
        end=datetime.now(timezone.utc) - timedelta(minutes=20),
        feed=config.DATA_FEED,  # same feed as the live bot, so volume/VWAP behave the same
    )
    raw = client.get_stock_bars(req).df
    return {s: strategy.add_indicators(strategy.regular_hours(raw.loc[s]))
            for s in symbols if s in raw.index.get_level_values(0)}


def load_crypto_bars(client, symbols, days):
    raw = client.get_crypto_bars(CryptoBarsRequest(
        symbol_or_symbols=symbols, timeframe=TimeFrame(config.BAR_MINUTES, TimeFrameUnit.Minute),
        start=datetime.now(timezone.utc) - timedelta(days=days))).df
    return {s: strategy.add_indicators(raw.loc[s]) for s in symbols if s in raw.index.get_level_values(0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--cost-bps", type=float, default=5.0, help="stocks: spread+slippage per side, in bps")
    ap.add_argument("--crypto-cost-bps", type=float, default=30.0,
                    help="crypto: fee+spread+slippage per side, in bps (Alpaca charges crypto trading fees)")
    args = ap.parse_args()

    load_dotenv()
    key_id, secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
    client = StockHistoricalDataClient(key_id, secret)
    symbols = sorted(set(config.WATCHLIST) | {config.REGIME_SYMBOL})
    bars = load_bars(client, symbols, args.days)
    spy = bars[config.REGIME_SYMBOL]
    spy_up = (spy["close"] > spy["vwap"])

    cost = args.cost_bps / 10_000
    all_trades, bar_index = [], spy.index
    for symbol, df in load_crypto_bars(CryptoHistoricalDataClient(key_id, secret),
                                       config.CRYPTO_WATCHLIST, args.days).items():
        df = df.assign(regime_ok=df["close"] > df["vwap"])  # same as live: own VWAP
        trades = simulate(df, args.crypto_cost_bps / 10_000, crypto=True)
        print(f"{symbol}: {len(trades)} trades, P&L ${sum(p for _, p in trades):,.2f} "
              f"(buy & hold {df['close'].iloc[-1] / df['open'].iloc[0] - 1:.2%})")
        all_trades += trades
        bar_index = bar_index.union(df.index)
    for symbol in config.WATCHLIST:
        df = bars.get(symbol)
        if df is None or len(df) < 30:
            print(f"{symbol}: not enough data, skipped")
            continue
        up = spy_up.reindex(df.index, fill_value=False)
        df = df.assign(regime_ok=~up if symbol in config.INVERSE else up)
        trades = simulate(df, cost)
        pnl = sum(p for _, p in trades)
        print(f"{symbol}: {len(trades)} trades, P&L ${pnl:,.2f}")
        all_trades += trades

    equity = equity_curve(all_trades, bar_index)
    print(f"\nStrategy, {args.days} days, stocks {args.cost_bps} / crypto {args.crypto_cost_bps} bps/side:")
    metrics.print_summary(metrics.summarize(equity, [p for _, p in all_trades]))
    spy_ret = spy["close"].iloc[-1] / spy["open"].iloc[0] - 1
    print(f"\nBuy & hold {config.REGIME_SYMBOL}: {spy_ret:.2%}")


if __name__ == "__main__":
    main()
