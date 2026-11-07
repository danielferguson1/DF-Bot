"""Replay the bot's rules on history.

    python backtest.py                      # the bot as configured: swing + momentum day trading, since 2016
    python backtest.py swing                # swing rules (swing.py) on daily bars, in detail
    python backtest.py momentum             # momentum day trading (momentum.py) on 1-minute bars, in detail
    python backtest.py intraday --days 60   # SMA/RSI rules on WATCHLIST + BTC on 5-minute bars

swing: the same swing.rule_states() / targets() / orders() calls as the bot, trading at each daily close (the bot
trades at 15:45 with the live price; replaying that on 2 years of 5-minute bars gave the same return within 0.2%).
Daily bars are adjusted for dividends, so returns include them. Idle cash earns nothing.

momentum: the same momentum.sigma() / bands() / target() / shares() calls as the bot, deciding at
momentum.check_minutes() on 1-minute consolidated-tape bars and flat at the close-out. Costs: half a 1-cent spread per
share plus the SEC fee on sales (--cost-bps to stress it). The bot decides on the free IEX feed instead, which made the
same decision at 99% of checks in a 2-year test. Downloads ~1M bars for 10 years (a couple of minutes).

intraday: same per-symbol rules as live: entry window after the open, no entries near the close, cooldown, minimum
reward:risk, ATR stop/target, end-of-day flatten for stocks.
Simplifications (results will look better than live):
- each symbol is simulated independently: no MAX_POSITIONS / exposure / sector / daily-loss / daily-risk /
  entries-per-day caps
- no spread check, cost gate, earnings check or Qwen news veto
- only config.WATCHLIST: the scanner's daily picks can't be replayed (past most-active lists aren't kept)
- if a bar touches both stop and target, the stop is assumed to hit first (conservative)
"""
import argparse
import math
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetCalendarRequest

import config
import metrics
import momentum
import strategy
import swing

START_EQUITY = 100_000.0
NY = "America/New_York"
PUBLISHED = pd.Timestamp("2024-05-10")   # first version of the momentum paper: later is out of its sample


def simulate(df: pd.DataFrame, cost: float, equity: float = START_EQUITY, crypto: bool = False) -> list[tuple]:
    """df: indicator columns + 'regime_ok'. Returns [(exit_time, pnl)].
    Signals are computed at a bar's close and filled at the next bar's open.
    crypto=True mirrors the live crypto path: fractional size, held overnight (no daily flatten),
    take-profit checked at bar close (the bot checks it in code), stop still intrabar."""
    trades = []
    last_activity = None  # time of the last entry or exit, for the cooldown
    pos = None            # dict(qty, entry, stop, tp)
    pending = None        # ("BUY", atr) or ("SELL", None)
    rows = list(df.itertuples())
    et_times = df.index.tz_convert("America/New_York")

    def close_pos(price, when):
        nonlocal pos, last_activity
        exit_price = price * (1 - cost)
        trades.append((when, pos["qty"] * (exit_price - pos["entry"])))
        pos, last_activity = None, when

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
                stop, tp = strategy.bracket_prices(entry, atr)
                if qty > 0 and strategy.reward_risk(entry, stop, tp) >= config.MIN_REWARD_RISK:
                    pos = {"qty": qty, "entry": entry, "stop": stop, "tp": tp}
                    last_activity = row.Index
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
        sent_at = row.Index + pd.Timedelta(minutes=config.BAR_MINUTES)  # the order goes out at the bar close
        if action == "BUY" and (
                (not crypto and minutes_to_close <= config.NO_ENTRY_MINUTES_BEFORE_CLOSE)
                or (not crypto and not strategy.in_entry_window(sent_at.tz_convert("America/New_York")))
                or not strategy.cooled_down(last_activity, sent_at)):
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
    return {s: strategy.add_indicators(raw.loc[s], sessions=False)
            for s in symbols if s in raw.index.get_level_values(0)}


# ---------- swing ----------

def simulate_swing(closes: pd.DataFrame, cost: float, equity: float = START_EQUITY,
                   fractional: bool = True) -> tuple[pd.DataFrame, list[tuple]]:
    """closes: date x symbol daily closes, no gaps. Each close, trade what swing.orders() says (as the bot does just
    before the close), paying `cost` per side. Returns (DataFrame of account value and fraction invested after each
    close, [(date, symbol, qty, price)])."""
    trend, dips = swing.rule_states(closes)
    trend_rows, dip_rows = trend.to_dict("records"), dips.to_dict("records")
    cash, held, curve, trades = equity, dict.fromkeys(trend.columns, 0.0), {}, []
    for i, (t, row) in enumerate(closes[list(trend.columns)].iterrows()):
        prices = row.to_dict()
        value = cash + sum(q * prices[s] for s, q in held.items())
        target = swing.targets(trend_rows[i], trend_rows[i - 1] if i else {}, dip_rows[i], dip_rows[i - 1] if i else {},
                               value, {s: q * prices[s] for s, q in held.items()})
        for s, q in swing.orders(target, value, prices, held, dict.fromkeys(held, fractional)):
            cash -= q * prices[s] + abs(q) * prices[s] * cost
            held[s] += q
            trades.append((t, s, q, prices[s]))
        invested = sum(q * prices[s] for s, q in held.items())
        curve[t] = (cash + invested, invested / (cash + invested))
    return pd.DataFrame.from_dict(curve, orient="index", columns=["equity", "exposure"]), trades


def performance(equity: pd.Series) -> dict:
    """equity: account value after each daily close."""
    rets = equity.pct_change().dropna()
    yearly = (1 + rets).groupby(rets.index.year).prod() - 1
    years = len(rets) / 252
    return {"cagr": (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else 0.0,
            "vol": rets.std() * math.sqrt(252),
            "sharpe": rets.mean() / rets.std() * math.sqrt(252) if rets.std() > 0 else 0.0,
            "max_drawdown": (equity / equity.cummax() - 1).min(), "worst_year": yearly.min(), "yearly": yearly}


def load_daily_closes(client, symbols, start) -> pd.DataFrame:
    """Dividend- and split-adjusted daily closes of past sessions (today's bar is still forming), date x symbol."""
    today = pd.Timestamp.now(tz=NY).normalize()
    raw = client.get_stock_bars(StockBarsRequest(
        symbol_or_symbols=symbols, timeframe=TimeFrame.Day, start=pd.Timestamp(start, tz=NY), end=today,
        feed=DataFeed.SIP, adjustment=Adjustment.ALL)).df
    closes = raw["close"].unstack(0)
    closes.index = pd.DatetimeIndex(closes.index.tz_convert(NY).date)
    return closes[closes.index < today.tz_localize(None)].dropna()


def swing_curve(client, start: str, cost: float):
    """(curve from the first close with a 200-day average, trades, closes)."""
    closes = load_daily_closes(client, sorted(set(swing.symbols()) | {"SPY"}), start)
    curve, trades = simulate_swing(closes, cost)
    first = closes.index[config.SWING_TREND_SMA - 1]
    return curve[curve.index >= first], trades, closes


def swing_main(client, args):
    cost = (2.0 if args.cost_bps is None else args.cost_bps) / 1e4
    curve, trades, closes = swing_curve(client, args.start, cost)
    start = curve.index[0]
    equity, exposure = curve["equity"], curve["exposure"]
    spy = closes.loc[closes.index >= start, "SPY"]

    sw, bh = performance(equity), performance(spy)
    mid = equity.index[len(equity) // 2]
    print(f"Swing rules {start.date()} .. {equity.index[-1].date()}, {cost * 1e4:g} bps/side, idle cash earns nothing")
    print(f"  trend: {config.SWING_TREND_WEIGHT:.0%} {config.SWING_TREND_SYMBOL} above its {config.SWING_TREND_SMA}-day "
          f"average | dips: {swing.dip_slot():.1%} each in {', '.join(config.SWING_DIP_SYMBOLS)} at RSI(2) < "
          f"{config.SWING_DIP_RSI_BELOW}, out above the {config.SWING_DIP_EXIT_SMA}-day average")
    for name, p, eq in (("swing", sw, equity), ("buy & hold SPY", bh, spy)):
        halves = [performance(eq[eq.index < mid]), performance(eq[eq.index >= mid])]
        print(f"  {name:<15} CAGR {p['cagr']:+6.1%}  vol {p['vol']:5.1%}  Sharpe {p['sharpe']:4.2f}  "
              f"max drawdown {p['max_drawdown']:+6.1%}  worst year {p['worst_year']:+6.1%}  | first half CAGR "
              f"{halves[0]['cagr']:+6.1%}, second half {halves[1]['cagr']:+6.1%}")
    years = len(equity) / 252
    print(f"  swing: {len(trades) / years:.0f} orders/yr, {exposure.mean():.0%} of equity invested on average "
          f"(Sharpe here: daily returns, no risk-free rate)")
    table = pd.DataFrame({"swing": sw["yearly"], "buy & hold SPY": bh["yearly"]}).T
    print("\nCalendar years\n" + table.to_string(float_format=lambda x: f"{x:+.1%}"))


# ---------- momentum ----------

def load_minute_sessions(key_id, secret, symbol: str, start: str) -> list[tuple]:
    """1-minute consolidated-tape bars of each regular session (Alpaca's calendar: half days end at 13:00), from
    `start` to the last full session: [(date, opening trade, minute prices, minute VWAPs, session minutes)]."""
    today = pd.Timestamp.now(tz=NY).normalize()
    cal = TradingClient(key_id, secret, paper=True).get_calendar(GetCalendarRequest(
        start=pd.Timestamp(start).date(), end=(today - pd.Timedelta(days=1)).date()))
    client = StockHistoricalDataClient(key_id, secret)

    def fetch(q):
        s = max(q.start_time.tz_localize(NY), pd.Timestamp(start, tz=NY))
        e = min(q.end_time.tz_localize(NY), today)
        df = client.get_stock_bars(StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Minute, start=s,
                                                    end=e, feed=DataFeed.SIP)).df
        return df.loc[symbol] if len(df) else None

    quarters = pd.period_range(pd.Timestamp(start), today - pd.Timedelta(days=1), freq="Q")
    with ThreadPoolExecutor(4) as pool:
        parts = [p for p in pool.map(fetch, quarters) if p is not None]
    bars = pd.concat(parts).sort_index()
    bars = bars[~bars.index.duplicated()]
    groups = dict(tuple(bars.groupby(bars.index.tz_convert(NY).date)))
    out = []
    for c in cal:
        g = groups.get(c.date)
        if g is None:
            continue
        o_ts, c_ts = momentum.session_bounds(c)
        g = g[(g.index >= o_ts) & (g.index < c_ts)]
        if g.empty:
            continue
        n = int((c_ts - o_ts).total_seconds() // 60)
        out.append((c.date, momentum.session_open(g, o_ts), momentum.minute_prices(g, o_ts, n),
                    momentum.vwap_by_minute(g, o_ts, n), n))
    return out


def simulate_momentum(sessions: list[tuple], cost="alpaca", equity: float = START_EQUITY) -> pd.DataFrame:
    """The bot's momentum rules on load_minute_sessions() output: decisions at momentum.check_minutes(), flat at the
    close-out. cost: "alpaca" = $0.005 a share (half a 1-cent spread) + SEC fee 0.278 bps on sales, or bps per side.
    Returns per session: equity before it, return, entries."""
    past, rows, prev = [], [], None
    for day, o, prices, vwaps, n in sessions:
        if prev is not None and len(past) >= config.MOMENTUM_LOOKBACK - 4:
            sig = momentum.sigma(past)
            size = momentum.shares(equity, o)
            q, cash, fees, entries = 0, 0.0, 0.0, 0
            for m in momentum.check_minutes(n) + [None]:   # None = the close-out
                k = n - 1 - config.FLATTEN_MINUTES_BEFORE_CLOSE if m is None else m - 1
                if m is None:
                    want = 0
                elif np.isnan(sig[k]):
                    continue
                else:
                    upper, lower = momentum.bands(o, prev, sig[k])
                    want = momentum.target(prices[k], vwaps[k], upper, lower) * size
                if want != q:
                    dq, px = want - q, prices[k]
                    cash -= dq * px
                    if cost == "alpaca":
                        fees += abs(dq) * 0.005 + max(-dq, 0) * px * 0.278e-4
                    else:
                        fees += abs(dq) * px * cost / 1e4
                    entries += want != 0
                    q = want
            rows.append((day, equity, (cash - fees) / equity, entries))
            equity += cash - fees
        past = (past + [(o, prices)])[-config.MOMENTUM_LOOKBACK:]
        prev = prices[n - 1]
    res = pd.DataFrame(rows, columns=["date", "equity", "ret", "entries"]).set_index("date")
    res.index = pd.DatetimeIndex(res.index)
    return res


def from_returns(r: pd.Series) -> pd.Series:
    """Equity curve starting at 1 the day before the first return."""
    return pd.concat([pd.Series([1.0], index=[r.index[0] - pd.Timedelta(days=1)]), (1 + r).cumprod()])


def summary_line(name: str, r: pd.Series) -> str:
    p = performance(from_returns(r))
    return (f"  {name:<24} {p['cagr']:+6.1%}/yr  vol {p['vol']:5.1%}  Sharpe {p['sharpe']:4.2f}  "
            f"max drawdown {p['max_drawdown']:+6.1%}  worst year {p['worst_year']:+6.1%}")


def momentum_main(key_id, secret, args):
    sym = config.MOMENTUM_SYMBOL
    cost = "alpaca" if args.cost_bps is None else args.cost_bps
    sessions = load_minute_sessions(key_id, secret, sym, args.start)
    res = simulate_momentum(sessions, cost)
    r = res["ret"]
    hold = pd.Series([p[n - 1] for _, _, p, _, n in sessions], index=pd.DatetimeIndex([s[0] for s in sessions]))
    hold = hold.pct_change().reindex(r.index)
    print(f"Momentum day trading {sym} {r.index[0].date()} .. {r.index[-1].date()}, {config.MOMENTUM_SIZE:g}x, "
          f"costs: {cost if cost == 'alpaca' else f'{cost:g} bps/side'}, flat {config.FLATTEN_MINUTES_BEFORE_CLOSE} min "
          f"before the close")
    print(summary_line("all", r))
    print(summary_line("before May 2024", r[r.index < PUBLISHED]))
    print(summary_line("after (out of sample)", r[r.index >= PUBLISHED]))
    print(summary_line(f"buy & hold {sym}", hold.dropna()))
    traded = res[res.entries > 0]
    print(f"  trades on {len(traded) / len(res):.0%} of days, {res.entries.mean():.2f} entries a day, "
          f"{np.mean(traded.ret > 0):.0%} of those days win, average traded day {traded.ret.mean() * 1e4:+.1f} bps")
    table = pd.DataFrame({"momentum": performance(from_returns(r))["yearly"],
                          f"buy & hold {sym}": performance(from_returns(hold.dropna()))["yearly"]}).T
    print("\nCalendar years\n" + table.to_string(float_format=lambda x: f"{x:+.1%}"))


def all_main(client, key_id, secret, args):
    """The bot as configured: swing + momentum on the same account (momentum sized on the whole account's equity,
    flat overnight, so the two daily returns add)."""
    bps = args.cost_bps   # --cost-bps applies to both parts here
    curve, _, closes = swing_curve(client, args.start, (2.0 if bps is None else bps) / 1e4)
    r_swing = curve["equity"].pct_change().dropna()
    r_mom = simulate_momentum(load_minute_sessions(key_id, secret, config.MOMENTUM_SYMBOL, args.start),
                              "alpaca" if bps is None else bps)["ret"]
    r_mom = r_mom.reindex(r_swing.index).fillna(0.0)
    both = r_swing + r_mom
    spy = closes["SPY"].pct_change().reindex(r_swing.index)
    print(f"The bot as configured, {r_swing.index[0].date()} .. {r_swing.index[-1].date()} (idle cash earns nothing)")
    for name, r in (("swing", r_swing), (f"momentum {config.MOMENTUM_SYMBOL}", r_mom), ("together", both),
                    ("buy & hold SPY", spy)):
        print(summary_line(name, r))
    print(summary_line("together after May 2024", both[both.index >= PUBLISHED]))
    table = pd.DataFrame({"swing": performance(from_returns(r_swing))["yearly"],
                          "momentum": performance(from_returns(r_mom))["yearly"],
                          "together": performance(from_returns(both))["yearly"],
                          "buy & hold SPY": performance(from_returns(spy))["yearly"]}).T
    print("\nCalendar years\n" + table.to_string(float_format=lambda x: f"{x:+.1%}"))


# ---------- command line ----------

def main():
    ap = argparse.ArgumentParser(description="python backtest.py [all|swing|momentum|intraday]; see the docstring")
    ap.add_argument("mode", nargs="?", default="all", choices=("all", "swing", "momentum", "intraday"))
    ap.add_argument("--start", default="2016-01-01", help="first day of history (Alpaca's starts in 2016)")
    ap.add_argument("--days", type=int, default=60, help="intraday: days of 5-minute bars")
    ap.add_argument("--cost-bps", type=float, default=None,
                    help="stocks: spread+slippage per side, in bps (default 2 for swing, half a cent a share + SEC "
                         "fee for momentum, 5 for intraday)")
    ap.add_argument("--crypto-cost-bps", type=float, default=30.0,
                    help="crypto: fee+spread+slippage per side, in bps (Alpaca charges crypto trading fees)")
    args = ap.parse_args()

    load_dotenv()
    key_id, secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
    client = StockHistoricalDataClient(key_id, secret)
    if args.mode == "swing":
        swing_main(client, args)
        return
    if args.mode == "momentum":
        momentum_main(key_id, secret, args)
        return
    if args.mode == "all":
        all_main(client, key_id, secret, args)
        return
    args.cost_bps = 5.0 if args.cost_bps is None else args.cost_bps
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
