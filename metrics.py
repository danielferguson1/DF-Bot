"""Performance metrics shared by backtest.py and report.py."""
import math
from collections import defaultdict, deque
import pandas as pd


def round_trips(fills) -> list[float]:
    """fills: iterable of (symbol, side, qty, price, order_id) in time order.
    FIFO-matches sells to buys; returns realized P&L per sell order.
    Sells with no matching logged buy (e.g. positions opened before logging) are ignored."""
    lots = defaultdict(deque)
    pnl_by_order = {}
    for symbol, side, qty, price, order_id in fills:
        if side == "buy":
            lots[symbol].append([qty, price])
            continue
        remaining, pnl, matched = qty, 0.0, False
        while remaining > 1e-9 and lots[symbol]:
            lot = lots[symbol][0]
            take = min(lot[0], remaining)
            pnl += take * (price - lot[1])
            lot[0] -= take
            remaining -= take
            matched = True
            if lot[0] <= 1e-9:
                lots[symbol].popleft()
        if matched:
            pnl_by_order[order_id] = pnl_by_order.get(order_id, 0.0) + pnl
    return list(pnl_by_order.values())


def summarize(equity: pd.Series, trade_pnls: list[float]) -> dict:
    """equity: account value indexed by tz-aware timestamp."""
    daily = equity.resample("1D").last().dropna()
    rets = daily.pct_change().dropna()
    std = rets.std()
    sharpe = rets.mean() / std * math.sqrt(252) if len(rets) > 1 and std > 0 else 0.0
    wins = [p for p in trade_pnls if p > 0]
    return {
        "trades": len(trade_pnls),
        "win_rate": len(wins) / len(trade_pnls) if trade_pnls else 0.0,
        "avg_pnl": sum(trade_pnls) / len(trade_pnls) if trade_pnls else 0.0,
        "total_return": equity.iloc[-1] / equity.iloc[0] - 1 if len(equity) > 1 else 0.0,
        "sharpe": sharpe,
        "max_drawdown": (equity / equity.cummax() - 1).min() if len(equity) else 0.0,
    }


def print_summary(s: dict) -> None:
    print(f"  Trades:        {s['trades']}")
    print(f"  Win rate:      {s['win_rate']:.1%}")
    print(f"  Avg P&L/trade: ${s['avg_pnl']:.2f}")
    print(f"  Total return:  {s['total_return']:.2%}")
    print(f"  Sharpe (ann.): {s['sharpe']:.2f}")
    print(f"  Max drawdown:  {s['max_drawdown']:.2%}")
