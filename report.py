"""Paper-trading performance: win rate from logged fills, Sharpe/drawdown from Alpaca equity history.

    python report.py --period 1M
"""
import argparse
import os

import pandas as pd
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetPortfolioHistoryRequest

import db
import metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="1M", help="e.g. 1W, 1M, 3M, 1A")
    args = ap.parse_args()

    load_dotenv()
    client = TradingClient(os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"), paper=True)
    hist = client.get_portfolio_history(GetPortfolioHistoryRequest(period=args.period, timeframe="1D"))
    equity = pd.Series(hist.equity, index=pd.to_datetime(hist.timestamp, unit="s", utc=True))
    equity = equity[equity > 0].dropna()

    db.init()
    pnls = metrics.round_trips(db.fills())
    print(f"Paper account, last {args.period} (win rate uses only trades logged by the bot):")
    metrics.print_summary(metrics.summarize(equity, pnls))


if __name__ == "__main__":
    main()
