# qwen-alpaca-bot

Intraday paper-trading bot for Alpaca. Code-defined rules decide trades. Qwen (via Ollama)
can only veto a BUY after reading the last 24h of headlines.

| File | What it does |
|---|---|
| `config.py` | Every setting: watchlist, risk limits, ATR stop/target, spread limit, feed |
| `strategy.py` | Indicators, BUY/SELL rules, position sizing (pure functions) |
| `bot.py` | Live loop: scans at each 5-min bar close, places bracket orders, logs, alerts |
| `backtest.py` | Replays the same rules on history and compares with buy-and-hold SPY |
| `report.py` | Paper-account performance (win rate, Sharpe, max drawdown) |
| `db.py` | SQLite log in `trades.db`: decisions, orders, fills |
| `test_strategy.py` | Unit tests for the logic above |

## Setup

```
pip install -r requirements.txt
ollama pull qwen2.5:7b
```

`.env`:
```
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
# optional Telegram alerts (fills, errors, risk events)
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
```

## Use, in this order

1. `python -m unittest -v`
2. `python backtest.py --days 60 --cost-bps 5`. If the strategy doesn't beat buy-and-hold
   after costs, change the rules in `strategy.py` before trading it.
3. `python bot.py` on paper for a few weeks. Press `q` to stop.
4. `python report.py --period 1M` to check results.

## Run it unattended (Windows Task Scheduler)

Create a task that starts at logon (or ~9:20 ET on weekdays):
- Program: full path to `python.exe`
- Arguments: `bot.py`
- Start in: this folder
- Choose "Run only when user is logged on", so the console window and the `q` key still work.
- Settings tab: "If the task fails, restart every 5 minutes".

Bracket orders stay on Alpaca's servers, so open positions keep their stop and target even if
the PC sleeps. Turn sleep off during market hours anyway, or the bot misses scans and the
end-of-day flatten.

## Bitcoin (BTC/USD)

Same startup (`python bot.py`). Stocks are scanned during market hours, BTC every 5 minutes 24/7.
Turn BTC off with `CRYPTO_WATCHLIST = []` in `config.py`.

- Entries are immediate-or-cancel limit orders at the ask, sized fractionally by the same risk rules.
- Alpaca has no bracket orders for crypto. After each fill the bot places a **stop-limit** on Alpaca
  and re-checks it every scan. The **take-profit is checked in code** at each 5-minute close,
  so it only works while the bot is running.
- A stop-limit can fail to fill if BTC drops more than `CRYPTO_STOP_LIMIT_GAP` (1%) past the
  stop price in one move.
- BTC is **not** closed at the end of the stock day. It's held overnight and on weekends.
- It counts toward `MAX_POSITIONS`, `MAX_EXPOSURE_USD` and the daily loss limit.
- Alpaca charges crypto trading fees (stocks are commission-free). The backtest assumes
  30 bps per side for BTC: `--crypto-cost-bps`.

## Notes

- Entries need whole shares (Alpaca bracket orders don't allow fractional), so the old $50
  cap is replaced by `MAX_POSITION_USD` and risk-based sizing.
- 10 minutes before the close the bot closes **every stock position** in the paper account,
  including ones you opened by hand. Crypto positions are left alone.
- Opening and closing the same day counts as a day trade. Accounts under $25k are limited
  by the pattern-day-trader rule.
- `.env` and `trades.db` are in OneDrive and sync to the cloud. Move the folder out of
  OneDrive if you don't want that.
