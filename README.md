# qwen-alpaca-bot

Intraday paper-trading bot for Alpaca. Code-defined rules decide trades.

- **Stocks** (`STRATEGY = "orb"`, the default): opening-range breakout on stocks in play, see below.
  `STRATEGY = "sma_rsi"` switches back to the SMA/RSI rules on `WATCHLIST`, where Qwen (via Ollama) can veto a
  BUY after reading the last 24h of headlines.
- **Crypto** (BTC/USD): the SMA/RSI rules, 24/7, either way.

| File | What it does |
|---|---|
| `config.py` | Every setting: watchlist, risk limits, ATR stop/target, spread limit, feed |
| `orb.py` | Stocks in play: universe, ranking, entry/stop prices, sizing (pure functions) |
| `orb_report.py` | How ORB fills on paper: entry and stop slippage, realized R per trade |
| `strategy.py` | SMA/RSI indicators, BUY/SELL rules, entry window, cooldown, reward:risk, sizing |
| `bot.py` | Live loop: scans at each 5-min bar close, places bracket orders, logs, alerts |
| `backtest.py` | Replays the same rules on history and compares with buy-and-hold SPY |
| `report.py` | Paper-account performance (win rate, Sharpe, max drawdown) |
| `db.py` | SQLite log in `trades.db`: decisions, orders, fills |
| `test_strategy.py`, `test_bot.py`, `test_orb.py` | Unit tests; offline, partly on real AAPL bars in `fixtures/` |

## Setup

```
pip install -r requirements.txt
ollama pull qwen2.5:7b
```

`.env`:
```
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
# earnings calendar (free key from alphavantage.co; without it the earnings check is off)
ALPHAVANTAGE_API_KEY=...
# optional Telegram alerts (fills, errors, risk events)
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
```

## Use, in this order

1. `python -m unittest -v`
2. `python backtest.py --days 60 --cost-bps 5`. If the strategy doesn't beat buy-and-hold
   after costs, change the rules in `strategy.py` before trading it.
3. `python bot.py` on paper for a few weeks. Press `q` to stop. Start it before 9:00 ET on trading days.
4. `python orb_report.py` (ORB fills) and `python report.py --period 1M` (account) to check results.

## Stocks in play (ORB)

From Zarattini, Barbon & Aziz (2023), "A Profitable Day Trading Strategy for the U.S. Equity Market".

1. **9:00 ET**: the universe = US stocks with prior close > $5, 14-day average volume >= 1M shares and
   14-day ATR > $0.50 (about 1,700 names), plus each one's 14-day average 9:30-9:35 volume. Takes about a minute.
2. **9:35 ET**: rank by today's 9:30-9:35 volume / its average. For the top 20 above 1x: if that bar closed up,
   a **buy stop** 1 cent above its high; if down, a **sell-short stop** 1 cent below its low (easy-to-borrow
   names only). Skipped if the price is already through the trigger, or if it wouldn't fit the daily risk
   budget. Size: 0.25% of equity at the stop, capped at $1,000 per position.
3. **On the fill** (websocket): a protective stop `ORB_STOP_ATR` (0.10) x ATR beyond the trigger, held by Alpaca.
   Every scan re-checks that each ORB position has its stop (right side and size).
4. **15:50**: everything still open is closed; unfilled entries are cancelled. No target.

Not applied to ORB: the earnings check, Qwen and the cooldown (stocks in play are usually
news or earnings days, which is the point). Applied: the daily loss limit (it also cancels pending ORB entries)
and the daily risk budget.

**Why paper only.** A 2-year, whole-market test (`C:\Users\06dan\qwen-bot-research`) put this setup between
-0.46R and +0.20R per trade in the holdout year after 5 bps/side, depending on fills that 1-minute bars can't
settle. Live, the bot ranks on IEX volume (the free feed), which picks different names than the full tape the
research used (6 of 19 in common on 2026-11-06). `orb_report.py` measures the real slippage and R. Expect a
few hundred closed trades (several weeks) before the number means much.

## What an SMA/RSI or crypto BUY has to pass

In order; the first one that fails is logged as the reason (`trades.db`, table `decisions`).

| gate | skips the BUY when |
|---|---|
| daily loss | equity is down `DAILY_LOSS_PCT` (2%) from yesterday's close |
| time | stocks: in the first `NO_ENTRY_MINUTES_AFTER_OPEN` (15) min or the last `NO_ENTRY_MINUTES_BEFORE_CLOSE` (30) |
| positions | already held/pending, `MAX_POSITIONS`, `MAX_PER_SECTOR`, or a long and an inverse index fund together |
| entries per day | `MAX_ENTRIES_PER_DAY` (6) entries already today, all symbols |
| cooldown | an entry or exit in that symbol in the last `SYMBOL_COOLDOWN_MIN` (30) min |
| earnings | stocks: the company reports today (Alpha Vantage calendar; can't be read = skip) |
| quote | bad quote or spread over `MAX_SPREAD_PCT` |
| size | under 1 share / crypto minimum, or over `MAX_EXPOSURE_USD` |
| reward:risk | (target - ask) / (ask - stop) under `MIN_REWARD_RISK` (1.5) |
| daily risk | today's entries at their full loss-at-stop, win or lose, plus this one, over `DAILY_RISK_PCT` (1%) |
| Qwen | the headline check vetoes it, or can't run |

The daily risk budget counts risk when an entry is sent, so 3 losing trades can't be followed by a 4th
just because the losses are still small; the unfilled part of a cancelled entry is given back.

## Morning scanner

At 9:45 ET each day the bot adds up to `SCAN_TOP` (5) of Alpaca's most-active stocks to the watchlist:
plain tickers, $10-$1000, not OTC, not leveraged/inverse/short funds, ranked by dollar volume. They get every
gate above. Each scanned stock is its own "sector", so the sector cap won't stop e.g. MU + AMD + INTC
together. The backtest can't replay this (past most-active lists aren't kept): paper-test it, or set
`USE_SCANNER = False`.

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
- Opening and closing the same day counts as a day trade. FINRA Regulatory Notice 26-10 (June 4, 2026) removed the $25k
  pattern-day-trader minimum, but brokers can phase the change in until Oct 2027: check what Alpaca applies to your
  account before trading live.
- `.env` and `trades.db` are in OneDrive and sync to the cloud. Move the folder out of
  OneDrive if you don't want that.
