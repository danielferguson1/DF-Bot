# qwen-alpaca-bot

Paper-trading bot for Alpaca. Code-defined rules decide every trade.

| part (config.py) | default | what it does |
|---|---|---|
| **Day trading** (`STRATEGY = "momentum"`) | **on** | Intraday momentum on QQQ: every 30 minutes from 10:00, long while QQQ is above its "noise area" and VWAP, short while below both, flat otherwise; flat by 15:50. Size 1x the account |
| **Swing** (`SWING`) | **on** | Index ETFs held for days to months, decided at 15:45 ET: 70% of equity in SPY while SPY is above its 200-day average, plus 10% per RSI(2) dip trade in SPY, IWM or DIA |
| Other intraday rules (`STRATEGY = "orb"` / `"sma_rsi"`) | off | Opening-range breakout on stocks in play, or SMA/RSI on `WATCHLIST` with the Qwen news veto. One `STRATEGY` at a time |
| Crypto (`TRADE_CRYPTO`) | off | SMA/RSI rules on BTC/USD, 24/7 |

## Does it make money? (analysis of 2026-11-07)

**The bot as configured** (`python backtest.py`, 2016-10 to 2026-11, Alpaca's costs, idle cash earns nothing):

| | per year | max drawdown | worst year | Sharpe |
|---|---|---|---|---|
| day trading QQQ (momentum) | +12.1% | -8.3% | +1.2% (2016, 3 months) | 1.44 |
| swing | +8.8% | -15.0% | -11.1% (2022) | 0.94 |
| **together** | **+22.1%** | -18.5% | **+4.2%** (2022) | 1.75 |
| together, after May 2024 only | +21.8% | -7.7% | | 1.72 |
| buy & hold SPY | +14.9% | -33.8% | -18.2% (2022) | 0.87 |

The two parts complement each other: day trading is flat overnight and goes short in falling markets. In 2022 the swing part lost 11% while day trading made 17%. Together they hold up to about 2× the account in positions during the day (margin), and at most 100% overnight.

**Day trading** follows Zarattini, Aziz & Barbon (2024), "Beat the Market". The backtest reproduces the paper's published yearly returns on Alpaca's data (2023: +37.5% vs the paper's +37.2%, using its own sizing). At the bot's settings on QQQ, it made +10.8%/yr before the paper came out (2016 – May 2024) and **+10.1%/yr after** (May 2024 – Nov 2026). It made money in 10 of 11 calendar years, and both its long and short trades made money in both periods. It survives 5× higher costs (+6.5%/yr after publication) and orders filling a minute late (+7.1%). On the free IEX feed it made the same decision as the full consolidated tape at 99% of checks (and at 12 of 12 checks when Friday 2026-11-06 was replayed through the bot's own code). **The risk is that it fades:**
- On the paper's own ETF, SPY, the same rules dropped to +1.9%/yr after publication. They lost on IWM and made about zero on DIA.
- The post-publication record is only 2.5 years. A result that strong would turn up by luck about 7% of the time.
- An average trading day earns only +0.07%, so execution matters: at 2 bps of cost per side the return falls to about +2%/yr.

**Swing**: SPY above its 200-day average, plus Connors RSI(2) dip buying. Both rules were published long before the 2016-2026 test, and both held up in each half of it. Swing earns less than buy & hold in a boom decade, but with half the drawdown.

**What the bot used to trade, and why it lost money:**

| part | result |
|---|---|
| BTC, SMA/RSI on 5-minute bars | ~$0 per trade before costs. At Alpaca's 0.25% taker fee per side, **every trade lost** in the second year (the ~0.45% target is smaller than the 0.50% round-trip fee) |
| ORB on stocks in play | -0.46R to +0.20R per trade after 5 bps/side, depending on fills 1-minute bars can't settle: -$7,100 to +$8,200 a year on a $10k account |
| intraday-only in general | SPY earned 10.4%/yr overnight and 3.8%/yr during the day (2017-2026); most published intraday rules lost after costs |

**Not a guarantee.** Ten years is a short sample, published edges decay, and the paper's SPY edge already has. Run it on paper and compare the real results with the backtest before trusting it. All research scripts and tables are in `C:\Users\06dan\qwen-bot-research\README.md`.

| File | What it does |
|---|---|
| `config.py` | Every setting |
| `momentum.py` | Day-trading rules: noise area, bands, target, size (pure functions) |
| `swing.py` | Swing rules: trend, RSI(2) dips, targets, orders (pure functions) |
| `orb.py` | Stocks in play: universe, ranking, entry/stop prices, sizing (pure functions) |
| `strategy.py` | SMA/RSI indicators, BUY/SELL rules, entry window, cooldown, reward:risk, cost in R, sizing |
| `bot.py` | Live loop: wakes at each 5-minute bar close, places orders, logs, alerts |
| `backtest.py` | `all` (default): the bot as configured. `swing`, `momentum`: each in detail. `intraday`: SMA/RSI + BTC |
| `orb_report.py` | How ORB fills on paper: entry and stop slippage, realized R per trade |
| `report.py` | Paper-account performance (win rate, Sharpe, max drawdown) |
| `db.py` | SQLite log in `trades.db`: decisions, orders, fills |
| `test_*.py` | Unit tests; offline, partly on real AAPL bars in `fixtures/` |

## Setup

```
pip install -r requirements.txt
ollama pull qwen2.5:7b      # only needed for STRATEGY = "sma_rsi" or TRADE_CRYPTO
```

`.env`:
```
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
# earnings calendar for sma_rsi (free key from alphavantage.co; without it the earnings check is off)
ALPHAVANTAGE_API_KEY=...
# optional Telegram alerts (fills, errors, risk events, daily summary)
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
```

## Use, in this order

1. `python -m unittest -v`
2. `python backtest.py` (the whole bot; downloads ~1M one-minute bars, a couple of minutes). `python backtest.py
   momentum --cost-bps 1` stresses the day-trading costs; `python backtest.py intraday --days 60` replays the SMA/RSI
   and BTC rules. If a rule doesn't beat buy-and-hold after costs, don't trade it.
3. `python bot.py` on paper. Press `q` to stop. It has to be running **from before 10:00 ET until 15:50 ET** on
   trading days (start it any time before 10:00).
4. `python report.py --period 1M` to check results.

## Day trading: intraday momentum (momentum.py), `STRATEGY = "momentum"`

1. **Each morning**, from the last 14 sessions of 1-minute bars (consolidated tape, free once 15 minutes old): the
   average move from the open at each minute of the day (sigma), and yesterday's close.
2. **The noise area**: upper band = max(today's open, yesterday's close) x (1 + sigma), lower band =
   min(today's open, yesterday's close) x (1 - sigma). Inside it, price moves are noise.
3. **At 10:00, 10:30, ... 15:30** (12:30 on half days): long while QQQ is above both the upper band and today's VWAP,
   short while it is below both the lower band and VWAP, flat otherwise. Today's open, VWAP and price come from the
   free IEX feed's 1-minute bars. A flip from long to short (or back) is two market orders: the close fills first.
4. **Size**: yesterday's equity / today's open, in whole shares (short sales can't be fractional), fixed for the
   day. `MOMENTUM_SIZE` scales it; the paper's 19.6%/yr headline used up to 4x borrowed money, which is not tested
   here at the bot's costs.
5. **15:50**: the close-out closes it. If the bot was off then, the position left over is closed the next morning
   if the bot is running before 10:00; otherwise the first check trades it to that check's target.

- **Missed checks are skipped.** If the bot isn't running at a check (within 5 minutes of it), nothing happens until
  the next one. Stale or missing data raises an alert and is retried every minute while the check is still due.
- **Exempt from the daily loss limit and risk budget.** It wasn't backtested with them, and the loss limit would
  often block it on its best days (big trend days). Its risk is the size and the trailing stops (band and VWAP),
  which are checked only at the decision times, as in the paper. Its worst day in the backtest was -3.6%.
- **QQQ belongs to it.** The swing rules don't trade QQQ, and the bot refuses to start if `MOMENTUM_SYMBOL` is also a
  swing symbol.
- **Pattern day trading.** Under the old FINRA pattern-day-trader rule, an account under $25k couldn't day trade this
  often. Check that Alpaca applies the new rule (see Notes) to your account.

## Swing rules (swing.py)

1. **15:45 ET** (15 minutes before the close; 12:45 on half days): past daily closes (consolidated tape, adjusted for
   dividends like the backtest) plus the live IEX price as today's close. A symbol with a short or gapped history,
   or no price from the last 15 minutes, is left as it is. The bot alerts and retries every minute until 15:50.
2. **Trend:** 70% of equity in SPY while SPY is above its 200-day average; none of it below.
3. **Dips** (each of SPY, IWM, DIA): buy 10% of equity at a close with RSI(2) < 10 above the 200-day average; sell at
   the first close above the 5-day average. A dip trade is bought **on its entry day only**: if the bot wasn't holding
   it (it was off, or just started), it doesn't join late at a different price. SPY in both sleeves holds the sum.
4. **Orders:** fractional market orders (day), sells first; exits close the whole position. Target changes smaller
   than half a slot (5% of equity) aren't traded, so equity drift doesn't cause daily churn. About 56 orders a year.
5. **Held overnight, no stops**: the trend exit and the dip exit are the published rules. The 15:50 close-out keeps
   these positions. If the bot is off at 15:45, nothing changes that day; the next run trades to the targets.

The bot owns the swing ETFs: it buys and sells them to its targets, including shares you bought by hand, and the
intraday rules and the scanner never trade them. The daily loss limit, daily risk budget and position caps don't
apply; total swing exposure is at most 100% of equity (no margin overnight). Turning `SWING` off means the next
15:50 close-out sells them.

## Stocks in play (ORB): off, `STRATEGY = "orb"`

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

Not applied to ORB: the earnings check, Qwen, the cooldown and the cost gate (stocks in play are usually news or
earnings days, which is the point). Applied: the daily loss limit (it also cancels pending ORB entries) and the
daily risk budget.

**Why it's off:** see the analysis above. Live, the bot ranks on IEX volume (the free feed), which picks different
names than the full tape the research used (6 of 19 in common on 2026-11-06). Turn it on only to measure real paper
fills with `orb_report.py`, expect a few hundred closed trades before the number means much, and consider a much
smaller `ORB_POSITION_USD`: 20 positions of $1,000 are twice a $10k account.

## What an SMA/RSI or crypto BUY has to pass

In order; the first one that fails is logged as the reason (`trades.db`, table `decisions`).

| gate | skips the BUY when |
|---|---|
| daily loss | equity is down `DAILY_LOSS_PCT` (2%) from yesterday's close (account-wide; swing and momentum still run) |
| time | stocks: in the first `NO_ENTRY_MINUTES_AFTER_OPEN` (15) min or the last `NO_ENTRY_MINUTES_BEFORE_CLOSE` (30) |
| positions | already held/pending, `MAX_POSITIONS`, `MAX_PER_SECTOR`, or a long and an inverse index fund together (swing holdings count for this one) |
| entries per day | `MAX_ENTRIES_PER_DAY` (6) entries already today, all symbols |
| cooldown | an entry or exit in that symbol in the last `SYMBOL_COOLDOWN_MIN` (30) min |
| earnings | stocks: the company reports today (Alpha Vantage calendar; can't be read = skip) |
| quote | bad quote or spread over `MAX_SPREAD_PCT` |
| size | under 1 share / crypto minimum, or over `MAX_EXPOSURE_USD` |
| reward:risk | (target - ask) / (ask - stop) under `MIN_REWARD_RISK` (1.5) |
| costs | the round trip (`STOCK_COST_BPS` 2 / `CRYPTO_COST_BPS` 25 per side) is more than `MAX_COST_R` (25%) of the risk to the stop. This is what makes 5-minute BTC trades impossible: they cost about 2.3R |
| daily risk | today's entries at their full loss-at-stop, win or lose, plus this one, over `DAILY_RISK_PCT` (1%) |
| Qwen | the headline check vetoes it, or can't run |

The daily risk budget counts risk when an entry is sent, so 3 losing trades can't be followed by a 4th
just because the losses are still small; the unfilled part of a cancelled entry is given back.

## Morning scanner (sma_rsi only)

At 9:45 ET each day the bot adds up to `SCAN_TOP` (5) of Alpaca's most-active stocks to the watchlist:
plain tickers, $10-$1000, not OTC, not leveraged/inverse/short funds, ranked by dollar volume. They get every
gate above. Each scanned stock is its own "sector", so the sector cap won't stop e.g. MU + AMD + INTC
together. The backtest can't replay this (past most-active lists aren't kept): paper-test it, or set
`USE_SCANNER = False`.

## Run it unattended (Windows Task Scheduler)

Create a task that starts at logon (or before 9:30 ET on weekdays):
- Program: full path to `python.exe`
- Arguments: `bot.py`
- Start in: this folder
- Choose "Run only when user is logged on", so the console window and the `q` key still work.
- Settings tab: "If the task fails, restart every 5 minutes".

Turn sleep off from 9:30 to 16:00 ET: the day-trading decisions, the swing decision and the 15:50 close-out all need
the bot running. A day-trading position has no stop order on Alpaca's servers between decisions; if the PC sleeps
through the close-out, it is held overnight and closed the next morning.

## Bitcoin (BTC/USD): off, `TRADE_CRYPTO = True`

Same startup (`python bot.py`). BTC is scanned every 5 minutes 24/7.

- Entries are immediate-or-cancel limit orders at the ask, sized fractionally by the same risk rules.
- Alpaca has no bracket orders for crypto. After each fill the bot places a **stop-limit** on Alpaca
  and re-checks it every scan. The **take-profit is checked in code** at each 5-minute close,
  so it only works while the bot is running.
- A stop-limit can fail to fill if BTC drops more than `CRYPTO_STOP_LIMIT_GAP` (1%) past the
  stop price in one move.
- BTC is **not** closed at the end of the stock day. It's held overnight and on weekends.
- It counts toward `MAX_POSITIONS`, `MAX_EXPOSURE_USD` and the daily loss limit.
- **The cost gate blocks these trades at today's settings** (see above): a 1.5-ATR stop on 5-minute bars is ~0.22%,
  less than half the round-trip fee. A BTC rule would need much wider stops and targets (daily bars) to clear fees.
  If you turn crypto off while holding BTC, its stop-limit stays on Alpaca but nothing else manages it.

## Notes

- Swing entries can use fractional shares. Day-trading and other intraday entries use whole shares.
- 10 minutes before the close the bot closes every stock position in the paper account **except the swing ETFs**,
  including ones you opened by hand. Crypto positions are left alone.
- Opening and closing the same day counts as a day trade (swing trades never do). FINRA Regulatory Notice 26-10
  (June 4, 2026) removed the $25k pattern-day-trader minimum, but brokers can phase the change in until Oct 2027:
  check what Alpaca applies to your account before day trading live.
- `.env` and `trades.db` are in OneDrive and sync to the cloud. Move the folder out of
  OneDrive if you don't want that.
