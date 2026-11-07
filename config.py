"""All tunable settings in one place. Shared by bot.py, backtest.py and report.py."""
from pathlib import Path
from alpaca.data.enums import DataFeed

# Swing (on by default): index ETFs held for days to months, decided once a day near the close (swing.py).
# Backtest 2016-10 .. 2026-11 on daily bars, 2 bps/side, idle cash earning nothing: +9.0%/yr, worst drawdown -15%,
# worst year -11% (buy & hold SPY: +14.9%/yr, -34%, -18%): less return in a decade stocks boomed, half the
# drawdown. `python backtest.py` re-runs it.
SWING = True
SWING_TREND_SYMBOL = "SPY"
SWING_TREND_WEIGHT = 0.70       # of equity, held while SPY closes above its 200-day average
SWING_TREND_SMA = 200
SWING_DIP_SYMBOLS = ("SPY", "IWM", "DIA")   # not QQQ: the momentum day trading below uses it (one position per symbol)
SWING_DIP_WEIGHT = 0.30         # split equally: 10% of equity per dip position
SWING_DIP_RSI_BELOW = 10        # buy at a close with RSI(2) < 10 above the 200-day average...
SWING_DIP_EXIT_SMA = 5          # ...sell at the first close above the 5-day average
SWING_RUN_MINUTES_BEFORE_CLOSE = 15   # decide and trade from 15:45 ET (live price = today's close) until the
                                      # 15:50 close-out, which keeps swing positions
SWING_HISTORY_DAYS = 450        # calendar days of daily bars fetched for the averages

WATCHLIST = ["AAPL", "MSFT", "NVDA", "TSLA", "SPY", "SH", "QQQ", "PSQ"]   # stocks: regular hours only
CRYPTO_WATCHLIST = ["BTC/USD"]  # crypto symbols the bot knows
TRADE_CRYPTO = False            # off: on 5-minute bars the 3-ATR target (~0.45%) is smaller than Alpaca's round-trip
                                # fee (0.25% taker x 2), so every trade lost in a 2-year backtest
SECTORS = {"AAPL": "tech", "MSFT": "tech", "NVDA": "tech", "TSLA": "auto", "SPY": "index", "SH": "index",
           "QQQ": "index", "PSQ": "index", "BTC/USD": "crypto"}
REGIME_SYMBOL = "SPY"   # market regime = SPY above/below its session VWAP
INVERSE = {"SH", "PSQ"} # symbols that profit when the market falls (traded only when SPY < VWAP)

# Data: IEX is the free real-time feed. Switch to DataFeed.SIP if you pay for it.
DATA_FEED = DataFeed.IEX
BAR_MINUTES = 5

# Risk
RISK_PCT = 0.0025           # equity risked per trade (entry -> stop distance)
MAX_POSITION_USD = 2000     # cap per position (bracket orders need whole shares)
MAX_EXPOSURE_USD = 6000     # cap across all positions + pending entries
MAX_POSITIONS = 3
MAX_PER_SECTOR = 2
DAILY_LOSS_PCT = 0.02       # no new entries once equity is down 2% vs yesterday's close
DAILY_RISK_PCT = 0.01       # no new entry if today's entries, each counted at its full loss-at-stop
                            # (win or lose), plus this one would exceed 1% of yesterday's equity
MAX_ENTRIES_PER_DAY = 6     # all symbols together, New York day
SYMBOL_COOLDOWN_MIN = 30    # no re-entry in a symbol within 30 min of its last entry or exit
STOP_ATR = 1.5              # stop-loss distance in ATRs
TAKE_PROFIT_ATR = 3.0       # take-profit distance in ATRs
MIN_REWARD_RISK = 1.5       # skip if (target - entry) / (entry - stop) is below this after rounding
STOCK_COST_BPS = 2          # assumed spread + slippage per side (stocks are commission-free)
CRYPTO_COST_BPS = 25        # Alpaca crypto taker fee per side (tier 1), before the spread
MAX_COST_R = 0.25           # SMA/RSI and crypto: skip if the round trip costs more than 25% of the risk to the stop

# Execution
MAX_SPREAD_PCT = 0.002      # skip entry if bid/ask spread > 0.2%
ENTRY_TIMEOUT_MIN = 10      # cancel unfilled entry orders after this long
NO_ENTRY_MINUTES_AFTER_OPEN = 15    # stocks: no entries 9:30-9:45 ET (opening noise; indicators are still
                                    # mostly yesterday's bars then)
NO_ENTRY_MINUTES_BEFORE_CLOSE = 30
FLATTEN_MINUTES_BEFORE_CLOSE = 10   # stocks only; BTC is held overnight

# Crypto: Alpaca has no bracket orders for crypto, so the bot places a stop-limit after each
# fill (re-checked every scan) and takes profit in code at bar close.
CRYPTO_STOP_LIMIT_GAP = 0.01        # stop-limit may sell up to 1% below the stop price
CRYPTO_MIN_ORDER_USD = 10

# Qwen news veto (code decides trades; the LLM can only block a BUY)
USE_LLM_VETO = True
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:7b"

# Earnings calendar (Alpha Vantage, free key in .env as ALPHAVANTAGE_API_KEY; one request a day).
# Stocks reporting today are skipped. Fails closed like the Qwen check: if the calendar can't be read,
# stock BUYs are skipped. With no key the check is off and the bot says so at startup.
USE_EARNINGS_CHECK = True
EARNINGS_BLOCK_NEXT_DAY = False     # True also skips stocks reporting on the next trading day (only matters
                                    # if you hold overnight; stocks are flattened before the close)

# Morning scanner: once a day after the open, add Alpaca's most-active stocks to WATCHLIST.
# Leveraged, inverse and short funds are excluded by name: the bot would trade them as plain longs.
# Not backtestable (today's most-active list didn't exist in the past), so paper-test it before trusting it.
USE_SCANNER = True
SCAN_CANDIDATES = 50        # how many most-active stocks to fetch from Alpaca's screener
SCAN_TOP = 5                # how many to add after filtering
SCAN_MIN_PRICE = 10.0
SCAN_MAX_PRICE = 1000.0     # must stay under MAX_POSITION_USD to buy at least 1 share
SCAN_EXCLUDE_NAME = r"\b[23]x\b|ultra|leveraged|inverse|\bshort\b|\bbull\b|\bbear\b|warrant|\bunits?\b|\bright"

# Intraday (day trading) strategy, flat by the close: "momentum" (default) = intraday momentum on one ETF
# (Zarattini, Aziz & Barbon 2024, "Beat the Market"), "orb" = opening-range breakout on stocks in play (Zarattini,
# Barbon & Aziz 2023), "sma_rsi" = the SMA/RSI rules on WATCHLIST + scanner, or "none". All leave the swing symbols
# alone. Crypto keeps its own rules (TRADE_CRYPTO).
# ORB and SMA/RSI showed no edge after costs (C:\Users\06dan\qwen-bot-research). ORB at the settings below came out
# between -$7,100 and +$8,200 in the test year on a $10k account, depending on fills 1-minute bars can't settle.
STRATEGY = "momentum"

# Momentum (momentum.py). Backtest on 1-minute bars at these settings (1x, Alpaca costs, flat by 15:50), QQQ:
# +10.8%/yr 2016 - May 2024, +10.1%/yr after the paper came out (May 2024 - Nov 2026), worst drop -8%. The same rules
# on the paper's own SPY faded to +1.9%/yr after publication, and lost on IWM: the main risk is that QQQ fades too.
# `python backtest.py momentum` re-runs it.
MOMENTUM_SYMBOL = "QQQ"
MOMENTUM_SIZE = 1.0             # position = this x yesterday's equity / today's open (the paper's headline used up to
                                # 4x borrowed; 1x is tested above). Whole shares: short sales can't be fractional
MOMENTUM_LOOKBACK = 14          # sessions averaged for the noise area (the paper's)
MOMENTUM_CHECK_MINUTES = 30     # decide every 30 minutes from 10:00 (HH:00 and HH:30), the last one 15:30
ORB_TOP = 20                    # stocks in play per day, by relative opening volume
ORB_SIDES = ("long", "short")   # long if the 9:30 bar closed up (buy stop above its high), short if down
ORB_STOP_ATR = 0.10             # protective stop distance in 14-day daily ATRs (the paper's)
ORB_RISK_PCT = 0.0025           # equity risked per trade at the stop, before the position cap
ORB_POSITION_USD = 1000         # cap per position
ORB_MIN_PRICE = 5.0             # prior close and today's open
ORB_MIN_AVG_VOLUME = 1_000_000  # 14-day average daily volume (consolidated tape)
ORB_MIN_ATR = 0.50              # 14-day daily ATR, $
ORB_MIN_RVOL = 1.0              # today's 9:30-9:35 volume vs its 14-day average (IEX, the live feed)
ORB_PREPARE_AT = 9 * 60         # minutes after midnight ET: build the day's universe (takes a few minutes)
ORB_PLACE_BY = 9 * 60 + 45      # entries go in after the 9:30 bar closes; skipped for the day after this
ORB_WORKERS = 8                 # parallel data requests (Alpaca allows 200/min)

DB_PATH = Path(__file__).with_name("trades.db")
