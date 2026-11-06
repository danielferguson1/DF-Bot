"""All tunable settings in one place. Shared by bot.py, backtest.py and report.py."""
from pathlib import Path
from alpaca.data.enums import DataFeed

WATCHLIST = ["AAPL", "MSFT", "NVDA", "TSLA", "SPY", "SH", "QQQ", "PSQ"]   # stocks: regular hours only
CRYPTO_WATCHLIST = ["BTC/USD"]                             # 24/7; set to [] to turn crypto off
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

# Time machine: before each stock BUY, replay the plan (same stop/target in %) on the symbol's last
# TM_DAYS of 5-minute bars: TM_REPLAYS simulated rest-of-days built from bars at the same time of day,
# plus every real past session entered at the same time. Logged with each decision.
USE_TIME_MACHINE = True
TM_DAYS = 60
TM_REPLAYS = 10_000
TM_MIN_EXPECTED_R = None    # None = log only. A number (e.g. 0.0) skips BUYs whose simulated average R
                            # is below it.

# Morning scanner: once a day after the open, add Alpaca's most-active stocks to WATCHLIST.
# Leveraged, inverse and short funds are excluded by name: the bot would trade them as plain longs.
# Not backtestable (today's most-active list didn't exist in the past), so paper-test it before trusting it.
USE_SCANNER = True
SCAN_CANDIDATES = 50        # how many most-active stocks to fetch from Alpaca's screener
SCAN_TOP = 5                # how many to add after filtering
SCAN_MIN_PRICE = 10.0
SCAN_MAX_PRICE = 1000.0     # must stay under MAX_POSITION_USD to buy at least 1 share
SCAN_EXCLUDE_NAME = r"\b[23]x\b|ultra|leveraged|inverse|\bshort\b|\bbull\b|\bbear\b|warrant|\bunits?\b|\bright"

# Stock strategy. "orb" = opening-range breakout on stocks in play (Zarattini, Barbon & Aziz 2023); the SMA/RSI
# rules ("sma_rsi") then don't trade stocks, and WATCHLIST / scanner / time machine are not used for stocks.
# Crypto keeps its own rules either way.
# Research (2 years, whole market, C:\Users\06dan\qwen-bot-research): no edge was proven that survives realistic
# fills and costs. Run it on PAPER to measure real fills and slippage: python orb_report.py
STRATEGY = "orb"
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
