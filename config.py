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
STOP_ATR = 1.5              # stop-loss distance in ATRs
TAKE_PROFIT_ATR = 3.0       # take-profit distance in ATRs

# Execution
MAX_SPREAD_PCT = 0.002      # skip entry if bid/ask spread > 0.2%
ENTRY_TIMEOUT_MIN = 10      # cancel unfilled entry orders after this long
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

DB_PATH = Path(__file__).with_name("trades.db")
