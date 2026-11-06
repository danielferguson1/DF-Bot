"""Paper-trading bot: US stocks during market hours, plus BTC/USD 24/7. Rules in strategy.py, settings in config.py.

Code decides trades. Qwen can only veto a BUY after reading recent headlines.
Stocks: every entry is a bracket order, so Alpaca holds the stop and target.
Crypto: Alpaca has no brackets for crypto, so each fill gets a stop-limit order (re-checked every
scan) and the take-profit is checked in code at each bar close.
"""
import csv
import io
import json
import os
import re
import threading
import time
from datetime import date, datetime, timedelta, timezone

import msvcrt
import pandas as pd
import requests
from dotenv import load_dotenv

from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.enums import MostActivesBy
from alpaca.data.historical.news import NewsClient
from alpaca.data.historical.screener import ScreenerClient
from alpaca.data.requests import (CryptoBarsRequest, CryptoLatestQuoteRequest, MostActivesRequest, NewsRequest,
                                  StockBarsRequest, StockLatestQuoteRequest, StockSnapshotRequest)
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import (AssetClass, OrderClass, OrderSide, OrderStatus, OrderType,
                                  QueryOrderStatus, TimeInForce)
from alpaca.trading.requests import (GetOrdersRequest, LimitOrderRequest, StopLimitOrderRequest,
                                     StopLossRequest, StopOrderRequest, TakeProfitRequest)
from alpaca.trading.stream import TradingStream

import config
import db
import metrics
import strategy
import timemachine

load_dotenv()
API_KEY, SECRET_KEY = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
if not (API_KEY and SECRET_KEY):
    raise SystemExit("Missing ALPACA_API_KEY / ALPACA_SECRET_KEY in .env")

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
crypto_client = CryptoHistoricalDataClient(API_KEY, SECRET_KEY)
news_client = NewsClient(API_KEY, SECRET_KEY)
screener_client = ScreenerClient(API_KEY, SECRET_KEY)
NY = "America/New_York"


# ---------- symbols ----------

def key(symbol: str) -> str:
    """Alpaca writes crypto as 'BTC/USD' in orders and data but 'BTCUSD' in positions."""
    return symbol.replace("/", "")


CRYPTO_KEYS = {key(s) for s in config.CRYPTO_WATCHLIST}


def is_crypto(symbol: str) -> bool:
    return key(symbol) in CRYPTO_KEYS


def sector(symbol: str) -> str:
    """Symbols added by the scanner have no listed sector: each gets its own, so the sector cap doesn't
    lump unrelated stocks together."""
    return {key(s): v for s, v in config.SECTORS.items()}.get(key(symbol), f"other:{key(symbol)}")


def et(ts) -> pd.Timestamp:
    return pd.Timestamp(ts).tz_convert(NY)


def et_day_start(now) -> str:
    """New York midnight of `now`, as the UTC ISO string trades.db timestamps compare against."""
    return et(now).normalize().tz_convert("UTC").isoformat()


# ---------- alerts ----------

def notify(msg: str):
    """Print, and send to Telegram if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are set."""
    print(msg)
    token, chat_id = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not (token and chat_id):
        return
    try:
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      data={"chat_id": chat_id, "text": msg}, timeout=10).raise_for_status()
    except requests.RequestException as e:
        print(f"   (alert failed: {type(e).__name__})")  # str(e) would print the token-bearing URL


# ---------- fills (event-driven via websocket) ----------

async def on_trade_update(data):
    event = getattr(data.event, "value", data.event)
    if event not in ("fill", "partial_fill"):
        return
    order = data.order
    side = getattr(order.side, "value", order.side)
    db.log("fills", symbol=key(order.symbol), side=side, qty=float(data.qty), price=float(data.price),
           order_id=str(order.id), event=event)
    notify(f"[FILL] {side.upper()} {float(data.qty):g} {order.symbol} @ ${float(data.price):,.2f} ({event})")


# ---------- market data ----------

def _prepare(raw, symbols, now, regular_hours_only: bool) -> dict:
    out = {}
    for s in symbols:
        if s not in raw.index.get_level_values(0):
            continue
        df = raw.loc[s]
        if regular_hours_only:
            df = strategy.regular_hours(df)
        df = df[df.index + timedelta(minutes=config.BAR_MINUTES) <= now]  # drop the bar still forming
        out[s] = strategy.add_indicators(df, sessions=regular_hours_only)
    return out


def fetch_stock_bars(symbols: list[str], now: datetime) -> dict:
    """One batched request; only completed regular-hours bars."""
    raw = data_client.get_stock_bars(StockBarsRequest(
        symbol_or_symbols=symbols, timeframe=TimeFrame(config.BAR_MINUTES, TimeFrameUnit.Minute),
        start=now - timedelta(days=5), feed=config.DATA_FEED)).df  # 5 days covers weekends
    return _prepare(raw, symbols, now, regular_hours_only=True)


def fetch_crypto_bars(symbols: list[str], now: datetime) -> dict:
    raw = crypto_client.get_crypto_bars(CryptoBarsRequest(
        symbol_or_symbols=symbols, timeframe=TimeFrame(config.BAR_MINUTES, TimeFrameUnit.Minute),
        start=now - timedelta(days=2))).df
    return _prepare(raw, symbols, now, regular_hours_only=False)


def latest_quote(symbol: str):
    if is_crypto(symbol):
        return crypto_client.get_crypto_latest_quote(CryptoLatestQuoteRequest(symbol_or_symbols=symbol))[symbol]
    return data_client.get_stock_latest_quote(
        StockLatestQuoteRequest(symbol_or_symbols=symbol, feed=config.DATA_FEED))[symbol]


_assets = {}


def asset_limits(symbol: str) -> tuple[float, float]:
    """(qty increment, minimum order qty) from Alpaca, cached."""
    if symbol not in _assets:
        a = trading_client.get_asset(key(symbol))
        _assets[symbol] = (float(a.min_trade_increment or 1e-9), float(a.min_order_size or 0))
    return _assets[symbol]


# ---------- Qwen news veto ----------

def llm_news_veto(symbol: str, now: datetime) -> tuple[bool, str]:
    """Ask Qwen whether recent headlines make a new long position unusually risky.
    Raises on any failure; the caller treats failure as a veto."""
    news = news_client.get_news(NewsRequest(symbols=key(symbol), start=now - timedelta(hours=24), limit=10))
    articles = news.data.get("news", [])
    if not articles:
        return False, "no headlines in last 24h"
    headlines = "\n".join(f"- {a.created_at:%Y-%m-%d %H:%M} UTC: {a.headline}" for a in articles)
    prompt = f"""You are a risk checker for a short-term long trade in {symbol}.
Recent headlines:
{headlines}

Veto the trade ONLY if the headlines show a specific near-term event risk: earnings today or
tomorrow, a trading halt, a major lawsuit or regulatory action, a guidance cut, an offering,
an exchange hack or outage, or a sharp breaking-news move. Ordinary news is not a reason to veto.

Respond ONLY with JSON: {{"veto": true or false, "reason": "short reason"}}"""
    resp = requests.post(f"{config.OLLAMA_HOST}/api/generate",
                         json={"model": config.OLLAMA_MODEL, "prompt": prompt, "format": "json", "stream": False},
                         timeout=60)
    resp.raise_for_status()
    answer = json.loads(resp.json()["response"])
    if not isinstance(answer.get("veto"), bool):
        raise ValueError(f"bad LLM answer: {answer}")
    news_check_status(None)
    return answer["veto"], str(answer.get("reason", ""))


def news_check_status(error: Exception | None):
    """Alert once when the news/Qwen check starts failing, and once when it recovers."""
    if error is not None and not state["news_down"]:
        state["news_down"] = True
        notify(f"[!] News/Qwen check failing ({type(error).__name__}: {error}). "
               f"Every BUY is skipped until it works again. Is Ollama running?")
    elif error is None and state["news_down"]:
        state["news_down"] = False
        notify("[OK] News/Qwen check is working again.")


def check_ollama():
    """Quick reachability + model check, so a dead Ollama is reported before the first BUY signal."""
    try:
        resp = requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=5)
        resp.raise_for_status()
        names = {m.get("name") for m in resp.json().get("models", [])}
        if config.OLLAMA_MODEL not in names:
            raise ValueError(f"model {config.OLLAMA_MODEL} not pulled (found: {', '.join(sorted(names)) or 'none'})")
    except Exception as e:
        news_check_status(e)
        return
    news_check_status(None)


def service_status(name: str, error: Exception | None, down_msg: str):
    """Alert once when a fail-closed check (earnings calendar, ...) starts failing, and once when it recovers."""
    flag = f"{name}_down"
    if error is not None and not state.get(flag):
        state[flag] = True
        notify(f"[!] {name} failing ({type(error).__name__}: {error}). {down_msg}")
    elif error is None and state.get(flag):
        state[flag] = False
        notify(f"[OK] {name} is working again.")


# ---------- earnings calendar (code, not the LLM) ----------

EARNINGS_URL = "https://www.alphavantage.co/query"
_earnings = {}  # New York date fetched -> {report date: set of symbols}


def parse_earnings(text: str) -> dict:
    """Alpha Vantage EARNINGS_CALENDAR CSV -> {date: {symbols}}. Errors and rate limits come back as JSON
    instead of CSV, and must not read as 'nobody reports'."""
    text = text.strip()
    if not text.startswith("symbol,"):
        raise ValueError(f"unexpected calendar response: {text[:200]}")
    out = {}
    for row in csv.DictReader(io.StringIO(text)):
        if row.get("symbol") and row.get("reportDate"):
            out.setdefault(date.fromisoformat(row["reportDate"]), set()).add(row["symbol"].strip().upper())
    return out


def earnings_calendar(today) -> dict:
    """The next 3 months of reports, fetched once per New York day (the free tier allows 25 calls a day)."""
    if today not in _earnings:
        resp = requests.get(EARNINGS_URL, timeout=20, params={
            "function": "EARNINGS_CALENDAR", "horizon": "3month", "apikey": os.getenv("ALPHAVANTAGE_API_KEY")})
        resp.raise_for_status()
        _earnings.clear()
        _earnings[today] = parse_earnings(resp.text)
    return _earnings[today]


def earnings_check_on() -> bool:
    return config.USE_EARNINGS_CHECK and bool(os.getenv("ALPHAVANTAGE_API_KEY"))


def earnings_block(symbol: str, ctx) -> str | None:
    """'reports today' / 'reports <date>' if the stock has earnings in the blocked window, else None.
    Raises if the calendar can't be read (the caller fails closed)."""
    today = et(ctx["now"]).date()
    cal = earnings_calendar(today)
    days = [today] + ([ctx["next_session"]] if config.EARNINGS_BLOCK_NEXT_DAY else [])
    for i, d in enumerate(days):
        if key(symbol) in cal.get(d, set()):
            return "reports today" if i == 0 else f"reports {d:%a %m-%d}"
    return None


# ---------- time machine ----------

_tm_bars = {}  # (symbol, New York date) -> regular-hours 5-minute bars of the last TM_DAYS sessions


def time_machine(symbol: str, entry: float, stop: float, target: float, now) -> dict | None:
    k = (symbol, et(now).date())
    if k not in _tm_bars:
        raw = data_client.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=symbol, timeframe=TimeFrame(config.BAR_MINUTES, TimeFrameUnit.Minute),
            start=now - timedelta(days=int(config.TM_DAYS * 1.5) + 7), end=et(now).normalize(),
            feed=config.DATA_FEED)).df  # end at today's midnight: only past sessions are needed
        _tm_bars[k] = strategy.regular_hours(raw.loc[symbol]) if symbol in raw.index.get_level_values(0) \
            else raw.iloc[0:0]
    return timemachine.run(_tm_bars[k], now, entry, stop, target, replays=config.TM_REPLAYS)


# ---------- morning scanner ----------

def scan_watchlist() -> list[str]:
    """Alpaca's most-active stocks, filtered to plain, liquid, affordable US shares, ranked by dollar volume."""
    actives = screener_client.get_most_actives(
        MostActivesRequest(top=config.SCAN_CANDIDATES, by=MostActivesBy.VOLUME)).most_actives
    known = {key(s) for s in config.WATCHLIST}
    candidates = [a for a in actives if re.fullmatch(r"[A-Z]{1,5}", a.symbol) and a.symbol not in known]
    if not candidates:
        return []
    snaps = data_client.get_stock_snapshot(StockSnapshotRequest(
        symbol_or_symbols=[a.symbol for a in candidates], feed=config.DATA_FEED))
    exclude = re.compile(config.SCAN_EXCLUDE_NAME, re.IGNORECASE)
    ranked = []
    for a in candidates:
        snap = snaps.get(a.symbol)
        price = float(snap.latest_trade.price) if snap and snap.latest_trade else 0.0
        if not config.SCAN_MIN_PRICE <= price <= config.SCAN_MAX_PRICE:
            continue
        asset = trading_client.get_asset(a.symbol)
        exchange = str(getattr(asset.exchange, "value", asset.exchange))
        if (not asset.tradable or asset.asset_class != AssetClass.US_EQUITY or exchange == "OTC"
                or exclude.search(asset.name or "")):
            continue
        ranked.append((a.volume * price, a.symbol))
    return [s for _, s in sorted(ranked, reverse=True)[:config.SCAN_TOP]]


def maybe_scan(now):
    """Once a New York day, at the start of the entry window (scanned stocks can't trade before it anyway).
    Until today's scan works, no scanned symbols are watched (yesterday's list is dropped)."""
    today = et(now).date()
    if not config.USE_SCANNER or state["scan_day"] == today:
        return
    state["scanned"] = []
    if not strategy.in_entry_window(et(now)):
        return
    try:
        scanned = scan_watchlist()
    except Exception as e:
        report_symbol_error("scanner", e)  # alerts once per distinct error; retried next scan
        return
    state["symbol_errors"].pop("scanner", None)
    state["scanned"], state["scan_day"] = scanned, today
    notify(f"[SCAN] Added today: {', '.join(scanned) or 'none passed the filters'}")


# ---------- orders ----------

def open_orders(symbol: str | None = None):
    """Filtered here rather than by the API, so 'BTC/USD' and 'BTCUSD' both match."""
    orders = trading_client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
    return [o for o in orders if symbol is None or key(o.symbol) == key(symbol)]


def place_stop(symbol: str, qty: float, stop_price: float):
    """Protective sell stop. Crypto only supports stop-limit, so its limit sits CRYPTO_STOP_LIMIT_GAP below."""
    if is_crypto(symbol):
        limit = strategy.round_to_increment(stop_price * (1 - config.CRYPTO_STOP_LIMIT_GAP), 0.01)
        req = StopLimitOrderRequest(symbol=symbol, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.GTC,
                                    stop_price=stop_price, limit_price=limit)
    else:
        req = StopOrderRequest(symbol=symbol, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.GTC,
                               stop_price=stop_price)
    return trading_client.submit_order(req)


def flatten(symbol: str):
    """Cancel open orders (they reserve the shares/coins), then close the position.
    If the close fails, put a stop back so the position is never left unprotected."""
    orders = open_orders(symbol)
    stop_price = next((float(o.stop_price) for o in orders if o.side == OrderSide.SELL and o.stop_price
                       and o.type in (OrderType.STOP, OrderType.STOP_LIMIT)), None)
    for o in orders:
        trading_client.cancel_order_by_id(o.id)
    for _ in range(10):
        if not open_orders(symbol):
            break
        time.sleep(0.5)
    try:
        trading_client.close_position(key(symbol))
    except Exception as e:
        if position_gone(symbol):  # its own stop/target filled moments ago: nothing to protect
            db.log("orders", symbol=key(symbol), side="sell", note="close skipped: already closed")
            print(f"   {symbol} was already closed")
            return
        restore_stop(symbol, stop_price, e)
        raise
    db.log("orders", symbol=key(symbol), side="sell", note="close position")
    notify(f"[EXIT] Closing {symbol}")


def position_gone(symbol: str) -> bool:
    """True only if Alpaca says 404 (no position). Any other failure counts as 'still open'."""
    try:
        trading_client.get_open_position(key(symbol))
        return False
    except APIError as e:
        return e.status_code == 404
    except Exception:
        return False


def restore_stop(symbol: str, stop_price: float | None, close_error: Exception):
    msg = f"[!] Close of {symbol} failed ({type(close_error).__name__}: {close_error})."
    try:
        if stop_price is None:
            raise ValueError("no previous stop price found")
        qty = abs(float(trading_client.get_open_position(key(symbol)).qty))
        place_stop(symbol, qty, stop_price)
        notify(f"{msg} Re-placed stop at ${stop_price:,.2f}.")
    except Exception as e:
        notify(f"{msg} POSITION IS UNPROTECTED - could not re-place stop ({type(e).__name__}: {e}). "
               f"Check Alpaca now.")


def crypto_stop_price(entry: float, atr: float) -> float:
    return strategy.round_to_increment(entry - config.STOP_ATR * atr, 0.01)


def ensure_crypto_stop(symbol: str, pos, row) -> str:
    """Crypto has no brackets: make sure a held position always has a working sell stop on Alpaca."""
    sells = [o for o in open_orders(symbol) if o.side == OrderSide.SELL]
    if sells:
        # A stop-limit that triggered while price fell through its limit stays open but never fills.
        if any(o.stop_price and row.close <= float(o.stop_price) for o in sells):
            flatten(symbol)
            return "stop triggered but unfilled: exited at market"
        return "stop in place"
    entry, atr = float(pos.avg_entry_price), row.atr_14
    if strategy._nan(atr) or atr <= 0:
        notify(f"[!] {symbol} has NO STOP: ATR unavailable. Check Alpaca.")
        return "no stop: ATR unavailable"
    stop = crypto_stop_price(entry, atr)
    if row.close <= stop:
        flatten(symbol)
        return "already below stop: exited"
    qty = float(pos.qty_available or pos.qty)
    place_stop(symbol, qty, stop)
    db.log("orders", symbol=key(symbol), side="sell", qty=qty, stop_price=stop, note="crypto stop")
    notify(f"[STOP] {symbol}: stop-limit placed at ${stop:,.2f} for {qty:g}")
    return f"placed stop {stop:.2f}"


def wait_for_order(order_id, seconds: float = 10) -> float:
    """Poll until the order is done; returns filled qty.
    If it's still open after `seconds`, cancel it so it can't fill later without a stop."""
    done = {OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.EXPIRED, OrderStatus.REJECTED}
    order = trading_client.get_order_by_id(order_id)
    for _ in range(int(seconds / 0.5)):
        if order.status in done:
            return float(order.filled_qty or 0)
        time.sleep(0.5)
        order = trading_client.get_order_by_id(order_id)
    if order.status not in done:
        try:
            trading_client.cancel_order_by_id(order_id)
        except Exception:
            pass  # it may have just filled; the re-read below tells us
        time.sleep(1)
        order = trading_client.get_order_by_id(order_id)
    return float(order.filled_qty or 0)


def refund_unfilled_risk(order):
    """A cancelled entry only risks what filled: give the rest back to today's risk budget.
    (filled_qty is read before the cancel; a fill in the last instant is still counted as unfilled.)"""
    logged = db.entry_risk(str(order.id))
    if not logged:
        return
    qty, risk = logged
    unfilled = max(0.0, qty - float(getattr(order, "filled_qty", None) or 0))
    if qty > 0 and unfilled > 0:
        db.log("risk", symbol=key(order.symbol), order_id=str(order.id), qty=-unfilled,
               risk_usd=-risk * unfilled / qty, note="refund: entry cancelled")


def cancel_stale_entries(orders, now: datetime) -> list:
    """Cancel entry orders unfilled after ENTRY_TIMEOUT_MIN. Returns the entries still pending."""
    pending = []
    for o in orders:
        if o.side != OrderSide.BUY:
            continue
        submitted = o.submitted_at or o.created_at  # submitted_at is optional in the SDK
        if submitted and now - submitted > timedelta(minutes=config.ENTRY_TIMEOUT_MIN):
            trading_client.cancel_order_by_id(o.id)
            db.log("orders", symbol=key(o.symbol), side="buy", order_id=str(o.id), note="cancel stale entry")
            refund_unfilled_risk(o)
            print(f"   Cancelled stale entry for {o.symbol}")
        else:
            pending.append(o)
    return pending


def market_side(symbol: str) -> int:
    """+1 long index fund (SPY, QQQ), -1 inverse fund (SH, PSQ), 0 anything else."""
    if key(symbol) in config.INVERSE:
        return -1
    return 1 if sector(symbol) == "index" else 0


def try_enter(symbol, row, ctx) -> str:
    """Run every risk gate, then submit the entry. Returns what happened (for the log)."""
    crypto = is_crypto(symbol)
    if ctx["loss_hit"]:
        return "skip: daily loss limit"
    if not crypto and ctx["minutes_to_close"] <= config.NO_ENTRY_MINUTES_BEFORE_CLOSE:
        return "skip: too close to market close"
    if not crypto and not strategy.in_entry_window(et(ctx["now"])):
        return f"skip: first {config.NO_ENTRY_MINUTES_AFTER_OPEN} min after the open"
    if key(symbol) in ctx["pending_symbols"]:
        return "skip: entry already pending"
    holdings = set(ctx["positions"]) | ctx["pending_symbols"]
    if len(holdings) >= config.MAX_POSITIONS:
        return "skip: max positions"
    sec = sector(symbol)
    if sum(sector(s) == sec for s in holdings) >= config.MAX_PER_SECTOR:
        return f"skip: sector cap ({sec})"
    side = market_side(symbol)
    if side and any(market_side(s) == -side for s in holdings):
        return "skip: opposite index fund held (would cancel out)"
    if ctx["entries_today"] >= config.MAX_ENTRIES_PER_DAY:
        return f"skip: {config.MAX_ENTRIES_PER_DAY} entries already today"
    if not strategy.cooled_down(last_activity(symbol), ctx["now"]):
        return f"skip: cooldown ({config.SYMBOL_COOLDOWN_MIN} min since last entry/exit)"
    if not crypto and earnings_check_on():
        try:
            reason = earnings_block(symbol, ctx)
        except Exception as e:  # fail closed, like the Qwen check
            service_status("Earnings calendar", e, "Every stock BUY is skipped until it works again.")
            return f"skip: earnings check failed ({type(e).__name__})"
        service_status("Earnings calendar", None, "")
        if reason:
            return f"skip: earnings ({reason})"

    quote = latest_quote(symbol)
    bid, ask = float(quote.bid_price), float(quote.ask_price)
    if bid <= 0 or ask <= 0 or ask < bid:
        return f"skip: bad quote {bid}/{ask}"
    if (ask - bid) / ask > config.MAX_SPREAD_PCT:
        return f"skip: spread {(ask - bid) / ask:.3%}"

    if crypto:
        qty_inc, min_qty = asset_limits(symbol)
        qty = strategy.position_size(ctx["equity"], ask, row.atr_14, qty_increment=qty_inc)
        if qty < min_qty or qty * ask < config.CRYPTO_MIN_ORDER_USD:
            return "skip: size below minimum order"
    else:
        qty = strategy.position_size(ctx["equity"], ask, row.atr_14)
        if qty < 1:
            return "skip: size < 1 share"
    if ctx["exposure"] + qty * ask > config.MAX_EXPOSURE_USD:
        return "skip: exposure cap"

    # the plan, measured from the live ask (not the bar close), exactly as enter_stock/enter_crypto will set it
    stop, target = strategy.bracket_prices(ask, row.atr_14)
    if crypto:
        stop = crypto_stop_price(ask, row.atr_14)
    rr = strategy.reward_risk(ask, stop, target)
    if rr < config.MIN_REWARD_RISK:
        return f"skip: reward:risk {rr:.2f} < {config.MIN_REWARD_RISK}"
    risk = qty * (ask - stop)
    budget = ctx["last_equity"] * config.DAILY_RISK_PCT
    if ctx["risk_today"] + risk > budget:
        return (f"skip: daily risk budget (today ${ctx['risk_today']:,.0f} + this ${risk:,.0f} "
                f"> ${budget:,.0f})")

    tm_note = ""
    if not crypto and config.USE_TIME_MACHINE:
        try:
            tm = time_machine(symbol, ask, stop, target, ctx["now"])
            tm_note = " | " + timemachine.describe(tm)
        except Exception as e:
            tm, tm_note = None, f" | TM failed ({type(e).__name__}: {e})"
            if config.TM_MIN_EXPECTED_R is not None:
                return "skip: time machine failed" + tm_note
        if (config.TM_MIN_EXPECTED_R is not None and tm is not None
                and tm["mc"]["avg_r"] < config.TM_MIN_EXPECTED_R):
            return f"skip: time machine avg {tm['mc']['avg_r']:+.2f}R < {config.TM_MIN_EXPECTED_R:+.2f}R" + tm_note

    if config.USE_LLM_VETO:
        try:
            veto, reason = llm_news_veto(symbol, ctx["now"])
        except Exception as e:  # fail closed: no veto check, no trade
            news_check_status(e)
            return f"skip: news check failed ({type(e).__name__}: {e})"
        if veto:
            return f"skip: Qwen veto - {reason}" + tm_note

    result = enter_crypto(symbol, qty, ask, row, ctx) if crypto else enter_stock(symbol, qty, ask, row, ctx)
    return result + tm_note


def last_activity(symbol: str):
    """Latest entry or exit fill in this symbol, from trades.db (so it survives restarts)."""
    ts = db.last_activity(key(symbol))
    return datetime.fromisoformat(ts) if ts else None


def commit_risk(symbol, order_id, qty, risk_usd, ctx):
    db.log("risk", symbol=key(symbol), order_id=str(order_id), qty=qty, risk_usd=risk_usd, note="entry")
    ctx["risk_today"] += risk_usd
    ctx["entries_today"] += 1


def enter_stock(symbol, qty, ask, row, ctx) -> str:
    stop, tp = strategy.bracket_prices(ask, row.atr_14)
    limit = round(ask, 2)
    order = trading_client.submit_order(LimitOrderRequest(
        symbol=symbol, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.GTC,
        limit_price=limit, order_class=OrderClass.BRACKET,
        take_profit=TakeProfitRequest(limit_price=tp), stop_loss=StopLossRequest(stop_price=stop)))
    db.log("orders", symbol=symbol, side="buy", qty=qty, limit_price=limit,
           stop_price=stop, take_profit=tp, order_id=str(order.id), note="entry: bracket")
    commit_risk(symbol, order.id, qty, qty * (limit - stop), ctx)
    ctx["exposure"] += qty * ask
    ctx["pending_symbols"].add(symbol)
    notify(f"[ENTRY] BUY {qty} {symbol} limit ${ask:.2f}, stop ${stop:.2f}, target ${tp:.2f}")
    return "entered"


def enter_crypto(symbol, qty, ask, row, ctx) -> str:
    """Immediate-or-cancel limit at the ask (no stale orders), then a stop as soon as it fills."""
    limit = strategy.round_to_increment(ask, 0.01, up=True)
    order = trading_client.submit_order(LimitOrderRequest(
        symbol=symbol, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.IOC, limit_price=limit))
    db.log("orders", symbol=key(symbol), side="buy", qty=qty, limit_price=limit,
           order_id=str(order.id), note="entry: crypto ioc")
    filled = wait_for_order(order.id)
    if filled <= 0:
        return "skip: IOC entry not filled"
    ctx["exposure"] += filled * ask
    ctx["positions"][key(symbol)] = pos = trading_client.get_open_position(key(symbol))
    entry = float(pos.avg_entry_price)
    commit_risk(symbol, order.id, filled, filled * (entry - crypto_stop_price(entry, row.atr_14)), ctx)
    _, tp = strategy.bracket_prices(float(pos.avg_entry_price), row.atr_14)
    notify(f"[ENTRY] BUY {filled:g} {symbol} @ ~${float(pos.avg_entry_price):,.2f}, target ${tp:,.2f} (checked in code)")
    return "entered; " + ensure_crypto_stop(symbol, pos, row)


# ---------- main cycle ----------

state = {"loss_alert_day": None, "flatten_day": None, "flatten_time": None,
         "summary_day": None, "last_error": None, "news_down": False, "symbol_errors": {},
         "scan_day": None, "scanned": []}


def scan_symbol(symbol, df, spy_up, ctx):
    if df is None or len(df) < 21:
        return
    if not strategy.bar_is_fresh(df.index[-1], ctx["now"]):
        print(f"-> {symbol}: latest bar {df.index[-1].astimezone():%m-%d %H:%M} is stale, skipping")
        return
    crypto = is_crypto(symbol)
    prev, row = df.iloc[-2], df.iloc[-1]
    if crypto:
        regime_ok = strategy.regime_up(row)  # BTC above its own session VWAP
    else:
        regime_ok = not spy_up if symbol in config.INVERSE else spy_up
    pos = ctx["positions"].get(key(symbol))
    action = strategy.signal(prev, row, pos is not None, regime_ok)

    result = ""
    if crypto and pos is not None and action == "HOLD":
        _, tp = strategy.bracket_prices(float(pos.avg_entry_price), row.atr_14)
        if row.close >= tp:
            action, result = "SELL", "take profit; "
        else:
            result = ensure_crypto_stop(symbol, pos, row)
    if action == "SELL":
        flatten(symbol)
        result += "exited"
    elif action == "BUY":
        result = try_enter(symbol, row, ctx)

    detail = f"rsi={row.rsi_14:.1f} sma5={row.sma_5:.2f} sma20={row.sma_20:.2f} " \
             f"vwap={row.vwap:.2f} relvol={row.rel_volume:.2f} regime_ok={regime_ok} {result}"
    db.log("decisions", symbol=key(symbol), action=action, price=float(row.close), detail=detail)
    print(f"-> {symbol}: {action} | {detail}")


def run_trading_cycle(clock, stocks: bool):
    now = clock.timestamp
    minutes_to_close = (clock.next_close - now).total_seconds() / 60
    what = f"stocks ({minutes_to_close:.0f} min to close)" if stocks else "stocks closed"
    print(f"\n[{now.astimezone():%Y-%m-%d %H:%M:%S}] Scan: {what}"
          + (f", crypto {', '.join(config.CRYPTO_WATCHLIST)}" if config.CRYPTO_WATCHLIST else ""))
    if config.USE_LLM_VETO and state["news_down"]:
        check_ollama()  # sends the recovery alert once Ollama is back

    account = trading_client.get_account()
    equity, last_equity = float(account.equity), float(account.last_equity)
    loss_hit = equity <= last_equity * (1 - config.DAILY_LOSS_PCT)
    if loss_hit and state["loss_alert_day"] != now.date():
        state["loss_alert_day"] = now.date()
        notify(f"[RISK] Daily loss limit hit (equity ${equity:,.2f} vs ${last_equity:,.2f}). No new entries today.")

    positions = {key(p.symbol): p for p in trading_client.get_all_positions()}
    pending = cancel_stale_entries(open_orders(), now)
    risk_today, entries_today = db.risk_today(et_day_start(now))
    ctx = {
        "now": now, "equity": equity, "last_equity": last_equity, "loss_hit": loss_hit,
        "minutes_to_close": minutes_to_close, "next_session": et(clock.next_open).date(),
        "risk_today": risk_today, "entries_today": entries_today,
        "positions": positions,
        "pending_symbols": {key(o.symbol) for o in pending},
        "exposure": sum(abs(float(p.market_value)) for p in positions.values())
                    + sum(float(o.qty) * float(o.limit_price or 0) for o in pending),
    }

    jobs = []  # (symbol, bars, spy_up)
    if stocks:
        maybe_scan(now)
        # held stocks are always scanned, so a scanned stock still gets its SELL check after a restart
        held = [p.symbol for p in positions.values() if p.asset_class != AssetClass.CRYPTO]
        watch = list(dict.fromkeys(config.WATCHLIST + state["scanned"] + held))
        bars = fetch_stock_bars(sorted(set(watch) | {config.REGIME_SYMBOL}), now)
        spy = bars.get(config.REGIME_SYMBOL)
        if spy is None or spy.empty or not strategy.bar_is_fresh(spy.index[-1], now):
            print("   No fresh SPY bar yet (e.g. first minutes after the open), skipping stocks this scan.")
        else:
            spy_up = strategy.regime_up(spy.iloc[-1])
            jobs += [(s, bars.get(s), spy_up) for s in watch]
    if config.CRYPTO_WATCHLIST:
        cbars = fetch_crypto_bars(config.CRYPTO_WATCHLIST, now)
        jobs += [(s, cbars.get(s), None) for s in config.CRYPTO_WATCHLIST]

    for symbol, df, spy_up in jobs:
        try:
            scan_symbol(symbol, df, spy_up, ctx)
            state["symbol_errors"].pop(symbol, None)
        except Exception as e:
            report_symbol_error(symbol, e)


def report_symbol_error(symbol: str, error: Exception):
    """Alert once per distinct error per symbol, not every scan. Cleared after a clean scan."""
    msg = f"   Error on {symbol}: {type(error).__name__}: {error}"
    if state["symbol_errors"].get(symbol) == msg:
        print(msg)
        return
    state["symbol_errors"][symbol] = msg
    notify(msg)


def flatten_all_once(now):
    """End-of-day close-out of STOCKS (crypto is held overnight). Marked done only after every close works."""
    if state["flatten_day"] == now.date():
        return
    for o in open_orders():
        if o.side == OrderSide.BUY and o.asset_class != AssetClass.CRYPTO:
            trading_client.cancel_order_by_id(o.id)  # pending stock entries
            refund_unfilled_risk(o)  # crypto keeps trading tonight on the same daily budget
    failed = []
    for p in trading_client.get_all_positions():
        if p.asset_class == AssetClass.CRYPTO:
            continue
        try:
            flatten(p.symbol)
        except Exception:
            failed.append(p.symbol)  # flatten() already alerted and re-placed the stop
    if failed:
        raise RuntimeError(f"EOD close failed for {', '.join(failed)}")
    state["flatten_day"], state["flatten_time"] = now.date(), now
    notify("[EOD] Flattened all stock positions before the close.")


def maybe_send_daily_summary(now):
    """Once per day, 2+ minutes after the stock close-out (so its fills are logged)."""
    today = now.date()
    if state["flatten_day"] != today or state["summary_day"] == today:
        return
    if now - state["flatten_time"] < timedelta(minutes=2):
        return
    account = trading_client.get_account()
    equity, last_equity = float(account.equity), float(account.last_equity)
    day_pnl = equity - last_equity
    since = datetime.combine(today, datetime.min.time(), tzinfo=now.tzinfo).astimezone(timezone.utc).isoformat()
    pnls = metrics.round_trips(db.fills(since))
    wins = sum(p > 0 for p in pnls)
    entries = db.count_since("orders", since, "note", "entry:%")
    vetoes = db.count_since("decisions", since, "detail", "%Qwen veto%")
    lines = [
        f"[DAILY SUMMARY] {today:%a %Y-%m-%d}",
        f"Day P&L: ${day_pnl:+,.2f} ({day_pnl / last_equity:+.2%})" if last_equity else f"Day P&L: ${day_pnl:+,.2f}",
        f"Equity: ${equity:,.2f}",
        f"Entries: {entries} | Closed trades: {len(pnls)} | Wins: {wins}"
        + (f" ({wins / len(pnls):.0%})" if pnls else ""),
    ]
    if pnls:
        lines.append(f"Best: ${max(pnls):+,.2f} | Worst: ${min(pnls):+,.2f}")
    open_crypto = [p for p in trading_client.get_all_positions() if p.asset_class == AssetClass.CRYPTO]
    for p in open_crypto:
        lines.append(f"Holding {float(p.qty):g} {p.symbol} overnight (P&L ${float(p.unrealized_pl):+,.2f})")
    lines.append(f"Qwen vetoes: {vetoes}" + (" | News check is DOWN" if state["news_down"] else ""))
    state["summary_day"] = today
    notify("\n".join(lines))


def key_pressed() -> bool:
    while msvcrt.kbhit():
        if msvcrt.getch().decode("utf-8", errors="ignore").lower() in ("q", "s"):
            return True
    return False


def sleep_with_key_listener(seconds: float) -> bool:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if key_pressed():
            return True
        time.sleep(min(1, max(0, end - time.monotonic())))
    return False


def seconds_until_next_bar() -> float:
    """Wake 10s after each 5-minute boundary, when the just-closed bar is available."""
    period = config.BAR_MINUTES * 60
    return period - (time.time() % period) + 10


def main():
    db.init()
    stream = TradingStream(API_KEY, SECRET_KEY, paper=True)
    stream.subscribe_trade_updates(on_trade_update)
    threading.Thread(target=stream.run, daemon=True).start()

    notify("=== Bot started (paper): stocks in market hours"
           + (f", {', '.join(config.CRYPTO_WATCHLIST)} 24/7" if config.CRYPTO_WATCHLIST else "")
           + " === Press 'q' or 's' to stop.")
    if config.USE_LLM_VETO:
        check_ollama()
    if config.USE_EARNINGS_CHECK and not earnings_check_on():
        notify("[!] Earnings check is OFF: no ALPHAVANTAGE_API_KEY in .env (free key: alphavantage.co). "
               "Only Qwen's headline check will catch earnings.")
    while True:
        try:
            if key_pressed():
                break
            clock = trading_client.get_clock()
            now = clock.timestamp
            stocks = clock.is_open
            if stocks and (clock.next_close - now).total_seconds() / 60 <= config.FLATTEN_MINUTES_BEFORE_CLOSE:
                flatten_all_once(now)  # raises on failure -> retried in 60s
                stocks = False
            maybe_send_daily_summary(now)

            if stocks or config.CRYPTO_WATCHLIST:
                run_trading_cycle(clock, stocks)
            if not clock.is_open:
                print(f"Stock market closed. Next open {clock.next_open.astimezone():%Y-%m-%d %H:%M}.")
            state["last_error"] = None
            if sleep_with_key_listener(seconds_until_next_bar()):
                break

        except KeyboardInterrupt:
            break
        except Exception as err:
            msg = f"[!] {type(err).__name__}: {err}"
            if msg != state["last_error"]:  # don't spam the same alert every minute
                notify(msg + " (retrying every 60s)")
                state["last_error"] = msg
            if sleep_with_key_listener(60):
                break

    stream.stop()
    notify("=== Bot stopped ===")


if __name__ == "__main__":
    main()
