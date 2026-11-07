"""Paper-trading bot for Alpaca. Settings in config.py; code decides every trade.

Day trading (STRATEGY = "momentum", default): intraday momentum on one ETF every 30 minutes (rules in momentum.py).
Swing (default): index ETFs held for days to months, decided once a day at 15:45 ET (rules in swing.py).
Other intraday rules (STRATEGY): opening-range breakout (orb.py) or SMA/RSI (strategy.py), flat by the close.
Crypto (TRADE_CRYPTO): SMA/RSI on BTC/USD 24/7. Alpaca has no brackets for crypto, so each fill gets a stop-limit
order (re-checked every scan) and the take-profit is checked in code at each bar close.
Qwen can only veto an SMA/RSI or crypto BUY after reading recent headlines.
"""
import csv
import io
import json
import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone

import msvcrt
import pandas as pd
import requests
from dotenv import load_dotenv

from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.enums import Adjustment, DataFeed, MostActivesBy
from alpaca.data.historical.news import NewsClient
from alpaca.data.historical.screener import ScreenerClient
from alpaca.data.requests import (CryptoBarsRequest, CryptoLatestQuoteRequest, MostActivesRequest, NewsRequest,
                                  StockBarsRequest, StockLatestQuoteRequest, StockSnapshotRequest)
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import (AssetClass, AssetStatus, OrderClass, OrderSide, OrderStatus, OrderType,
                                  QueryOrderStatus, TimeInForce)
from alpaca.trading.requests import (GetAssetsRequest, GetCalendarRequest, GetOrdersRequest, LimitOrderRequest,
                                     MarketOrderRequest, StopLimitOrderRequest, StopLossRequest, StopOrderRequest,
                                     TakeProfitRequest)
from alpaca.trading.stream import TradingStream

import config
import db
import metrics
import momentum
import orb
import strategy
import swing

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


def crypto_symbols() -> list[str]:
    """Crypto symbols to trade: none unless TRADE_CRYPTO."""
    return config.CRYPTO_WATCHLIST if config.TRADE_CRYPTO else []


def swing_symbols() -> set:
    """Symbols the swing rules own while SWING is on: intraday rules leave them alone and the close-out keeps them."""
    return set(swing.symbols()) if config.SWING else set()


def llm_in_use() -> bool:
    """Qwen only checks SMA/RSI and crypto BUYs."""
    return config.USE_LLM_VETO and (config.STRATEGY == "sma_rsi" or bool(crypto_symbols()))


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
    client_id = getattr(order, "client_order_id", None) or ""
    db.log("fills", symbol=key(order.symbol), side=side, qty=float(data.qty), price=float(data.price),
           order_id=str(order.id), event=event, client_id=client_id)
    notify(f"[FILL] {side.upper()} {float(data.qty):g} {order.symbol} @ ${float(data.price):,.2f} ({event})")
    if client_id.startswith(ORB_ENTRY):
        # the protective stop goes in now, not at the next 5-minute scan; off the websocket thread
        threading.Thread(target=ensure_orb_stop_safe, args=(order.symbol,), daemon=True).start()


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


# ---------- morning scanner ----------

def scan_watchlist() -> list[str]:
    """Alpaca's most-active stocks, filtered to plain, liquid, affordable US shares, ranked by dollar volume."""
    actives = screener_client.get_most_actives(
        MostActivesRequest(top=config.SCAN_CANDIDATES, by=MostActivesBy.VOLUME)).most_actives
    known = {key(s) for s in config.WATCHLIST} | swing_symbols()
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


# ---------- ORB: opening-range breakout on stocks in play (orb.py has the rules) ----------

ORB_ENTRY, ORB_STOP = "orb-e-", "orb-s-"   # client_order_id prefixes
_orb_lock = threading.Lock()


def is_orb(order) -> bool:
    return (getattr(order, "client_order_id", None) or "").startswith(("orb-",))


def _parallel(fn, items):
    with ThreadPoolExecutor(config.ORB_WORKERS) as pool:
        return list(pool.map(fn, items))


def _open_bar(day, symbols, feed):
    """The 9:30-9:35 bar of `day` for `symbols` -> DataFrame indexed by symbol."""
    t0 = pd.Timestamp(day).tz_localize(NY) + pd.Timedelta(hours=9, minutes=30)

    def one(batch):
        df = data_client.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=batch, timeframe=TimeFrame(5, TimeFrameUnit.Minute),
            start=t0, end=t0 + pd.Timedelta(minutes=5), feed=feed)).df
        if not len(df):
            return None
        df = df.reset_index()
        return df[df["timestamp"] == t0.tz_convert("UTC")].set_index("symbol")
    parts = [x for x in _parallel(one, [symbols[i:i + 400] for i in range(0, len(symbols), 400)]) if x is not None]
    return pd.concat(parts)[["open", "high", "low", "close", "volume"]] if parts else pd.DataFrame()


def orb_prepare(today, past_sessions: list):
    """The day's universe and each name's 14-day average opening volume. ~3-5 minutes; once per day.
    past_sessions: the last 14+ trading dates before today."""
    t0 = time.monotonic()
    assets = trading_client.get_all_assets(GetAssetsRequest(asset_class=AssetClass.US_EQUITY,
                                                            status=AssetStatus.ACTIVE))
    plain = {a.symbol: a for a in assets if a.tradable and re.fullmatch(r"[A-Z]{1,5}", a.symbol)
             and str(getattr(a.exchange, "value", a.exchange)) in ("NYSE", "NASDAQ", "ARCA", "AMEX", "BATS")
             and a.symbol not in swing_symbols()}   # one Alpaca position per symbol: swing owns its ETFs
    syms = sorted(plain)
    start = pd.Timestamp(past_sessions[0]).tz_localize(NY)
    end = pd.Timestamp(today).tz_localize(NY)  # daily bars of past days only; SIP is free when > 15 min old

    def daily(batch):
        return data_client.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=batch, timeframe=TimeFrame.Day, start=start, end=end, feed=DataFeed.SIP,
            adjustment=Adjustment.SPLIT)).df
    parts = [df for df in _parallel(daily, [syms[i:i + 200] for i in range(0, len(syms), 200)]) if len(df)]
    elig = orb.eligible(pd.concat(parts)[["open", "high", "low", "close", "volume"]], today)
    names = sorted(elig.index)
    # past opening bars from IEX, the same feed the bot sees live at 9:35 (free SIP is 15 minutes late)
    opens = {d: _open_bar(d, names, config.DATA_FEED) for d in past_sessions[-14:]}
    avg = orb.avg_open_volume(opens)
    shortable = {s for s, a in plain.items() if a.shortable and a.easy_to_borrow}
    state["orb"] = {"day": today, "elig": elig, "avg": avg, "shortable": shortable}
    notify(f"[ORB] Universe ready for {today}: {len(elig)} eligible stocks, {len(avg)} with opening-volume "
           f"history ({time.monotonic() - t0:.0f}s)")


def orb_maybe_prepare(clock):
    """Once per trading day, from ORB_PREPARE_AT New York time until the entry deadline."""
    if config.STRATEGY != "orb":
        return
    now = et(clock.timestamp)
    today = now.date()
    session_today = clock.is_open or et(clock.next_open).date() == today
    minutes = now.hour * 60 + now.minute
    if (not session_today or (state.get("orb") or {}).get("day") == today
            or not config.ORB_PREPARE_AT <= minutes < config.ORB_PLACE_BY):
        return
    cal = trading_client.get_calendar(GetCalendarRequest(start=today - timedelta(days=30), end=today))
    past = [c.date for c in cal if c.date < today]
    if len(past) < 15:
        raise RuntimeError(f"only {len(past)} past sessions in Alpaca's calendar")
    orb_prepare(today, past[-15:])


def orb_place(ctx):
    """After the 9:30 bar closes: rank, plan, size and send the day's stop entries. Once per day."""
    now = et(ctx["now"])
    today, minutes = now.date(), now.hour * 60 + now.minute
    prep = state.get("orb") or {}
    if state.get("orb_placed") == today or minutes < 9 * 60 + 35:
        return
    since = et_day_start(ctx["now"])
    if db.orb_plans(since):           # placed earlier today (e.g. before a restart)
        state["orb_placed"] = today
        return
    if minutes >= config.ORB_PLACE_BY or prep.get("day") != today:
        state["orb_placed"] = today
        notify(f"[ORB] No entries today: " + ("universe not ready" if prep.get("day") != today
                                              else f"bot wasn't running at 9:35 (now {now:%H:%M})"))
        return
    if ctx["loss_hit"]:
        state["orb_placed"] = today
        return
    bars = _open_bar(today, sorted(prep["elig"].index), config.DATA_FEED)
    plans = orb.select(bars, prep["avg"], prep["elig"], prep["shortable"])
    snaps = data_client.get_stock_snapshot(StockSnapshotRequest(
        symbol_or_symbols=[p.symbol for p in plans], feed=config.DATA_FEED)) if plans else {}
    ready, skipped = [], []
    for p in plans:
        p.qty = orb.size(p, ctx["equity"])
        snap = snaps.get(p.symbol)
        price = float(snap.latest_trade.price) if snap and snap.latest_trade else None
        if p.qty < 1:
            skipped.append(f"{p.symbol} (size < 1)")
        elif price is None or orb.already_through(p, price):
            skipped.append(f"{p.symbol} ({'no price' if price is None else f'already through at {price:.2f}'})")
        else:
            ready.append(p)
    budget = ctx["last_equity"] * config.DAILY_RISK_PCT - ctx["risk_today"]
    fitting = orb.trim_to_budget(ready, budget)
    skipped += [f"{p.symbol} (daily risk budget)" for p in ready if p not in fitting]
    placed = []
    for p in fitting:
        side = OrderSide.BUY if p.side == "long" else OrderSide.SELL
        try:
            order = trading_client.submit_order(StopOrderRequest(
                symbol=p.symbol, qty=p.qty, side=side, time_in_force=TimeInForce.DAY, stop_price=p.trigger,
                client_order_id=f"{ORB_ENTRY}{today:%Y%m%d}-{p.symbol}"))
        except Exception as e:
            skipped.append(f"{p.symbol} (rejected: {str(e)[:60]})")
            continue
        db.log("orders", symbol=p.symbol, side=side.value, qty=p.qty, limit_price=p.trigger, stop_price=p.stop,
               order_id=str(order.id), note=f"entry: orb {p.side}")
        placed.append(f"{p.symbol} {p.side} {p.qty} @ {p.trigger:.2f} stop {p.stop:.2f} (rvol {p.rvol:.1f})")
    state["orb_placed"] = today
    notify(f"[ORB] {len(placed)} entries placed from {len(plans)} stocks in play:\n  " + "\n  ".join(placed or ["none"])
           + (f"\nSkipped: {', '.join(skipped)}" if skipped else ""))


def ensure_orb_stop(symbol: str):
    """Make the protective stop match the ORB position: right side, right qty, plan price. Commits the fill's
    risk to today's budget. If the price is already past the stop, closes the position instead."""
    with _orb_lock:
        plan = db.orb_plans(et_day_start(datetime.now(timezone.utc))).get(key(symbol))
        if plan is None:
            return "no ORB plan today"
        order_id, side, trigger, stop, _ = plan
        try:
            pos = trading_client.get_open_position(key(symbol))
        except APIError as e:
            if e.status_code == 404:
                return "flat"
            raise
        qty = abs(float(pos.qty))
        exit_side = OrderSide.SELL if side == "long" else OrderSide.BUY
        stops = [o for o in open_orders(symbol) if o.side == exit_side and o.stop_price]
        covered = sum(float(o.qty) for o in stops)
        entry = float(pos.avg_entry_price)
        logged = db.entry_risk(order_id)
        logged_qty = logged[0] if logged else 0.0
        if qty > logged_qty:   # new fill: its risk at the stop counts toward today's budget
            db.log("risk", symbol=key(symbol), order_id=order_id, qty=qty - logged_qty,
                   risk_usd=(qty - logged_qty) * abs(entry - stop), note="orb")
        if abs(covered - qty) < 1e-9:
            return "stop in place"
        price = float(pos.current_price or entry)
        if (side == "long" and price <= stop) or (side == "short" and price >= stop):
            flatten(symbol)
            return "already past the stop: closed"
        if not cancel_and_wait(symbol, stops):
            raise RuntimeError(f"old stop still open after cancel; {qty:g} covered by only {covered:g}")
        place_stop(symbol, qty, stop, exit_side, client_id=f"{ORB_STOP}{key(symbol)}-{uuid.uuid4().hex[:12]}")
        db.log("orders", symbol=key(symbol), side=exit_side.value, qty=qty, stop_price=stop,
               order_id=order_id, note="orb protective stop")
        notify(f"[STOP] {symbol}: ORB {side} stop at ${stop:,.2f} for {qty:g} (entry fill ${entry:,.2f}, "
               f"trigger ${trigger:,.2f})")
        return f"placed stop {stop:.2f}"


def ensure_orb_stop_safe(symbol: str):
    """For the websocket thread: never let an exception kill it; the next scan retries."""
    try:
        ensure_orb_stop(symbol)
    except Exception as e:
        notify(f"[!] ORB stop for {symbol} failed ({type(e).__name__}: {e}). Retrying at the next scan.")


def orb_cancel_pending(reason: str):
    pending = [o for o in open_orders() if is_orb(o) and (o.client_order_id or "").startswith(ORB_ENTRY)]
    for o in pending:
        trading_client.cancel_order_by_id(o.id)
    if pending:
        notify(f"[ORB] Cancelled {len(pending)} pending entries: {reason}")


def orb_cycle(ctx):
    """Every scan while stocks trade: place the day's entries once, then keep each position's stop right."""
    if ctx["loss_hit"]:
        orb_cancel_pending("daily loss limit")
    orb_place(ctx)
    plans = db.orb_plans(et_day_start(ctx["now"]))
    for sym in ctx["positions"]:
        if sym in plans:
            try:
                ensure_orb_stop(sym)
                state["symbol_errors"].pop(sym, None)
            except Exception as e:
                report_symbol_error(sym, e)


# ---------- swing: index ETFs held for days to months (swing.py has the rules) ----------

SWING_PREFIX = "swg-"   # client_order_id prefix
_fractionable = {}


def fractionable(symbol: str) -> bool:
    if symbol not in _fractionable:
        _fractionable[symbol] = bool(trading_client.get_asset(symbol).fractionable)
    return _fractionable[symbol]


def swing_closes(symbols: list[str], today) -> pd.DataFrame:
    """Daily closes of the sessions before `today` (New York date), date x symbol, adjusted for splits and dividends
    like the backtest. Consolidated tape (free once 15 minutes old)."""
    start = pd.Timestamp(today - timedelta(days=config.SWING_HISTORY_DAYS)).tz_localize(NY)
    raw = data_client.get_stock_bars(StockBarsRequest(
        symbol_or_symbols=symbols, timeframe=TimeFrame.Day, start=start, end=pd.Timestamp(today).tz_localize(NY),
        feed=DataFeed.SIP, adjustment=Adjustment.ALL)).df
    if raw.empty:
        return pd.DataFrame()
    closes = raw["close"].unstack(0)
    closes.index = pd.DatetimeIndex(closes.index.tz_convert(NY).date)
    return closes[closes.index < pd.Timestamp(today)]


def swing_cycle(ctx):
    """Once a day, from SWING_RUN_MINUTES_BEFORE_CLOSE minutes before the close until the stock close-out: targets
    from past daily closes plus the live price as today's close, then market orders for the gaps. Held overnight.
    Raises if a symbol couldn't be handled, so the main loop retries within the window (symbols already at their
    target, or with an order still open, are left alone on the retry)."""
    now = ctx["now"]
    today = et(now).date()
    if (not config.SWING or state.get("swing_day") == today
            or not config.FLATTEN_MINUTES_BEFORE_CLOSE < ctx["minutes_to_close"] <= config.SWING_RUN_MINUTES_BEFORE_CLOSE):
        return
    syms = swing.symbols()
    cal = trading_client.get_calendar(GetCalendarRequest(start=today - timedelta(days=10), end=today))
    prev_session = max(c.date for c in cal if c.date < today)
    closes = swing_closes(syms, today)
    snaps = data_client.get_stock_snapshot(StockSnapshotRequest(symbol_or_symbols=syms, feed=config.DATA_FEED))
    problems, live = [], {}
    need = config.SWING_TREND_SMA + 20
    for s in syms:
        recent = closes[s].iloc[-need:] if s in closes else pd.Series(dtype=float)
        trade = getattr(snaps.get(s), "latest_trade", None)
        # a short or gapped history would read as "below the 200-day average" (NaN) and sell: never act on one
        if len(recent) < need or recent.isna().any() or recent.index[-1].date() != prev_session:
            problems.append(f"{s}: daily bars missing or not up to {prev_session}")
        elif trade is None or float(trade.price) <= 0 or now - trade.timestamp > timedelta(minutes=15):
            problems.append(f"{s}: no fresh price")
        else:
            live[s] = float(trade.price)
    hist = closes[list(live)].copy() if live else pd.DataFrame()
    if live:
        hist.loc[pd.Timestamp(today)] = [live[s] for s in hist.columns]   # the live price stands in for today's close
    busy = {key(o.symbol) for o in open_orders()}   # orders before positions: a fill in between can't double-trade
    held = {key(p.symbol): float(p.qty) for p in trading_client.get_all_positions()}
    target = swing.today_targets(hist, ctx["equity"], {s: held.get(s, 0.0) * live[s] for s in live}) if live else {}
    plan = swing.orders(target, ctx["equity"], live, held, {s: fractionable(s) for s in live})

    lines, failed, placed = [], [], 0
    for s in live:
        x = swing.explain(hist[s])
        dip = ""
        if x["dip"]:
            trend_w = config.SWING_TREND_WEIGHT if s == config.SWING_TREND_SYMBOL and x["trend"] else 0.0
            dip = (", dip trade on" if target[s] - trend_w > swing.dip_slot() / 2
                   else ", dip trade running but not joined (it started before the bot held it)")
        lines.append(f"{s} ${x['price']:,.2f}: trend {'in' if x['trend'] else 'out'} (200d ${x['sma_trend']:,.2f}), "
                     f"RSI(2) {x['rsi2']:.0f}{dip} -> {target[s]:.1%} of equity")
        db.log("decisions", symbol=s, action=f"TARGET {target[s]:.1%}", price=x["price"],
               detail=f"swing sma{config.SWING_TREND_SMA}={x['sma_trend']:.2f} sma{config.SWING_DIP_EXIT_SMA}="
                      f"{x['sma_exit']:.2f} rsi2={x['rsi2']:.1f} trend={x['trend']} dip={x['dip']} "
                      f"held={held.get(s, 0.0):g}")
    for s, qty in plan:
        if s in busy:
            lines.append(f"{s}: an order is already open, left alone")
            continue
        side = OrderSide.BUY if qty > 0 else OrderSide.SELL
        try:
            if target[s] <= 0:
                order = trading_client.close_position(s)   # the whole position, fractional part included
            else:
                order = trading_client.submit_order(MarketOrderRequest(
                    symbol=s, qty=abs(qty), side=side, time_in_force=TimeInForce.DAY,
                    client_order_id=f"{SWING_PREFIX}{today:%Y%m%d}-{s}-{uuid.uuid4().hex[:8]}"))
        except Exception as e:
            failed.append(f"{s}: {type(e).__name__}: {e}")
            continue
        placed += 1
        db.log("orders", symbol=s, side=side.value, qty=abs(qty), limit_price=live[s], order_id=str(order.id),
               note=f"swing: market, target {target[s]:.1%}")
        lines.append(f"{side.value.upper()} {abs(qty):g} {s} (~${abs(qty) * live[s]:,.0f})")
    if live and not plan:
        lines.append("no orders: positions already match the targets")
    if placed or state.get("swing_tried") != today:   # a retry only reports what it changed
        notify("[SWING] " + "\n  ".join(lines or ["nothing to decide"]))
    state["swing_tried"] = today
    if problems or failed:
        raise RuntimeError("swing: " + "; ".join(problems + failed))
    state["swing_day"] = today


# ---------- momentum: intraday trend on one ETF (momentum.py has the rules) ----------

MOM_PREFIX = "mom-"   # client_order_id prefix
_mom = {}             # today's session times, yesterday's close and noise area; rebuilt each session


def momentum_prepare(now) -> dict:
    """Once a session: today's open and length, yesterday's close and the noise area from the last
    MOMENTUM_LOOKBACK sessions of 1-minute bars (consolidated tape, as backtested; free once 15 minutes old)."""
    today = et(now).date()
    if _mom.get("day") == today:
        return _mom
    sym = config.MOMENTUM_SYMBOL
    cal = trading_client.get_calendar(GetCalendarRequest(start=today - timedelta(days=45), end=today))
    if not cal or cal[-1].date != today:
        raise RuntimeError(f"momentum: {today} is not a session in Alpaca's calendar")
    past = cal[-1 - config.MOMENTUM_LOOKBACK:-1]
    start, end = momentum.session_bounds(past[0])[0], momentum.session_bounds(past[-1])[1]
    raw = data_client.get_stock_bars(StockBarsRequest(symbol_or_symbols=sym, timeframe=TimeFrame.Minute,
                                                      start=start, end=end, feed=DataFeed.SIP)).df
    bars = raw.loc[sym] if len(raw) else pd.DataFrame()
    t = bars.index.tz_convert(NY) if len(bars) else pd.DatetimeIndex([], tz=NY)
    sessions = []
    for c in past:
        o_ts, c_ts = momentum.session_bounds(c)
        b = bars[(t >= o_ts) & (t < c_ts)]
        minutes = int((c_ts - o_ts).total_seconds() // 60)
        if len(b) < 0.9 * minutes:   # never build the noise area on a patchy history
            raise RuntimeError(f"momentum: {sym} 1-minute bars missing on {c.date} ({len(b)} of {minutes})")
        sessions.append((momentum.session_open(b, o_ts), momentum.minute_prices(b, o_ts, minutes)))
    o_ts, c_ts = momentum.session_bounds(cal[-1])
    _mom.clear()
    _mom.update(day=today, open=o_ts, length=int((c_ts - o_ts).total_seconds() // 60),
                sigma=momentum.sigma(sessions), prev=float(sessions[-1][1][-1]), done=set(), shares=None)
    return _mom


def momentum_trade(sym: str, held: float, goal: int, price: float, today) -> list[str]:
    """Market orders from `held` to `goal` shares. A flip goes in two steps, because Alpaca won't sell more than a
    long holds (or buy more than a short owes) in one order: the close must fill before the new side opens."""
    if open_orders(sym):
        raise RuntimeError(f"momentum: an order in {sym} is still open; not adding another")
    done = []
    if held and (goal == 0 or (held > 0) != (goal > 0)):
        order = trading_client.close_position(key(sym))   # the whole position, fractional part included
        filled = wait_for_order(order.id)
        db.log("orders", symbol=key(sym), side="sell" if held > 0 else "buy", qty=abs(held), limit_price=price,
               order_id=str(order.id), note="momentum: exit")
        if filled < abs(held) - 1e-9:
            raise RuntimeError(f"momentum: closing {sym} filled {filled:g} of {abs(held):g}")
        done.append(f"{'SELL' if held > 0 else 'BUY'} {abs(held):g} to close")
        held = 0
    if goal != held:
        qty, side = abs(goal - held), OrderSide.BUY if goal > held else OrderSide.SELL
        order = trading_client.submit_order(MarketOrderRequest(
            symbol=sym, qty=qty, side=side, time_in_force=TimeInForce.DAY,
            client_order_id=f"{MOM_PREFIX}{today:%Y%m%d}-{uuid.uuid4().hex[:8]}"))
        db.log("orders", symbol=key(sym), side=side.value, qty=qty, limit_price=price, order_id=str(order.id),
               note=f"entry: momentum {'long' if goal > 0 else 'short'}")
        done.append(f"{side.value.upper()} {qty:g}")
    return done


def momentum_cycle(ctx):
    """Decides within 5 minutes of each check time, from today's 1-minute bars on the free real-time feed (IEX: the
    same decision as the consolidated tape at 99% of checks in a 2-year test): the opening trade, VWAP and the latest
    close. Trades the gap to the target. Not subject to the daily loss limit or risk budget (it wasn't backtested
    with them); its risk is the size and the trailing stops. The close-out flattens it; a position still open from an
    earlier session (the bot was off at the close-out) is closed before the first decision. Raises on missing or
    stale data, so the main loop retries in 60s while the check is still due."""
    if config.STRATEGY != "momentum":
        return
    sym = config.MOMENTUM_SYMBOL
    now = et(ctx["now"])
    st = momentum_prepare(ctx["now"])
    minutes = (now - st["open"]).total_seconds() / 60
    pos = ctx["positions"].get(key(sym))
    held = float(pos.qty) if pos else 0.0
    checks = momentum.check_minutes(st["length"])
    if held and minutes < checks[0] and not st["done"]:
        flatten(sym)
        notify(f"[MOMENTUM] Closed {held:g} {sym} left open from an earlier session (the bot missed a close-out).")
        return
    due = [m for m in checks if m <= minutes < m + 5 and m not in st["done"]]
    if not due:
        return
    m = due[-1]
    raw = data_client.get_stock_bars(StockBarsRequest(symbol_or_symbols=sym, timeframe=TimeFrame.Minute,
                                                      start=st["open"], end=ctx["now"], feed=config.DATA_FEED)).df
    bars = raw.loc[sym] if len(raw) else pd.DataFrame()
    if bars.empty or bars.index[0].tz_convert(NY) - st["open"] > timedelta(minutes=5):
        raise RuntimeError(f"momentum: no {sym} bars from today's open yet")
    last_end = bars.index[-1].tz_convert(NY) + timedelta(minutes=1)
    if now - last_end > timedelta(minutes=3):
        raise RuntimeError(f"momentum: latest {sym} bar ended at {last_end:%H:%M}, too old to decide on")
    k = min(int((last_end - st["open"]).total_seconds() // 60) - 1, st["length"] - 1)
    price = momentum.minute_prices(bars, st["open"], st["length"])[k]
    vwap = momentum.vwap_by_minute(bars, st["open"], st["length"])[k]
    o, sig = momentum.session_open(bars, st["open"]), st["sigma"][k]
    if pd.isna(sig) or pd.isna(price) or pd.isna(vwap):
        st["done"].add(m)
        print(f"-> {sym}: momentum {now:%H:%M} skipped (no noise area or price at minute {k + 1})")
        return
    if st["shares"] is None:
        st["shares"] = momentum.shares(ctx["last_equity"], o)
    upper, lower = momentum.bands(o, st["prev"], sig)
    want = momentum.target(price, vwap, upper, lower)
    goal = want * st["shares"]
    db.log("decisions", symbol=key(sym), action=f"TARGET {goal:+d}", price=price,
           detail=f"momentum {now:%H:%M} upper={upper:.2f} lower={lower:.2f} vwap={vwap:.2f} sigma={sig:.4%} "
                  f"open={o:.2f} prev={st['prev']:.2f} held={held:g}")
    where = (f"above the upper band ${upper:,.2f} and VWAP ${vwap:,.2f}" if want > 0 else
             f"below the lower band ${lower:,.2f} and VWAP ${vwap:,.2f}" if want < 0 else
             f"inside the noise area (${lower:,.2f}-${upper:,.2f}) or on the wrong side of VWAP ${vwap:,.2f}")
    if goal == held:
        st["done"].add(m)
        print(f"-> {sym}: momentum {now:%H:%M} ${price:,.2f} {where}: hold {held:g}")
        return
    done = momentum_trade(sym, held, goal, price, now.date())   # raises on a failure: retried while still due
    st["done"].add(m)
    label = f"long {goal}" if goal > 0 else f"short {-goal}" if goal < 0 else "flat"
    notify(f"[MOMENTUM] {now:%H:%M} {sym} ${price:,.2f}: {where} -> {label} ({', '.join(done)})")


# ---------- orders ----------

def open_orders(symbol: str | None = None):
    """Filtered here rather than by the API, so 'BTC/USD' and 'BTCUSD' both match."""
    orders = trading_client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
    return [o for o in orders if symbol is None or key(o.symbol) == key(symbol)]


def place_stop(symbol: str, qty: float, stop_price: float, side: OrderSide = OrderSide.SELL,
               client_id: str | None = None):
    """Protective stop: SELL for a long, BUY for a short. Crypto only supports stop-limit, so its limit sits
    CRYPTO_STOP_LIMIT_GAP below."""
    if is_crypto(symbol):
        limit = strategy.round_to_increment(stop_price * (1 - config.CRYPTO_STOP_LIMIT_GAP), 0.01)
        req = StopLimitOrderRequest(symbol=symbol, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.GTC,
                                    stop_price=stop_price, limit_price=limit)
    else:
        req = StopOrderRequest(symbol=symbol, qty=qty, side=side, time_in_force=TimeInForce.GTC,
                               stop_price=stop_price, client_order_id=client_id)
    return trading_client.submit_order(req)


def protective_stop(orders):
    """(side, stop price) of the first stop order among a held symbol's open orders, or (None, None)."""
    o = next((o for o in orders if o.stop_price and o.type in (OrderType.STOP, OrderType.STOP_LIMIT)), None)
    return (o.side, float(o.stop_price)) if o else (None, None)


def cancel_and_wait(symbol: str, orders, seconds: float = 5) -> bool:
    """Cancel `orders` and wait until Alpaca has released them: cancels are asynchronous, and until then the old
    orders still reserve the shares, so a replacement or a close can be rejected. True if all are gone."""
    ids = {str(o.id) for o in orders}
    for o in orders:
        trading_client.cancel_order_by_id(o.id)
    for _ in range(int(seconds / 0.5)):
        if not ids & {str(o.id) for o in open_orders(symbol)}:
            return True
        time.sleep(0.5)
    return False


def flatten(symbol: str):
    """Cancel open orders (they reserve the shares/coins), then close the position.
    If the close fails, put a stop back so the position is never left unprotected."""
    orders = open_orders(symbol)
    stop_side, stop_price = protective_stop(orders)
    cancel_and_wait(symbol, orders)  # if one lingers, close_position fails below and the stop is restored
    try:
        trading_client.close_position(key(symbol))
    except Exception as e:
        if position_gone(symbol):  # its own stop/target filled moments ago: nothing to protect
            db.log("orders", symbol=key(symbol), side="sell", note="close skipped: already closed")
            print(f"   {symbol} was already closed")
            return
        restore_stop(symbol, stop_price, e, stop_side or OrderSide.SELL)
        raise
    db.log("orders", symbol=key(symbol), side="close", note="close position")
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


def restore_stop(symbol: str, stop_price: float | None, close_error: Exception, side: OrderSide = OrderSide.SELL):
    msg = f"[!] Close of {symbol} failed ({type(close_error).__name__}: {close_error})."
    try:
        if stop_price is None:
            raise ValueError("no previous stop price found")
        qty = abs(float(trading_client.get_open_position(key(symbol)).qty))
        place_stop(symbol, qty, stop_price, side)
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
        if is_orb(o):          # ORB entries wait for their trigger all day; the EOD close-out cancels them
            pending.append(o)
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
    if side and any(market_side(s) == -side for s in holdings | ctx.get("swing_held", set())):
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
    cost_r = strategy.cost_in_r(ask, stop, config.CRYPTO_COST_BPS if crypto else config.STOCK_COST_BPS)
    if cost_r > config.MAX_COST_R:
        return f"skip: round-trip costs {cost_r:.2f}R > {config.MAX_COST_R}R of the risk"
    risk = qty * (ask - stop)
    budget = ctx["last_equity"] * config.DAILY_RISK_PCT
    if ctx["risk_today"] + risk > budget:
        return (f"skip: daily risk budget (today ${ctx['risk_today']:,.0f} + this ${risk:,.0f} "
                f"> ${budget:,.0f})")

    if config.USE_LLM_VETO:
        try:
            veto, reason = llm_news_veto(symbol, ctx["now"])
        except Exception as e:  # fail closed: no veto check, no trade
            news_check_status(e)
            return f"skip: news check failed ({type(e).__name__}: {e})"
        if veto:
            return f"skip: Qwen veto - {reason}"

    return enter_crypto(symbol, qty, ask, row, ctx) if crypto else enter_stock(symbol, qty, ask, row, ctx)


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
          + (f", crypto {', '.join(crypto_symbols())}" if crypto_symbols() else ""))
    if llm_in_use() and state["news_down"]:
        check_ollama()  # sends the recovery alert once Ollama is back

    account = trading_client.get_account()
    equity, last_equity = float(account.equity), float(account.last_equity)
    loss_hit = equity <= last_equity * (1 - config.DAILY_LOSS_PCT)
    if loss_hit and state["loss_alert_day"] != now.date():
        state["loss_alert_day"] = now.date()
        exempt = [n for n, on in (("swing", config.SWING), ("momentum", config.STRATEGY == "momentum")) if on]
        notify(f"[RISK] Daily loss limit hit (equity ${equity:,.2f} vs ${last_equity:,.2f}). No new ORB, SMA/RSI or "
               f"crypto entries today" + (f"; the {' and '.join(exempt)} rules still run." if exempt else "."))

    keep = swing_symbols()   # swing positions follow their own rules, not the intraday caps below
    all_positions = {key(p.symbol): p for p in trading_client.get_all_positions()}
    positions = {k: p for k, p in all_positions.items() if k not in keep}
    pending = cancel_stale_entries([o for o in open_orders() if key(o.symbol) not in keep], now)
    risk_today, entries_today = db.risk_today(et_day_start(now))
    ctx = {
        "now": now, "equity": equity, "last_equity": last_equity, "loss_hit": loss_hit,
        "minutes_to_close": minutes_to_close, "next_session": et(clock.next_open).date(),
        "risk_today": risk_today, "entries_today": entries_today,
        "positions": positions, "swing_held": set(all_positions) & keep,
        "pending_symbols": {key(o.symbol) for o in pending},
        "exposure": sum(abs(float(p.market_value)) for p in positions.values())
                    + sum(float(o.qty) * float(o.limit_price or 0) for o in pending),
    }

    jobs = []  # (symbol, bars, spy_up)
    if stocks and config.STRATEGY == "orb":
        orb_cycle(ctx)
    elif stocks and config.STRATEGY == "sma_rsi":
        maybe_scan(now)
        # held stocks are always scanned, so a scanned stock still gets its SELL check after a restart
        held = [p.symbol for p in positions.values() if p.asset_class != AssetClass.CRYPTO]
        watch = [s for s in dict.fromkeys(config.WATCHLIST + state["scanned"] + held) if key(s) not in keep]
        bars = fetch_stock_bars(sorted(set(watch) | {config.REGIME_SYMBOL}), now)
        spy = bars.get(config.REGIME_SYMBOL)
        if spy is None or spy.empty or not strategy.bar_is_fresh(spy.index[-1], now):
            print("   No fresh SPY bar yet (e.g. first minutes after the open), skipping stocks this scan.")
        else:
            spy_up = strategy.regime_up(spy.iloc[-1])
            jobs += [(s, bars.get(s), spy_up) for s in watch]
    if crypto_symbols():
        cbars = fetch_crypto_bars(crypto_symbols(), now)
        jobs += [(s, cbars.get(s), None) for s in crypto_symbols()]

    for symbol, df, spy_up in jobs:
        try:
            scan_symbol(symbol, df, spy_up, ctx)
            state["symbol_errors"].pop(symbol, None)
        except Exception as e:
            report_symbol_error(symbol, e)
    if stocks:   # last: these raise on failure so the main loop retries in 60s; one failing doesn't stop the other
        errors = []
        for step in (momentum_cycle, swing_cycle):
            try:
                step(ctx)
            except Exception as e:
                errors.append(e)
        if len(errors) == 1:
            raise errors[0]
        if errors:   # both: one message, so the main loop's repeat filter can't hide the second behind the first
            raise RuntimeError("; ".join(f"{type(e).__name__}: {e}" for e in errors))


def report_symbol_error(symbol: str, error: Exception):
    """Alert once per distinct error per symbol, not every scan. Cleared after a clean scan."""
    msg = f"   Error on {symbol}: {type(error).__name__}: {error}"
    if state["symbol_errors"].get(symbol) == msg:
        print(msg)
        return
    state["symbol_errors"][symbol] = msg
    notify(msg)


def flatten_all_once(now):
    """End-of-day close-out of intraday STOCK positions. Crypto and the swing ETFs are held overnight.
    Marked done only after every close works."""
    if state["flatten_day"] == now.date():
        return
    keep = swing_symbols()
    for o in open_orders():
        if (o.asset_class != AssetClass.CRYPTO and key(o.symbol) not in keep
                and (o.side == OrderSide.BUY or is_orb(o))):
            trading_client.cancel_order_by_id(o.id)  # pending stock entries (ORB shorts are sells)
            refund_unfilled_risk(o)  # crypto keeps trading tonight on the same daily budget
    failed, closed = [], []
    for p in trading_client.get_all_positions():
        if p.asset_class == AssetClass.CRYPTO or key(p.symbol) in keep:
            continue
        try:
            flatten(p.symbol)
            closed.append(p.symbol)
        except Exception:
            failed.append(p.symbol)  # flatten() already alerted and re-placed the stop
    if failed:
        raise RuntimeError(f"EOD close failed for {', '.join(failed)}")
    state["flatten_day"], state["flatten_time"] = now.date(), now
    msg = f"[EOD] Closed {', '.join(closed)} before the close." if closed else "[EOD] No intraday stock positions to close."
    (notify if closed else print)(msg)


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
    held = trading_client.get_all_positions()
    for p in held:
        if p.asset_class == AssetClass.CRYPTO:
            lines.append(f"Holding {float(p.qty):g} {p.symbol} overnight (P&L ${float(p.unrealized_pl):+,.2f})")
        elif key(p.symbol) in swing_symbols():
            lines.append(f"Swing: {float(p.qty):g} {p.symbol} ${float(p.market_value):,.0f} "
                         f"(unrealized ${float(p.unrealized_pl):+,.2f})")
    if llm_in_use():
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


def check_config():
    if config.STRATEGY not in ("none", "orb", "sma_rsi", "momentum"):
        raise SystemExit(f"config.STRATEGY must be 'none', 'orb', 'sma_rsi' or 'momentum', not {config.STRATEGY!r}")
    if config.STRATEGY == "momentum" and config.MOMENTUM_SYMBOL in swing_symbols():
        raise SystemExit(f"{config.MOMENTUM_SYMBOL} is both MOMENTUM_SYMBOL and a swing symbol: an account holds one "
                         f"position per symbol, so take it out of SWING_DIP_SYMBOLS / SWING_TREND_SYMBOL")
    if config.SWING and config.SWING_TREND_WEIGHT + config.SWING_DIP_WEIGHT > 1 + 1e-9:
        raise SystemExit("SWING_TREND_WEIGHT + SWING_DIP_WEIGHT is over 1: the swing rules would need margin")
    if config.SWING and config.SWING_RUN_MINUTES_BEFORE_CLOSE <= config.FLATTEN_MINUTES_BEFORE_CLOSE:
        raise SystemExit("SWING_RUN_MINUTES_BEFORE_CLOSE must be more than FLATTEN_MINUTES_BEFORE_CLOSE")


def startup_banner() -> str:
    parts = []
    if config.SWING:
        parts.append(f"swing ETFs ({config.SWING_TREND_SYMBOL} trend {config.SWING_TREND_WEIGHT:.0%}, RSI(2) dips "
                     f"{config.SWING_DIP_WEIGHT:.0%} in {', '.join(config.SWING_DIP_SYMBOLS)}; decided "
                     f"{config.SWING_RUN_MINUTES_BEFORE_CLOSE} min before the close)")
    if config.STRATEGY == "momentum":
        parts.append(f"day trading {config.MOMENTUM_SYMBOL} (intraday momentum, {config.MOMENTUM_SIZE:g}x, "
                     f"decided every {config.MOMENTUM_CHECK_MINUTES} min from 10:00)")
    elif config.STRATEGY != "none":
        parts.append(f"intraday stocks ({config.STRATEGY})")
    if crypto_symbols():
        parts.append(f"{', '.join(crypto_symbols())} 24/7")
    return f"=== Bot started (paper): {'; '.join(parts) or 'nothing enabled in config.py'} === Press 'q' or 's' to stop."


def main():
    check_config()
    db.init()
    stream = TradingStream(API_KEY, SECRET_KEY, paper=True)
    stream.subscribe_trade_updates(on_trade_update)
    threading.Thread(target=stream.run, daemon=True).start()

    notify(startup_banner())
    if llm_in_use():
        check_ollama()
    if config.STRATEGY == "sma_rsi" and config.USE_EARNINGS_CHECK and not earnings_check_on():
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
            orb_maybe_prepare(clock)  # once a day before the open; raises on failure -> retried in 60s

            if stocks or crypto_symbols():
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
