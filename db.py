"""SQLite log of decisions, orders and fills. Opens a connection per call so it is safe across threads."""
import sqlite3
from contextlib import closing
from datetime import datetime, timezone

from config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (ts TEXT, symbol TEXT, action TEXT, price REAL, detail TEXT);
CREATE TABLE IF NOT EXISTS orders (ts TEXT, symbol TEXT, side TEXT, qty REAL, limit_price REAL,
                                   stop_price REAL, take_profit REAL, order_id TEXT, note TEXT);
CREATE TABLE IF NOT EXISTS fills (ts TEXT, symbol TEXT, side TEXT, qty REAL, price REAL, order_id TEXT, event TEXT);
-- risk committed by each entry (qty x entry-to-stop), and refunds for the unfilled part of cancelled entries
CREATE TABLE IF NOT EXISTS risk (ts TEXT, symbol TEXT, order_id TEXT, qty REAL, risk_usd REAL, note TEXT);
"""


def _connect():
    return sqlite3.connect(DB_PATH, timeout=10)


def init():
    with closing(_connect()) as conn:
        conn.executescript(SCHEMA)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(fills)")}
        if "client_id" not in cols:  # added for the ORB fill report; older trades.db files get it here
            conn.execute("ALTER TABLE fills ADD COLUMN client_id TEXT")
            conn.commit()


def log(table: str, **row):
    """table and column names come from code, never from user input."""
    row.setdefault("ts", datetime.now(timezone.utc).isoformat())
    cols = ", ".join(row)
    marks = ", ".join("?" * len(row))
    with closing(_connect()) as conn, conn:
        conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", list(row.values()))


def fills(since: str = ""):
    """since: UTC ISO timestamp; '' = all."""
    with closing(_connect()) as conn:
        return conn.execute("SELECT symbol, side, qty, price, order_id FROM fills WHERE ts >= ? ORDER BY ts",
                            (since,)).fetchall()


def count_since(table: str, since: str, column: str, like: str) -> int:
    """Rows in table logged since `since` whose column matches the SQL LIKE pattern."""
    with closing(_connect()) as conn:
        return conn.execute(f"SELECT COUNT(*) FROM {table} WHERE ts >= ? AND {column} LIKE ?",
                            (since, like)).fetchone()[0]


def risk_today(since: str) -> tuple[float, int]:
    """(risk committed since `since` by all entries, net of refunds; number of SMA/RSI and crypto entries since
    `since`; ORB fills are logged with note 'orb' and don't count toward MAX_ENTRIES_PER_DAY)."""
    with closing(_connect()) as conn:
        total, entries = conn.execute(
            "SELECT COALESCE(SUM(risk_usd), 0), COALESCE(SUM(risk_usd > 0 AND note = 'entry'), 0) "
            "FROM risk WHERE ts >= ?",
            (since,)).fetchone()
    return float(total), int(entries)


def entry_risk(order_id: str):
    """(qty, risk_usd) committed for an entry order (summed over its fills), or None."""
    with closing(_connect()) as conn:
        row = conn.execute("SELECT SUM(qty), SUM(risk_usd) FROM risk WHERE order_id = ? AND risk_usd > 0",
                           (order_id,)).fetchone()
    return row if row and row[0] is not None else None


def last_activity(symbol: str):
    """UTC ISO time of the symbol's latest entry (risk row) or sell fill, or None."""
    with closing(_connect()) as conn:
        return conn.execute(
            "SELECT MAX(ts) FROM (SELECT ts FROM risk WHERE symbol = ? AND risk_usd > 0 "
            "UNION ALL SELECT ts FROM fills WHERE symbol = ? AND side = 'sell')",
            (symbol, symbol)).fetchone()[0]


def orb_plans(since: str) -> dict:
    """Today's ORB entries: {symbol: (order_id, side 'long'/'short', trigger, stop, qty)}."""
    with closing(_connect()) as conn:
        rows = conn.execute("SELECT symbol, order_id, note, limit_price, stop_price, qty FROM orders "
                            "WHERE ts >= ? AND note LIKE 'entry: orb %' ORDER BY ts", (since,)).fetchall()
    return {s: (oid, note.split()[-1], trig, stop, qty) for s, oid, note, trig, stop, qty in rows}
