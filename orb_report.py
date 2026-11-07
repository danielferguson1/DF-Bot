"""How ORB really fills on paper, from trades.db: the numbers the backtest couldn't settle.

    python orb_report.py            # every ORB day logged
    python orb_report.py --days 20  # the last 20 calendar days

Per trade: entry slippage (fill vs trigger), stop slippage (fill vs stop price), realized R (P&L per share /
planned risk per share). The 2-year research put the paper's setup between -0.46R (worst-case fills) and +0.20R
(best case) per trade in the test year, after 5 bps/side; real fills say which end is true.
"""
import argparse
import sqlite3
from datetime import datetime, timedelta, timezone

import pandas as pd

import config
import db

ET = "America/New_York"


def load(since: str):
    with sqlite3.connect(config.DB_PATH) as conn:
        plans = pd.read_sql_query("SELECT ts, symbol, order_id, note, limit_price AS trigger, stop_price AS stop, "
                                  "qty FROM orders WHERE note LIKE 'entry: orb %' AND ts >= ?", conn, params=(since,))
        fills = pd.read_sql_query("SELECT ts, symbol, side, qty, price, order_id, client_id FROM fills "
                                  "WHERE ts >= ?", conn, params=(since,))
    return plans, fills


def trades(plans: pd.DataFrame, fills: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for p in plans.itertuples():
        side = p.note.split()[-1]
        sign = 1 if side == "long" else -1
        day = pd.Timestamp(p.ts).tz_convert(ET).date()
        f = fills[(fills.symbol == p.symbol)
                  & (pd.to_datetime(fills.ts, utc=True).dt.tz_convert(ET).dt.date == day)]
        entry = f[f.order_id == p.order_id]
        row = {"day": day, "symbol": p.symbol, "side": side, "trigger": p.trigger, "stop": p.stop,
               "planned_qty": p.qty, "filled": entry.qty.sum()}
        if row["filled"] > 0:
            e_px = (entry.qty * entry.price).sum() / entry.qty.sum()
            exits = f[(f.order_id != p.order_id) & (f.side == ("sell" if side == "long" else "buy"))
                      & (pd.to_datetime(f.ts, utc=True) >= pd.to_datetime(entry.ts, utc=True).min())]
            risk = abs(p.trigger - p.stop)
            row["entry_px"] = e_px
            row["entry_slip_bps"] = sign * (e_px - p.trigger) / p.trigger * 1e4
            if exits.qty.sum() > 0:
                x_px = (exits.qty * exits.price).sum() / exits.qty.sum()
                stopped = exits.client_id.fillna("").str.startswith("orb-s-").any()
                row.update(exit_px=x_px, exit="stop" if stopped else "close",
                           R=sign * (x_px - e_px) / risk if risk else None)
                if stopped:
                    row["stop_slip_bps"] = sign * (p.stop - x_px) / p.stop * 1e4
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=None)
    args = ap.parse_args()
    db.init()  # adds fills.client_id to an older trades.db
    since = (datetime.now(timezone.utc) - timedelta(days=args.days)).isoformat() if args.days else ""
    plans, fills = load(since)
    if plans.empty:
        print("No ORB entries logged yet.")
        return
    t = trades(plans, fills)
    done = t.dropna(subset=["R"]) if "R" in t else t.iloc[0:0]
    print(f"ORB on paper: {t.day.nunique()} days, {len(t)} entries placed, {int((t.filled > 0).sum())} filled "
          f"({(t.filled > 0).mean():.0%}), {len(done)} closed")
    if "entry_slip_bps" in t:
        print(f"Entry slippage vs trigger: median {t.entry_slip_bps.median():+.1f} bps, "
              f"mean {t.entry_slip_bps.mean():+.1f} bps (positive = paid worse)")
    if "stop_slip_bps" in t and t.stop_slip_bps.notna().any():
        print(f"Stop slippage vs stop price: median {t.stop_slip_bps.median():+.1f} bps, "
              f"mean {t.stop_slip_bps.mean():+.1f} bps")
    if len(done):
        se = done.R.std() / len(done) ** 0.5 if len(done) > 1 else float("nan")
        print(f"Realized R per trade: {done.R.mean():+.3f} (+-{se:.3f}), win rate {(done.R > 0).mean():.0%}, "
              f"stopped {(done.exit == 'stop').mean():.0%}")
        print("Research expectation (test year, 5 bps/side): -0.46R (worst-case fills) .. +0.20R (best case).")
        if len(done) < 200:
            print(f"Only {len(done)} closed trades: too few to tell those apart yet (~200+ needed).")
    cols = [c for c in ("day", "symbol", "side", "trigger", "entry_px", "stop", "exit_px", "exit", "R",
                        "entry_slip_bps", "stop_slip_bps") if c in t]
    print("\n" + t[cols].tail(30).to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
