"""ORB tests: rules (orb.py), order handling in bot.py (Alpaca mocked), and the fill report. Offline."""
import unittest
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace as NS
from unittest import mock

import pandas as pd

import config
import orb
import orb_report
from test_bot import BotTestCase, bot  # sets dummy keys before importing bot
from alpaca.common.exceptions import APIError
from alpaca.trading.enums import AssetClass, OrderSide, OrderType
from alpaca.trading.requests import StopOrderRequest

ET = "America/New_York"
TODAY = date(2026, 11, 6)
AT_935 = datetime(2026, 11, 6, 14, 35, 10, tzinfo=timezone.utc)   # 9:35:10 ET


def daily_bars(spec, days=16):
    """spec: {symbol: (close, volume, daily range)}; `days` sessions ending the day before TODAY, plus TODAY."""
    rows = []
    dates = pd.bdate_range(end=pd.Timestamp(TODAY), periods=days + 1)
    for sym, (close, vol, rng) in spec.items():
        for d in dates:
            ts = pd.Timestamp(d).tz_localize(ET).tz_convert("UTC")
            rows.append((sym, ts, close, close + rng / 2, close - rng / 2, close, vol if d.date() != TODAY else 1))
    return pd.DataFrame(rows, columns=["symbol", "timestamp", "open", "high", "low", "close", "volume"]) \
        .set_index(["symbol", "timestamp"])


class Eligible(unittest.TestCase):
    def test_filters_and_uses_only_past_days(self):
        elig = orb.eligible(daily_bars({"OK": (50, 2e6, 2.0), "CHEAP": (4, 5e6, 1.0),
                                        "THIN": (50, 5e5, 2.0), "CALM": (50, 2e6, 0.4)}), TODAY)
        self.assertEqual(list(elig.index), ["OK"])
        self.assertEqual(elig.loc["OK", "avgvol14"], 2e6)    # today's tiny volume isn't counted
        self.assertAlmostEqual(elig.loc["OK", "atr14"], 2.0)

    def test_needs_14_days(self):
        self.assertTrue(orb.eligible(daily_bars({"NEW": (50, 2e6, 2.0)}, days=10), TODAY).empty)

    def test_average_opening_volume_needs_10_days(self):
        days = {date(2026, 10, d): pd.DataFrame({"volume": [100.0] + ([50.0] if d <= 10 else [])},
                                                index=["A"] + (["B"] if d <= 10 else [])) for d in range(1, 13)}
        avg = orb.avg_open_volume(days)
        self.assertEqual(avg.to_dict(), {"A": 100.0, "B": 50.0})
        days = {d: df.drop(index="B", errors="ignore") if d.day > 9 else df for d, df in days.items()}
        self.assertNotIn("B", orb.avg_open_volume(days))   # only 9 days of history


ELIG = pd.DataFrame({"prevclose": [50.0] * 5, "avgvol14": [2e6] * 5, "atr14": [2.0] * 5},
                    index=["UP", "DOWN", "NOSHORT", "QUIET", "DOJI"])
AVG = pd.Series({"UP": 1000.0, "DOWN": 1000.0, "NOSHORT": 1000.0, "QUIET": 1000.0, "DOJI": 1000.0})
TODAY_OPEN = pd.DataFrame({
    "open":   [50.00, 50.00, 50.00, 50.00, 50.00],
    "high":   [51.23, 50.40, 50.40, 50.50, 50.30],
    "low":    [49.90, 49.11, 49.11, 49.80, 49.70],
    "close":  [51.00, 49.20, 49.20, 50.40, 50.00],
    "volume": [5000.0, 3000.0, 4000.0, 900.0, 8000.0],
}, index=["UP", "DOWN", "NOSHORT", "QUIET", "DOJI"])


class Select(unittest.TestCase):
    def test_rank_sides_and_prices(self):
        plans = orb.select(TODAY_OPEN, AVG, ELIG, shortable={"UP", "DOWN"})
        # DOJI: rvol 8 but closed flat; NOSHORT: not shortable; QUIET: rvol 0.9 <= 1
        self.assertEqual([(p.symbol, p.side) for p in plans], [("UP", "long"), ("DOWN", "short")])
        up, down = plans
        self.assertEqual((up.trigger, up.stop), (51.24, 51.04))       # 1 cent above the high; 0.10 x $2 ATR
        self.assertEqual((down.trigger, down.stop), (49.10, 49.30))   # 1 cent below the low
        self.assertAlmostEqual(up.rvol, 5.0)

    def test_top_n(self):
        with mock.patch.object(config, "ORB_TOP", 1):
            self.assertEqual([p.symbol for p in orb.select(TODAY_OPEN, AVG, ELIG, {"DOWN"})], [])  # DOJI ranks 1st
        with mock.patch.object(config, "ORB_SIDES", ("long",)):
            self.assertEqual([p.symbol for p in orb.select(TODAY_OPEN, AVG, ELIG, {"DOWN"})], ["UP"])

    def test_size_and_helpers(self):
        p = orb.Plan("X", "long", 50.0, 49.8, 3.0, 2.0)
        self.assertEqual(orb.size(p, 100_000), 20)          # risk: 250 / 0.20 = 1250 sh; cap: $1000 / $50 = 20
        with mock.patch.object(config, "ORB_POSITION_USD", 1e9):
            self.assertEqual(orb.size(p, 100_000), 1250)
        self.assertTrue(orb.already_through(p, 50.0))
        self.assertFalse(orb.already_through(orb.Plan("X", "short", 50.0, 50.2, 3, 2), 50.01))
        p.qty, q = 20, orb.Plan("Y", "long", 10.0, 9.0, 2, 2, qty=5)   # risks $4 and $5
        self.assertEqual([x.symbol for x in orb.trim_to_budget([p, q], 8.0)], ["X"])


def make_ctx(**kw):
    ctx = {"now": AT_935, "equity": 100_000.0, "last_equity": 100_000.0, "loss_hit": False, "risk_today": 0.0,
           "entries_today": 0, "positions": {}, "pending_symbols": set(), "exposure": 0.0, "minutes_to_close": 380}
    ctx.update(kw)
    return ctx


class Place(BotTestCase):
    def setUp(self):
        super().setUp()
        bot.state.update(orb={"day": TODAY, "elig": ELIG, "avg": AVG, "shortable": {"UP", "DOWN"}}, orb_placed=None)
        bot.db.orb_plans.return_value = {}
        self.snaps = {"UP": NS(latest_trade=NS(price=51.10)), "DOWN": NS(latest_trade=NS(price=49.15))}
        for target, new in (("_open_bar", mock.Mock(return_value=TODAY_OPEN)),
                            ("data_client", mock.Mock(**{"get_stock_snapshot.return_value": self.snaps}))):
            patcher = mock.patch.object(bot, target, new)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.client.submit_order.side_effect = lambda req: NS(id=f"id-{req.symbol}")

    def test_places_stop_entries_once(self):
        bot.orb_place(make_ctx())
        reqs = [c.args[0] for c in self.client.submit_order.call_args_list]
        self.assertEqual([(r.symbol, r.side, r.qty, r.stop_price) for r in reqs],
                         [("UP", OrderSide.BUY, 19, 51.24), ("DOWN", OrderSide.SELL, 20, 49.10)])
        self.assertTrue(all(isinstance(r, StopOrderRequest) and r.client_order_id.startswith("orb-e-20261106-")
                            for r in reqs))
        logged = [c.kwargs for c in bot.db.log.call_args_list if c.args == ("orders",)]
        self.assertEqual(logged[0]["note"], "entry: orb long")
        self.assertEqual((logged[0]["limit_price"], logged[0]["stop_price"]), (51.24, 51.04))
        bot.orb_place(make_ctx(now=AT_935 + timedelta(minutes=5)))
        self.assertEqual(self.client.submit_order.call_count, 2)

    def test_skips_price_already_through_and_budget(self):
        self.snaps["UP"] = NS(latest_trade=NS(price=51.30))
        bot.orb_place(make_ctx(risk_today=999.0))   # $1000 budget left $1: DOWN's $4 risk doesn't fit
        self.client.submit_order.assert_not_called()
        self.assertIn("UP (already through at 51.30)", self.notes[-1])
        self.assertIn("DOWN (daily risk budget)", self.notes[-1])

    def test_not_before_935_and_not_after_deadline(self):
        bot.orb_place(make_ctx(now=AT_935 - timedelta(minutes=1)))
        self.client.submit_order.assert_not_called()
        self.assertIsNone(bot.state["orb_placed"])
        bot.orb_place(make_ctx(now=AT_935 + timedelta(minutes=10)))
        self.client.submit_order.assert_not_called()
        self.assertIn("wasn't running at 9:35", self.notes[-1])

    def test_already_placed_today_after_restart(self):
        bot.db.orb_plans.return_value = {"UP": ("o1", "long", 51.24, 51.04, 19)}
        bot.orb_place(make_ctx())
        self.client.submit_order.assert_not_called()


def position(qty, avg, price):
    return NS(symbol="DOWN", qty=str(qty), avg_entry_price=str(avg), current_price=str(price),
              asset_class=AssetClass.US_EQUITY)


class ProtectiveStop(BotTestCase):
    def setUp(self):
        super().setUp()
        bot.db.orb_plans.return_value = {"DOWN": ("o2", "short", 49.10, 49.30, 20)}
        self.client.get_orders.return_value = []

    def test_short_fill_gets_buy_stop_and_commits_risk(self):
        self.client.get_open_position.return_value = position(-20, 49.08, 49.00)
        self.assertEqual(bot.ensure_orb_stop("DOWN"), "placed stop 49.30")
        req = self.client.submit_order.call_args.args[0]
        self.assertEqual((req.side, req.qty, req.stop_price), (OrderSide.BUY, 20, 49.30))
        self.assertTrue(req.client_order_id.startswith("orb-s-DOWN-"))
        risk = next(c.kwargs for c in bot.db.log.call_args_list if c.args == ("risk",))
        self.assertEqual((risk["qty"], round(risk["risk_usd"], 2), risk["note"]), (20, 4.4, "orb"))

    def test_existing_stop_is_left_alone(self):
        bot.db.entry_risk.return_value = (20, 4.4)
        self.client.get_open_position.return_value = position(-20, 49.08, 49.00)
        self.client.get_orders.return_value = [NS(id="s1", symbol="DOWN", side=OrderSide.BUY, stop_price="49.30",
                                                  qty="20", type=OrderType.STOP)]
        self.assertEqual(bot.ensure_orb_stop("DOWN"), "stop in place")
        self.client.submit_order.assert_not_called()
        self.assertFalse(any(c.args == ("risk",) for c in bot.db.log.call_args_list))

    def test_partial_fill_grows_stop(self):
        bot.db.entry_risk.return_value = (5, 1.1)
        self.client.get_open_position.return_value = position(-20, 49.08, 49.00)
        old = NS(id="s1", symbol="DOWN", side=OrderSide.BUY, stop_price="49.30", qty="5", type=OrderType.STOP)
        self.client.get_orders.side_effect = [[old], [old], []]   # still open on the first check after cancel
        bot.ensure_orb_stop("DOWN")
        self.client.cancel_order_by_id.assert_called_once_with("s1")
        self.assertEqual(self.client.submit_order.call_args.args[0].qty, 20)
        risk = next(c.kwargs for c in bot.db.log.call_args_list if c.args == ("risk",))
        self.assertEqual(risk["qty"], 15)

    def test_no_replacement_while_old_stop_lingers(self):
        bot.db.entry_risk.return_value = (5, 1.1)
        self.client.get_open_position.return_value = position(-20, 49.08, 49.00)
        self.client.get_orders.return_value = [NS(id="s1", symbol="DOWN", side=OrderSide.BUY, stop_price="49.30",
                                                  qty="5", type=OrderType.STOP)]
        with self.assertRaisesRegex(RuntimeError, "old stop still open"):
            bot.ensure_orb_stop("DOWN")   # the next scan retries; ensure_orb_stop_safe alerts
        self.client.submit_order.assert_not_called()

    def test_already_past_stop_closes(self):
        self.client.get_open_position.return_value = position(-20, 49.08, 49.35)
        with mock.patch.object(bot, "flatten") as flat:
            self.assertEqual(bot.ensure_orb_stop("DOWN"), "already past the stop: closed")
        flat.assert_called_once_with("DOWN")

    def test_flat_and_unplanned(self):
        self.client.get_open_position.side_effect = APIError("nope", http_error=NS(response=NS(status_code=404)))
        self.assertEqual(bot.ensure_orb_stop("DOWN"), "flat")
        self.assertEqual(bot.ensure_orb_stop("AAPL"), "no ORB plan today")


class Housekeeping(BotTestCase):
    def test_fill_of_entry_triggers_stop_thread(self):
        order = NS(symbol="DOWN", side=OrderSide.SELL, id="o2", client_order_id="orb-e-20261106-DOWN")
        with mock.patch.object(bot.threading, "Thread") as thread:
            import asyncio
            asyncio.run(bot.on_trade_update(NS(event="fill", order=order, qty="20", price="49.08")))
        self.assertEqual(thread.call_args.kwargs["args"], ("DOWN",))
        fill = next(c.kwargs for c in bot.db.log.call_args_list if c.args == ("fills",))
        self.assertEqual(fill["client_id"], "orb-e-20261106-DOWN")

    def test_stale_timeout_skips_orb_and_eod_cancels_orb_shorts(self):
        old = AT_935 - timedelta(hours=2)
        entry = NS(id="e", symbol="UP", side=OrderSide.BUY, submitted_at=old, created_at=None,
                   client_order_id="orb-e-20261106-UP", asset_class=AssetClass.US_EQUITY)
        short = NS(id="s", symbol="DOWN", side=OrderSide.SELL, submitted_at=old, created_at=None,
                   client_order_id="orb-e-20261106-DOWN", asset_class=AssetClass.US_EQUITY)
        self.assertEqual(bot.cancel_stale_entries([entry, short], AT_935), [entry])
        self.client.cancel_order_by_id.assert_not_called()
        self.client.get_orders.return_value = [entry, short]
        self.client.get_all_positions.return_value = []
        bot.flatten_all_once(AT_935 + timedelta(hours=6))
        self.assertEqual({c.args[0] for c in self.client.cancel_order_by_id.call_args_list}, {"e", "s"})

    def test_failed_close_of_short_restores_buy_stop(self):
        stop = NS(id="s1", symbol="DOWN", side=OrderSide.BUY, type=OrderType.STOP, stop_price="49.30")
        self.client.get_orders.side_effect = [[stop], []]
        self.client.close_position.side_effect = RuntimeError("rejected")
        self.client.get_open_position.return_value = position(-20, 49.08, 49.0)
        with self.assertRaises(RuntimeError):
            bot.flatten("DOWN")
        req = self.client.submit_order.call_args.args[0]
        self.assertEqual((req.side, req.qty, req.stop_price), (OrderSide.BUY, 20, 49.30))

    def test_daily_loss_cancels_pending_entries(self):
        self.client.get_orders.return_value = [NS(id="e", symbol="UP", side=OrderSide.BUY,
                                                  client_order_id="orb-e-20261106-UP")]
        bot.db.orb_plans.return_value = {}
        bot.state.update(orb_placed=TODAY)
        bot.orb_cycle(make_ctx(loss_hit=True))
        self.client.cancel_order_by_id.assert_called_once_with("e")


class RoundTrips(unittest.TestCase):
    def test_short_then_long_in_the_same_symbol(self):
        import metrics
        fills = [("DOWN", "sell", 20, 49.08, "e2"), ("DOWN", "buy", 20, 49.32, "s2"),   # short: -$4.80
                 ("DOWN", "buy", 10, 50.0, "e3"), ("DOWN", "sell", 10, 51.0, "x3")]     # long: +$10
        self.assertEqual([round(p, 2) for p in metrics.round_trips(fills)], [-4.8, 10.0])

    def test_partial_cover_and_crypto_presale_ignored(self):
        import metrics
        fills = [("DOWN", "sell", 10, 50.0, "e"), ("DOWN", "buy", 4, 49.0, "c1"), ("DOWN", "buy", 6, 51.0, "c2"),
                 ("BTCUSD", "sell", 0.01, 70_000, "old"), ("BTCUSD", "buy", 0.01, 69_000, "b"),
                 ("BTCUSD", "sell", 0.01, 71_000, "x")]
        # DOWN: +4 then -6; BTC: the first sell predates logging, so the buy opens a long that closes +$20
        self.assertEqual([round(p, 2) for p in metrics.round_trips(fills)], [4.0, -6.0, 20.0])


class Report(unittest.TestCase):
    def test_slippage_and_r(self):
        plans = pd.DataFrame([
            ("2026-11-06T14:35:30+00:00", "UP", "e1", "entry: orb long", 51.24, 51.04, 19),
            ("2026-11-06T14:35:31+00:00", "DOWN", "e2", "entry: orb short", 49.10, 49.30, 20),
            ("2026-11-06T14:35:32+00:00", "MISS", "e3", "entry: orb long", 10.00, 9.90, 50),
        ], columns=["ts", "symbol", "order_id", "note", "trigger", "stop", "qty"])
        fills = pd.DataFrame([
            ("2026-11-06T14:40:00+00:00", "UP", "buy", 19, 51.26, "e1", "orb-e-20261106-UP"),
            ("2026-11-06T20:50:00+00:00", "UP", "sell", 19, 51.86, "x1", ""),
            ("2026-11-06T14:41:00+00:00", "DOWN", "sell", 20, 49.08, "e2", "orb-e-20261106-DOWN"),
            ("2026-11-06T14:52:00+00:00", "DOWN", "buy", 20, 49.32, "s2", "orb-s-DOWN-abc"),
        ], columns=["ts", "symbol", "side", "qty", "price", "order_id", "client_id"])
        t = orb_report.trades(plans, fills).set_index("symbol")
        self.assertAlmostEqual(t.loc["UP", "entry_slip_bps"], 0.02 / 51.24 * 1e4)
        self.assertAlmostEqual(t.loc["UP", "R"], (51.86 - 51.26) / 0.20)            # +3R at the close
        self.assertEqual(t.loc["UP", "exit"], "close")
        self.assertAlmostEqual(t.loc["DOWN", "entry_slip_bps"], 0.02 / 49.10 * 1e4)  # sold 2c below the trigger: worse
        self.assertAlmostEqual(t.loc["DOWN", "R"], -(49.32 - 49.08) / 0.20)          # -1.2R: stop slipped
        self.assertAlmostEqual(t.loc["DOWN", "stop_slip_bps"], 0.02 / 49.30 * 1e4)
        self.assertEqual(t.loc["MISS", "filled"], 0)


if __name__ == "__main__":
    unittest.main()
