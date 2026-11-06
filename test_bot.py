"""Order-handling tests with Alpaca mocked out. Nothing here touches the network or trades.db."""
import os
import unittest
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace as NS
from unittest import mock

import pandas as pd

os.environ.setdefault("ALPACA_API_KEY", "test")
os.environ.setdefault("ALPACA_SECRET_KEY", "test")
import bot  # noqa: E402
from alpaca.trading.enums import AssetClass, OrderSide, OrderStatus, OrderType  # noqa: E402
from alpaca.trading.requests import StopLimitOrderRequest  # noqa: E402
from alpaca.common.exceptions import APIError  # noqa: E402
from backtest import equity_curve  # noqa: E402


class BotTestCase(unittest.TestCase):
    def setUp(self):
        self.client = mock.Mock()
        self.notes = []
        for target, new in (("trading_client", self.client), ("db", mock.Mock()),
                            ("notify", self.notes.append), ("time", mock.Mock()),
                            ("earnings_calendar", mock.Mock(return_value={}))):
            patcher = mock.patch.object(bot, target, new)
            patcher.start()
            self.addCleanup(patcher.stop)
        bot.state.update(flatten_day=None, flatten_time=None, summary_day=None, news_down=False,
                         scan_day=None, scanned=[], symbol_errors={})
        bot.db.last_activity.return_value = None
        bot.db.entry_risk.return_value = None
        bot.db.risk_today.return_value = (0.0, 0)


def make_ctx(now, positions=None, pending=(), minutes_to_close=300, **kw):
    ctx = {"now": now, "equity": 100_000.0, "last_equity": 100_000.0, "loss_hit": False,
           "minutes_to_close": minutes_to_close, "next_session": None, "risk_today": 0.0, "entries_today": 0,
           "positions": positions or {}, "pending_symbols": set(pending), "exposure": 0.0}
    ctx.update(kw)
    return ctx


ET = timezone(timedelta(hours=-5))
CLOSE_OUT = datetime(2026, 11, 6, 15, 50, tzinfo=ET)


AAPL_POS = NS(symbol="AAPL", asset_class=AssetClass.US_EQUITY, qty="5", avg_entry_price="100")
BTC_POS = NS(symbol="BTCUSD", asset_class=AssetClass.CRYPTO, qty="0.01", qty_available="0.01",
             avg_entry_price="70000", unrealized_pl="12.5")


class EndOfDay(BotTestCase):
    def test_failed_flatten_is_retried_and_btc_is_kept(self):
        self.client.get_orders.return_value = []
        self.client.get_all_positions.return_value = [AAPL_POS, BTC_POS]
        with mock.patch.object(bot, "flatten", side_effect=RuntimeError("rejected")) as flat:
            with self.assertRaises(RuntimeError):
                bot.flatten_all_once(CLOSE_OUT)
        self.assertIsNone(bot.state["flatten_day"])
        flat.assert_called_once_with("AAPL")  # BTC never touched

        with mock.patch.object(bot, "flatten") as flat:
            bot.flatten_all_once(CLOSE_OUT)
            bot.flatten_all_once(CLOSE_OUT)  # second call is a no-op
        flat.assert_called_once_with("AAPL")
        self.assertEqual(bot.state["flatten_day"], date(2026, 11, 6))


class DailySummary(BotTestCase):
    def test_waits_for_close_out_fills_then_sends_once(self):
        bot.maybe_send_daily_summary(CLOSE_OUT)  # no close-out yet
        self.assertEqual(self.notes, [])

        bot.state.update(flatten_day=CLOSE_OUT.date(), flatten_time=CLOSE_OUT)
        bot.maybe_send_daily_summary(CLOSE_OUT + timedelta(minutes=1))  # too soon
        self.assertEqual(self.notes, [])

        self.client.get_account.return_value = NS(equity="100150", last_equity="100000")
        self.client.get_all_positions.return_value = [BTC_POS]
        bot.db.fills.return_value = [("AAPL", "buy", 10, 100, "a"), ("AAPL", "sell", 10, 115, "b"),
                                     ("MSFT", "buy", 5, 200, "c"), ("MSFT", "sell", 5, 190, "d")]
        bot.db.count_since.side_effect = lambda table, *_: {"orders": 2, "decisions": 1}[table]
        bot.maybe_send_daily_summary(CLOSE_OUT + timedelta(minutes=2))
        bot.maybe_send_daily_summary(CLOSE_OUT + timedelta(minutes=3))  # already sent

        self.assertEqual(len(self.notes), 1)
        msg = self.notes[0]
        for expected in ("Day P&L: $+150.00 (+0.15%)", "Entries: 2 | Closed trades: 2 | Wins: 1 (50%)",
                         "Best: $+150.00 | Worst: $-50.00", "Qwen vetoes: 1",
                         "Holding 0.01 BTCUSD overnight (P&L $+12.50)"):
            self.assertIn(expected, msg)
        since = bot.db.fills.call_args.args[0]
        self.assertTrue(since.startswith("2026-11-06T05:00:00"))  # ET midnight in UTC


class OllamaAlert(BotTestCase):
    def test_alerts_once_when_down_and_once_on_recovery(self):
        with mock.patch.object(bot.requests, "get", side_effect=ConnectionError("refused")):
            bot.check_ollama()
            bot.check_ollama()
        self.assertEqual(len(self.notes), 1)
        self.assertIn("Is Ollama running?", self.notes[0])

        tags = mock.Mock(json=lambda: {"models": [{"name": bot.config.OLLAMA_MODEL}]})
        with mock.patch.object(bot.requests, "get", return_value=tags):
            bot.check_ollama()
            bot.check_ollama()
        self.assertEqual(len(self.notes), 2)
        self.assertIn("working again", self.notes[1])

    def test_missing_model_counts_as_down(self):
        tags = mock.Mock(json=lambda: {"models": [{"name": "llama3:8b"}]})
        with mock.patch.object(bot.requests, "get", return_value=tags):
            bot.check_ollama()
        self.assertIn("not pulled", self.notes[0])


class Flatten(BotTestCase):
    def setUp(self):
        super().setUp()
        stop_leg = NS(id="leg1", symbol="AAPL", side=OrderSide.SELL, type=OrderType.STOP, stop_price="98.5")
        self.client.get_orders.side_effect = [[stop_leg], []]  # before cancel, after cancel
        self.client.close_position.side_effect = RuntimeError("rejected")
        self.client.get_open_position.return_value = NS(qty="7")

    def test_failed_close_restores_stop(self):
        with self.assertRaises(RuntimeError):
            bot.flatten("AAPL")
        order = self.client.submit_order.call_args.args[0]
        self.assertEqual((order.symbol, order.qty, order.stop_price), ("AAPL", 7, 98.5))
        self.assertIn("Re-placed stop at $98.50", self.notes[-1])

    def test_failed_restore_says_unprotected(self):
        self.client.submit_order.side_effect = RuntimeError("also rejected")
        with self.assertRaises(RuntimeError):
            bot.flatten("AAPL")
        self.assertIn("UNPROTECTED", self.notes[-1])

    def test_already_closed_is_not_an_alarm(self):
        not_found = APIError("position not found", http_error=NS(response=NS(status_code=404)))
        self.client.get_open_position.side_effect = not_found
        bot.flatten("AAPL")  # no exception
        self.client.submit_order.assert_not_called()
        self.assertFalse(any("UNPROTECTED" in n for n in self.notes))

    def test_network_error_on_check_still_alerts(self):
        self.client.get_open_position.side_effect = ConnectionError("timeout")
        with self.assertRaises(RuntimeError):
            bot.flatten("AAPL")
        self.assertIn("UNPROTECTED", self.notes[-1])


class SymbolErrors(BotTestCase):
    def test_same_error_alerts_once_until_cleared(self):
        bot.state["symbol_errors"] = {}
        for _ in range(3):
            bot.report_symbol_error("BTC/USD", RuntimeError("stop rejected"))
        bot.report_symbol_error("BTC/USD", RuntimeError("different"))
        self.assertEqual(len(self.notes), 2)
        bot.state["symbol_errors"].pop("BTC/USD")  # what a clean scan does
        bot.report_symbol_error("BTC/USD", RuntimeError("different"))
        self.assertEqual(len(self.notes), 3)


NOW = datetime(2026, 11, 7, 3, 0, 10, tzinfo=timezone.utc)  # a Saturday night: stocks closed


def btc_bars(close=70_000.0, atr=100.0):
    idx = pd.date_range(end=NOW - timedelta(minutes=5, seconds=10), periods=21, freq="5min")
    return pd.DataFrame({"close": close, "sma_5": 70_100.0, "sma_20": 70_000.0, "rsi_14": 50.0,
                         "rel_volume": 1.0, "atr_14": atr, "vwap": 69_900.0}, index=idx)


class Crypto(BotTestCase):
    def ctx(self, positions):
        return make_ctx(NOW, positions, minutes_to_close=0)

    def test_symbol_formats_match(self):
        self.assertTrue(bot.is_crypto("BTC/USD") and bot.is_crypto("BTCUSD"))
        self.assertFalse(bot.is_crypto("AAPL"))
        self.assertEqual(bot.sector("BTCUSD"), "crypto")

    def test_missing_stop_is_placed_as_stop_limit(self):
        self.client.get_orders.return_value = []
        row = btc_bars().iloc[-1]
        bot.ensure_crypto_stop("BTC/USD", BTC_POS, row)
        order = self.client.submit_order.call_args.args[0]
        self.assertIsInstance(order, StopLimitOrderRequest)
        # stop = 70000 - 1.5 * 100; limit 1% below
        self.assertEqual((order.symbol, order.qty, order.stop_price, order.limit_price),
                         ("BTC/USD", 0.01, 69_850.0, 69_151.5))

    def test_existing_stop_is_left_alone(self):
        self.client.get_orders.return_value = [NS(symbol="BTC/USD", side=OrderSide.SELL, stop_price="69850")]
        self.assertEqual(bot.ensure_crypto_stop("BTC/USD", BTC_POS, btc_bars().iloc[-1]), "stop in place")
        self.client.submit_order.assert_not_called()

    def test_triggered_but_unfilled_stop_exits_at_market(self):
        self.client.get_orders.return_value = [NS(symbol="BTC/USD", side=OrderSide.SELL, stop_price="69850")]
        with mock.patch.object(bot, "flatten") as flat:
            result = bot.ensure_crypto_stop("BTC/USD", BTC_POS, btc_bars(close=69_500).iloc[-1])
        flat.assert_called_once_with("BTC/USD")
        self.assertIn("exited at market", result)

    def test_slow_ioc_is_cancelled_before_trusting_fill_qty(self):
        still_open = NS(status=OrderStatus.NEW, filled_qty="0")
        after_cancel = NS(status=OrderStatus.CANCELED, filled_qty="0.004")  # part-filled at the last moment
        self.client.get_order_by_id.side_effect = [still_open] * 21 + [after_cancel]
        self.assertEqual(bot.wait_for_order("e1"), 0.004)
        self.client.cancel_order_by_id.assert_called_once_with("e1")

    def test_take_profit_in_code(self):
        with mock.patch.object(bot, "flatten") as flat:  # target = 70000 + 3 * 100 = 70300
            bot.scan_symbol("BTC/USD", btc_bars(close=70_400), None, self.ctx({"BTCUSD": BTC_POS}))
        flat.assert_called_once_with("BTC/USD")

    def test_failed_close_restores_stop_limit(self):
        stop = NS(id="s", symbol="BTC/USD", side=OrderSide.SELL, type=OrderType.STOP_LIMIT, stop_price="69850")
        self.client.get_orders.side_effect = [[stop], []]
        self.client.close_position.side_effect = RuntimeError("rejected")
        self.client.get_open_position.return_value = BTC_POS
        with self.assertRaises(RuntimeError):
            bot.flatten("BTC/USD")
        self.client.close_position.assert_called_once_with("BTCUSD")  # no slash in the URL path
        self.assertIsInstance(self.client.submit_order.call_args.args[0], StopLimitOrderRequest)

    def test_entry_is_ioc_then_stop(self):
        self.client.get_orders.return_value = []
        self.client.submit_order.return_value = NS(id="e1")
        self.client.get_order_by_id.return_value = NS(status=OrderStatus.FILLED, filled_qty="0.01")
        self.client.get_open_position.return_value = BTC_POS
        ctx = self.ctx({})
        result = bot.enter_crypto("BTC/USD", 0.01, 70_000.0, btc_bars().iloc[-1], ctx)
        entry, stop = (c.args[0] for c in self.client.submit_order.call_args_list)
        self.assertEqual(entry.time_in_force, bot.TimeInForce.IOC)
        self.assertIsInstance(stop, StopLimitOrderRequest)
        self.assertIn("placed stop", result)
        self.assertIn("BTCUSD", ctx["positions"])

    def test_unfilled_ioc_places_nothing_else(self):
        self.client.submit_order.return_value = NS(id="e1")
        self.client.get_order_by_id.return_value = NS(status=OrderStatus.CANCELED, filled_qty="0")
        result = bot.enter_crypto("BTC/USD", 0.01, 70_000.0, btc_bars().iloc[-1], self.ctx({}))
        self.assertEqual(result, "skip: IOC entry not filled")
        self.assertEqual(self.client.submit_order.call_count, 1)


class OppositeIndexFunds(BotTestCase):
    def ctx(self, held=(), pending=()):
        return make_ctx(NOW, {s: NS(symbol=s) for s in held}, pending)

    def test_long_and_inverse_never_held_together(self):
        row = NS(atr_14=1.0)
        for symbol, ctx in (("PSQ", self.ctx(held=["QQQ"])), ("SH", self.ctx(pending=["SPY"])),
                            ("QQQ", self.ctx(held=["SH"]))):
            self.assertEqual(bot.try_enter(symbol, row, ctx), "skip: opposite index fund held (would cancel out)")
        self.client.submit_order.assert_not_called()

    def test_same_direction_and_stocks_are_not_blocked(self):
        # reaching the quote fetch means the direction gate let it through
        with mock.patch.object(bot, "latest_quote", side_effect=RuntimeError("reached quote")):
            for symbol, held in (("PSQ", ["SH"]), ("AAPL", ["PSQ"]), ("QQQ", ["SPY"])):
                with self.assertRaisesRegex(RuntimeError, "reached quote"):
                    bot.try_enter(symbol, NS(atr_14=1.0), self.ctx(held=held))


class StaleEntries(BotTestCase):
    def test_missing_timestamps_do_not_crash(self):
        now = datetime(2026, 11, 6, 15, tzinfo=timezone.utc)
        old = NS(id="a", symbol="AAPL", side=OrderSide.BUY, submitted_at=now - timedelta(minutes=30), created_at=None)
        unknown = NS(id="b", symbol="MSFT", side=OrderSide.BUY, submitted_at=None, created_at=None)
        self.assertEqual(bot.cancel_stale_entries([old, unknown], now), [unknown])
        self.client.cancel_order_by_id.assert_called_once_with("a")


MARKET = datetime(2026, 11, 6, 15, 0, 10, tzinfo=timezone.utc)  # Friday 10:00 ET
QUOTE = NS(bid_price="99.95", ask_price="100.00")
ROW = NS(atr_14=1.0)  # plan from the $100 ask: stop 98.50, target 103.00, 20 shares ($2000 cap), $30 at risk


class EntryGates(BotTestCase):
    def setUp(self):
        super().setUp()
        for target, new in (("latest_quote", mock.Mock(return_value=QUOTE)),
                            ("time_machine", mock.Mock(return_value=None))):
            patcher = mock.patch.object(bot, target, new)
            patcher.start()
            self.addCleanup(patcher.stop)
        for patcher in (mock.patch.dict(os.environ, {"ALPHAVANTAGE_API_KEY": "k"}),
                        mock.patch.object(bot.config, "USE_LLM_VETO", False)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.client.submit_order.return_value = NS(id="o1")

    def enter(self, now=MARKET, **ctx):
        return bot.try_enter("AAPL", ROW, make_ctx(now, **ctx))

    def test_enters_and_commits_risk(self):
        ctx = make_ctx(MARKET)
        self.assertTrue(bot.try_enter("AAPL", ROW, ctx).startswith("entered"))
        risk_row = next(c.kwargs for c in bot.db.log.call_args_list if c.args == ("risk",))
        self.assertEqual((risk_row["qty"], risk_row["risk_usd"]), (20, 30.0))
        self.assertEqual((ctx["risk_today"], ctx["entries_today"]), (30.0, 1))

    def test_first_minutes_after_open(self):
        self.assertEqual(self.enter(MARKET - timedelta(minutes=20)), "skip: first 15 min after the open")

    def test_entries_per_day(self):
        self.assertIn("entries already today", self.enter(entries_today=bot.config.MAX_ENTRIES_PER_DAY))

    def test_cooldown_from_db(self):
        bot.db.last_activity.return_value = (MARKET - timedelta(minutes=10)).isoformat()
        self.assertIn("cooldown", self.enter())
        bot.db.last_activity.return_value = (MARKET - timedelta(minutes=31)).isoformat()
        self.assertTrue(self.enter().startswith("entered"))

    def test_earnings_today_blocks(self):
        bot.earnings_calendar.return_value = {date(2026, 11, 6): {"AAPL"}}
        self.assertEqual(self.enter(), "skip: earnings (reports today)")

    def test_earnings_next_day_only_when_enabled(self):
        bot.earnings_calendar.return_value = {date(2026, 11, 9): {"AAPL"}}
        self.assertTrue(self.enter(next_session=date(2026, 11, 9)).startswith("entered"))
        with mock.patch.object(bot.config, "EARNINGS_BLOCK_NEXT_DAY", True):
            self.assertEqual(self.enter(next_session=date(2026, 11, 9)), "skip: earnings (reports Mon 11-09)")

    def test_earnings_failure_fails_closed_and_alerts_once(self):
        bot.earnings_calendar.side_effect = ValueError("rate limited")
        self.assertIn("earnings check failed", self.enter())
        self.assertIn("earnings check failed", self.enter())
        self.assertEqual(sum("Earnings calendar failing" in n for n in self.notes), 1)
        self.client.submit_order.assert_not_called()
        bot.earnings_calendar.side_effect = None
        self.assertTrue(self.enter().startswith("entered"))
        self.assertTrue(any("Earnings calendar is working again" in n for n in self.notes))

    def test_no_key_means_no_earnings_check(self):
        bot.earnings_calendar.return_value = {date(2026, 11, 6): {"AAPL"}}
        with mock.patch.dict(os.environ, {"ALPHAVANTAGE_API_KEY": ""}):
            self.assertTrue(self.enter().startswith("entered"))

    def test_min_reward_risk(self):
        with mock.patch.object(bot.config, "MIN_REWARD_RISK", 2.5):
            self.assertEqual(self.enter(), "skip: reward:risk 2.00 < 2.5")

    def test_daily_risk_budget(self):
        # budget 1% of $100k = $1000; $975 already committed + $30 is over
        self.assertIn("daily risk budget", self.enter(risk_today=975.0))
        self.assertTrue(self.enter(risk_today=970.0).startswith("entered"))

    def test_time_machine_logs_and_filters(self):
        tm = {"mc": {"n": 10000, "target": .2, "stop": .6, "open": .2, "avg_r": -0.3},
              "history": {"n": 60, "target": .2, "stop": .5, "open": .3, "avg_r": -0.1}}
        bot.time_machine.return_value = tm
        self.assertIn("| TM 60d: replays target 20%", self.enter())  # log only by default
        with mock.patch.object(bot.config, "TM_MIN_EXPECTED_R", 0.0):
            self.assertIn("skip: time machine avg -0.30R", self.enter())
        bot.time_machine.assert_called_with("AAPL", 100.0, 98.5, 103.0, MARKET)


class RiskRefund(BotTestCase):
    def test_cancelled_entry_gives_back_the_unfilled_part(self):
        bot.db.entry_risk.return_value = (20, 30.0)
        old = NS(id="a", symbol="AAPL", side=OrderSide.BUY, submitted_at=MARKET - timedelta(minutes=30),
                 created_at=None, filled_qty="5")
        bot.cancel_stale_entries([old], MARKET)
        refund = next(c.kwargs for c in bot.db.log.call_args_list if c.args == ("risk",))
        self.assertEqual((refund["qty"], refund["risk_usd"]), (-15, -22.5))

    def test_day_start_is_new_york_midnight(self):
        self.assertEqual(bot.et_day_start(MARKET), "2026-11-06T05:00:00+00:00")  # EST since Nov 1, 2026


class Earnings(unittest.TestCase):
    def test_parse_csv(self):
        text = ("symbol,name,reportDate,fiscalDateEnding,estimate,currency,timeOfTheDay\r\n"
                "AAPL,Apple Inc,2026-11-06,2026-09-30,1.8,USD,post-market\r\n"
                "msft,Microsoft,2026-11-09,2026-09-30,3.1,USD,pre-market\r\n")
        self.assertEqual(bot.parse_earnings(text), {date(2026, 11, 6): {"AAPL"}, date(2026, 11, 9): {"MSFT"}})

    def test_error_json_is_not_an_empty_calendar(self):
        with self.assertRaises(ValueError):
            bot.parse_earnings('{"Information": "rate limit reached"}')


class Scanner(BotTestCase):
    def test_scans_once_per_day_from_the_entry_window(self):
        bot.state["scanned"] = ["OLD"]
        with mock.patch.object(bot, "scan_watchlist", return_value=["MU", "AMD"]) as scan:
            bot.maybe_scan(MARKET - timedelta(minutes=20))   # 9:40: too early, yesterday's list dropped
            self.assertEqual(bot.state["scanned"], [])
            bot.maybe_scan(MARKET)
            bot.maybe_scan(MARKET + timedelta(minutes=5))
        scan.assert_called_once()
        self.assertEqual(bot.state["scanned"], ["MU", "AMD"])
        self.assertIn("[SCAN] Added today: MU, AMD", self.notes)

    def test_failure_retries_next_scan(self):
        with mock.patch.object(bot, "scan_watchlist", side_effect=RuntimeError("503")):
            bot.maybe_scan(MARKET)
        self.assertIsNone(bot.state["scan_day"])
        with mock.patch.object(bot, "scan_watchlist", return_value=["MU"]):
            bot.maybe_scan(MARKET + timedelta(minutes=5))
        self.assertEqual(bot.state["scanned"], ["MU"])

    def test_filters(self):
        actives = [NS(symbol=s, volume=v) for s, v in
                   (("SOXL", 9e7), ("PENNY", 8e7), ("BRK.B", 5e7), ("AAPL", 4e7), ("MU", 3e7), ("AMD", 2e7))]
        prices = {"SOXL": 40, "PENNY": 2, "AAPL": 270, "MU": 120, "AMD": 250}
        names = {"SOXL": "Direxion Daily Semiconductor Bull 3X Shares", "MU": "Micron", "AMD": "AMD"}
        with mock.patch.object(bot, "screener_client") as sc, mock.patch.object(bot, "data_client") as dc:
            sc.get_most_actives.return_value = NS(most_actives=actives)
            dc.get_stock_snapshot.return_value = {s: NS(latest_trade=NS(price=p)) for s, p in prices.items()}
            self.client.get_asset.side_effect = lambda s: NS(
                tradable=True, asset_class=AssetClass.US_EQUITY, exchange="NASDAQ", name=names[s])
            # AAPL is already watched; BRK.B isn't a plain ticker; PENNY is under $10; SOXL is leveraged
            self.assertEqual(bot.scan_watchlist(), ["AMD", "MU"])  # by dollar volume: $5.0B > $3.6B

    def test_scanned_symbols_get_their_own_sector(self):
        self.assertEqual(bot.sector("MU"), "other:MU")
        self.assertEqual(bot.market_side("MU"), 0)

class EquityCurve(unittest.TestCase):
    def test_flat_days_are_kept_and_no_exit_is_dropped(self):
        idx = pd.date_range("2026-11-02 15:00", periods=3, freq="1D", tz="UTC")
        off_index_exit = idx[2] + pd.Timedelta(minutes=5)
        eq = equity_curve([(idx[0], 100.0), (off_index_exit, -50.0)], idx, start_equity=1000)
        self.assertEqual(list(eq), [1100, 1100, 1100, 1050])


if __name__ == "__main__":
    unittest.main()
