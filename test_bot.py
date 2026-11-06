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
                            ("notify", self.notes.append), ("time", mock.Mock())):
            patcher = mock.patch.object(bot, target, new)
            patcher.start()
            self.addCleanup(patcher.stop)
        bot.state.update(flatten_day=None, flatten_time=None, summary_day=None, news_down=False)


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
        return {"now": NOW, "equity": 100_000.0, "loss_hit": False, "minutes_to_close": 0,
                "positions": positions, "pending_symbols": set(), "exposure": 0.0}

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


class StaleEntries(BotTestCase):
    def test_missing_timestamps_do_not_crash(self):
        now = datetime(2026, 11, 6, 15, tzinfo=timezone.utc)
        old = NS(id="a", symbol="AAPL", side=OrderSide.BUY, submitted_at=now - timedelta(minutes=30), created_at=None)
        unknown = NS(id="b", symbol="MSFT", side=OrderSide.BUY, submitted_at=None, created_at=None)
        self.assertEqual(bot.cancel_stale_entries([old, unknown], now), [unknown])
        self.client.cancel_order_by_id.assert_called_once_with("a")


class EquityCurve(unittest.TestCase):
    def test_flat_days_are_kept_and_no_exit_is_dropped(self):
        idx = pd.date_range("2026-11-02 15:00", periods=3, freq="1D", tz="UTC")
        off_index_exit = idx[2] + pd.Timedelta(minutes=5)
        eq = equity_curve([(idx[0], 100.0), (off_index_exit, -50.0)], idx, start_equity=1000)
        self.assertEqual(list(eq), [1100, 1100, 1100, 1050])


if __name__ == "__main__":
    unittest.main()
