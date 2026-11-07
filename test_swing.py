"""Swing tests: rules (swing.py), the backtest replay, and order handling in bot.py (Alpaca mocked). Offline."""
import unittest
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace as NS
from unittest import mock

import pandas as pd

import config
import strategy
import swing
from backtest import simulate_swing
from test_bot import BotTestCase, bot, make_ctx  # sets dummy keys before importing bot
from alpaca.trading.enums import AssetClass, OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

SYMS = ["SPY", "QQQ", "IWM", "DIA"]
_pinned = []


def setUpModule():
    """These tests use a 4-ETF dip list with 7.5% slots and no intraday strategy: pin that, so they test the logic
    whatever the defaults in config.py are."""
    p = mock.patch.multiple(config, SWING_TREND_WEIGHT=0.70, SWING_DIP_WEIGHT=0.30, SWING_DIP_SYMBOLS=tuple(SYMS),
                            STRATEGY="none")
    p.start()
    _pinned.append(p)


def tearDownModule():
    for p in _pinned:
        p.stop()


def short_averages():
    """A 20-day trend average instead of 200, so test series stay short."""
    return mock.patch.object(config, "SWING_TREND_SMA", 20)


def rise_then_dip(end="2026-11-05"):
    """+1 a day for 21 closes, then three -1.5 closes (RSI(2) 40, 18, 8.7), then a close above the 5-day average.
    Index 23 (115.5): RSI(2) < 10 and above the 20-day average (112.75) -> dip entry. Index 24 (119 > 118): exit."""
    c = [100.0 + i for i in range(21)] + [118.5, 117.0, 115.5, 119.0, 120.0]
    return pd.Series(c, index=pd.bdate_range(end=end, periods=len(c)))


class Rules(unittest.TestCase):
    def test_rsi2_is_wilders(self):
        self.assertAlmostEqual(swing.rsi(pd.Series([10.0, 11.0, 10.0])).iloc[-1], 50.0)
        self.assertEqual(swing.rsi(pd.Series([1.0, 2.0, 3.0, 4.0])).iloc[-1], 100.0)
        r = swing.rsi(rise_then_dip())
        self.assertAlmostEqual(r.iloc[21], 40.0)
        self.assertAlmostEqual(r.iloc[23], 100 - 100 / (1 + 0.125 / 1.3125))   # 8.7

    def test_dip_entry_and_exit(self):
        with short_averages():
            held = swing.dip_on(rise_then_dip())
            trend = swing.trend_on(rise_then_dip())
        self.assertEqual(list(held[held].index.strftime("%m-%d")), [held.index[23].strftime("%m-%d")])
        self.assertTrue(trend.iloc[23])
        self.assertFalse(trend.iloc[18])            # no 20-day average yet

    def test_no_dip_below_the_trend_average(self):
        c = rise_then_dip()
        with mock.patch.object(config, "SWING_TREND_SMA", 10):   # 10-day average 117.0 > 115.5: no entry
            self.assertFalse(swing.dip_on(c).any())

    def test_weights_add_up_per_symbol(self):
        closes = pd.DataFrame({s: rise_then_dip() for s in SYMS})
        with short_averages():
            trend, dips = swing.rule_states(closes)
        w = trend + dips * swing.dip_slot()
        self.assertAlmostEqual(w["SPY"].iloc[23], 0.70 + 0.075)   # trend + its own dip slot
        self.assertAlmostEqual(w["QQQ"].iloc[23], 0.075)
        self.assertEqual(w["QQQ"].iloc[24], 0.0)
        self.assertAlmostEqual(w["SPY"].iloc[24], 0.70)
        self.assertEqual(list(swing.rule_states(closes[["QQQ"]])[0].columns), ["QQQ"])   # missing symbols left out


class Targets(unittest.TestCase):
    """targets(trend_now, trend_before, dip_now, dip_before, equity, held_value). Dip slot 7.5%, trend 70%."""

    def test_dip_bought_on_its_entry_day_only(self):
        self.assertEqual(swing.targets({}, {}, {"QQQ": True}, {"QQQ": False}, 10_000, {}), {"QQQ": 0.075})
        # running since before, not held (bot was off or just started): not joined at today's price
        self.assertEqual(swing.targets({}, {}, {"QQQ": True}, {"QQQ": True}, 10_000, {}), {"QQQ": 0.0})
        # running and held (drifted a bit): kept until the exit
        self.assertEqual(swing.targets({}, {}, {"QQQ": True}, {"QQQ": True}, 10_000, {"QQQ": 700.0}), {"QQQ": 0.075})
        self.assertEqual(swing.targets({}, {}, {"QQQ": False}, {"QQQ": True}, 10_000, {"QQQ": 750.0}), {"QQQ": 0.0})

    def test_spy_dip_on_top_of_the_trend(self):
        held = {"SPY": 7_750.0}   # 70% trend + 7.5% dip
        self.assertAlmostEqual(swing.targets({"SPY": 0.7}, {"SPY": 0.7}, {"SPY": True}, {"SPY": True}, 10_000,
                                             held)["SPY"], 0.775)
        # trend exit today: the dip part stays
        self.assertAlmostEqual(swing.targets({"SPY": 0.0}, {"SPY": 0.7}, {"SPY": True}, {"SPY": True}, 10_000,
                                             held)["SPY"], 0.075)
        # trend back on while the dip (750 held) runs: measured against yesterday's trend weight, so it's kept
        self.assertAlmostEqual(swing.targets({"SPY": 0.7}, {"SPY": 0.0}, {"SPY": True}, {"SPY": True}, 10_000,
                                             {"SPY": 750.0})["SPY"], 0.775)
        # holding only the trend part: a running dip isn't joined
        self.assertAlmostEqual(swing.targets({"SPY": 0.7}, {"SPY": 0.7}, {"SPY": True}, {"SPY": True}, 10_000,
                                             {"SPY": 7_000.0})["SPY"], 0.7)


class Orders(unittest.TestCase):
    def test_buy_from_flat_and_ignore_drift(self):
        self.assertEqual(swing.orders({"SPY": 0.7}, 10_000, {"SPY": 500.0}, {}, {"SPY": True}), [("SPY", 14.0)])
        # equity up 3%: the target moves $210, less than half the smallest slot (3.75% = $386)
        self.assertEqual(swing.orders({"SPY": 0.7}, 10_300, {"SPY": 500.0}, {"SPY": 14.0}, {"SPY": True}), [])

    def test_exit_closes_everything_and_sells_come_first(self):
        plan = swing.orders({"SPY": 0.0, "QQQ": 0.075, "IWM": 0.0}, 10_000, {"SPY": 500.0, "QQQ": 400.0, "IWM": 1.0},
                            {"SPY": 3.33333, "QQQ": 0.0}, {"SPY": True, "QQQ": True, "IWM": True})
        self.assertEqual(plan, [("SPY", -3.33333), ("QQQ", 1.875)])    # IWM: nothing held, nothing to do

    def test_rounding_toward_zero(self):
        self.assertEqual(swing.orders({"QQQ": 0.1}, 1234.56, {"QQQ": 100.0}, {}, {"QQQ": True}), [("QQQ", 1.2345)])
        self.assertEqual(swing.orders({"QQQ": 0.1}, 2_900.0, {"QQQ": 100.0}, {}, {"QQQ": False}), [("QQQ", 2)])
        self.assertEqual(swing.orders({"QQQ": 0.5}, 10_000, {"QQQ": 100.0}, {"QQQ": 80.0}, {"QQQ": False}),
                         [("QQQ", -30)])

    def test_cost_in_r(self):
        # BTC at the bot's old settings: 0.25% x 2 of $70,000 = $350 against a $150 stop -> 2.33R
        self.assertAlmostEqual(strategy.cost_in_r(70_000, 69_850, 25), 350 / 150)
        self.assertEqual(strategy.cost_in_r(100, 100, 2), float("inf"))


class Backtest(unittest.TestCase):
    def test_trend_then_dip_round_trip(self):
        closes = pd.DataFrame({s: rise_then_dip() for s in SYMS})
        with short_averages():
            curve, trades = simulate_swing(closes, cost=0.0, equity=100_000)
        days = closes.index
        by = {(t, s): q for t, s, q, _ in trades}
        spy0 = by[(days[19], "SPY")]
        self.assertAlmostEqual(spy0, 0.7 * 100_000 / 119, places=3)           # trend in at the first average
        qqq = by[(days[23], "QQQ")]
        self.assertAlmostEqual(qqq * 115.5, 0.075 * curve["equity"].iloc[23], delta=1)
        self.assertEqual(by[(days[24], "QQQ")], -qqq)                          # out at the next close above SMA5
        self.assertEqual(curve["equity"].iloc[18], 100_000)                    # flat before the average exists
        self.assertGreater(curve["exposure"].iloc[23], curve["exposure"].iloc[22])
        with short_averages():
            costly, _ = simulate_swing(closes, cost=0.001, equity=100_000)
        self.assertLess(costly["equity"].iloc[-1], curve["equity"].iloc[-1])


NOW = datetime(2026, 11, 6, 20, 45, 10, tzinfo=timezone.utc)   # Fri 15:45:10 ET
TODAY, PREV = date(2026, 11, 6), date(2026, 11, 5)


def history(kind="up", n=60):
    if kind == "up":
        c = [100.0 + i for i in range(n)]
    else:
        c = [200.0 - i for i in range(n)]
    return pd.Series(c, index=pd.bdate_range(end=PREV, periods=n))


class SwingCycle(BotTestCase):
    def setUp(self):
        super().setUp()
        bot.state.update(swing_day=None, swing_tried=None)
        bot._fractionable.clear()
        self.closes = pd.DataFrame({s: history("up") for s in SYMS})
        self.prices = {s: 160.0 for s in SYMS}
        self.ages = {s: timedelta(seconds=30) for s in SYMS}
        self.snaps = mock.Mock(**{"get_stock_snapshot.side_effect": lambda req: {
            s: NS(latest_trade=NS(price=self.prices[s], timestamp=NOW - self.ages[s])) for s in SYMS}})
        for target, new in (("swing_closes", mock.Mock(side_effect=lambda syms, today: self.closes)),
                            ("data_client", self.snaps)):
            patcher = mock.patch.object(bot, target, new)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = short_averages()
        patcher.start()
        self.addCleanup(patcher.stop)
        c = self.client
        c.get_calendar.return_value = [NS(date=date(2026, 11, 4)), NS(date=PREV), NS(date=TODAY)]
        c.get_orders.return_value = []
        c.get_all_positions.return_value = []
        c.get_asset.return_value = NS(fractionable=True)
        c.submit_order.side_effect = lambda req: NS(id=f"id-{req.symbol}")
        c.close_position.return_value = NS(id="close-1")

    def run_cycle(self, minutes_to_close=14.8):
        bot.swing_cycle(make_ctx(NOW, minutes_to_close=minutes_to_close))

    def test_trades_only_in_the_window_and_once_a_day(self):
        self.run_cycle(minutes_to_close=20)
        self.run_cycle(minutes_to_close=10)        # the close-out's turn
        self.client.submit_order.assert_not_called()
        self.run_cycle()
        self.run_cycle(minutes_to_close=12)
        self.assertEqual(self.client.submit_order.call_count, 1)
        self.assertEqual(bot.state["swing_day"], TODAY)

    def test_trend_buy_is_a_fractional_market_order(self):
        self.run_cycle()
        req = self.client.submit_order.call_args.args[0]
        self.assertIsInstance(req, MarketOrderRequest)
        # equity $100k x 70% / $160 live price
        self.assertEqual((req.symbol, req.side, req.qty, req.time_in_force), ("SPY", OrderSide.BUY, 437.5,
                                                                              TimeInForce.DAY))
        self.assertTrue(req.client_order_id.startswith("swg-20261106-SPY-"))
        logged = [c.kwargs for c in bot.db.log.call_args_list if c.args == ("orders",)]
        self.assertEqual((logged[0]["note"], logged[0]["limit_price"]), ("swing: market, target 70.0%", 160.0))
        self.assertEqual(sum(c.args == ("decisions",) for c in bot.db.log.call_args_list), 4)
        self.assertIn("SPY $160.00: trend in", self.notes[-1])

    def test_trend_exit_closes_the_whole_position(self):
        self.closes = pd.DataFrame({s: history("down") for s in SYMS})
        self.prices = {s: 140.0 for s in SYMS}
        self.client.get_all_positions.return_value = [NS(symbol="SPY", qty="10.5")]
        self.run_cycle()
        self.client.close_position.assert_called_once_with("SPY")
        self.client.submit_order.assert_not_called()

    def test_fresh_start_skips_a_dip_trade_already_running(self):
        # QQQ: +1 a day to 155, then 153.5, 152, 150.5 (RSI(2) 8.7 above the 20-day average 147.75: the rule buys
        # on 11-04), then 150.0 yesterday and 150.4 now, both under the 5-day average: the trade is still running
        self.closes["QQQ"] = [100.0 + i for i in range(56)] + [153.5, 152.0, 150.5, 150.0]
        self.prices["QQQ"] = 150.4
        self.run_cycle()
        self.assertEqual([c.args[0].symbol for c in self.client.submit_order.call_args_list], ["SPY"])
        self.assertIn("dip trade running but not joined", self.notes[-1])
        # had the bot bought it on 11-04, it keeps it
        self.client.submit_order.reset_mock()
        bot.state.update(swing_day=None, swing_tried=None)
        self.client.get_all_positions.return_value = [NS(symbol="SPY", qty="437.5"), NS(symbol="QQQ", qty="50")]
        self.run_cycle()
        self.client.submit_order.assert_not_called()
        self.client.close_position.assert_not_called()
        self.assertIn("dip trade on -> 7.5% of equity", self.notes[-1])

    def test_holding_the_target_places_nothing(self):
        self.client.get_all_positions.return_value = [NS(symbol="SPY", qty="437.5")]
        self.run_cycle()
        self.client.submit_order.assert_not_called()
        self.assertIn("positions already match", self.notes[-1])

    def test_open_order_is_left_alone(self):
        self.client.get_orders.return_value = [NS(symbol="SPY", id="o1")]
        self.run_cycle()
        self.client.submit_order.assert_not_called()
        self.assertIn("SPY: an order is already open", self.notes[-1])

    def test_stale_price_skips_that_symbol_then_raises_for_a_retry(self):
        self.ages["QQQ"] = timedelta(hours=1)
        self.client.get_all_positions.return_value = [NS(symbol="QQQ", qty="5")]   # held, but unpriced: kept
        with self.assertRaisesRegex(RuntimeError, "QQQ: no fresh price"):
            self.run_cycle()
        self.assertEqual([c.args[0].symbol for c in self.client.submit_order.call_args_list], ["SPY"])
        self.client.close_position.assert_not_called()
        self.assertIsNone(bot.state["swing_day"])
        notes = len(self.notes)
        self.client.get_all_positions.return_value = [NS(symbol="SPY", qty="437.5"), NS(symbol="QQQ", qty="5")]
        with self.assertRaises(RuntimeError):
            self.run_cycle(minutes_to_close=13.8)       # the retry: SPY done, nothing new to report
        self.assertEqual(self.client.submit_order.call_count, 1)
        self.assertEqual(len(self.notes), notes)

    def test_short_old_or_gapped_history_never_trades(self):
        self.client.get_all_positions.return_value = [NS(symbol="SPY", qty="10")]
        gapped = self.closes.copy()
        gapped.iloc[-10] = float("nan")   # one missing day would make the average NaN = "trend out" = sell
        for closes in (self.closes.iloc[-30:], self.closes.iloc[:-1], gapped):   # short; ends before yesterday; gap
            self.closes = closes
            with self.assertRaisesRegex(RuntimeError, "daily bars missing"):
                self.run_cycle()
        self.client.submit_order.assert_not_called()
        self.client.close_position.assert_not_called()


class SwingAndIntraday(BotTestCase):
    def test_close_out_keeps_swing_positions_and_orders(self):
        spy = NS(symbol="SPY", asset_class=AssetClass.US_EQUITY)
        aapl = NS(symbol="AAPL", asset_class=AssetClass.US_EQUITY)
        self.client.get_all_positions.return_value = [spy, aapl]
        self.client.get_orders.return_value = [
            NS(id="o1", symbol="SPY", side=OrderSide.BUY, asset_class=AssetClass.US_EQUITY, client_order_id="swg-x"),
            NS(id="o2", symbol="AAPL", side=OrderSide.BUY, asset_class=AssetClass.US_EQUITY, client_order_id=None)]
        with mock.patch.object(bot, "flatten") as flat:
            bot.flatten_all_once(NOW)
        flat.assert_called_once_with("AAPL")
        self.client.cancel_order_by_id.assert_called_once_with("o2")

    def test_close_out_flattens_them_when_swing_is_off(self):
        self.client.get_all_positions.return_value = [NS(symbol="SPY", asset_class=AssetClass.US_EQUITY)]
        self.client.get_orders.return_value = []
        with mock.patch.object(config, "SWING", False), mock.patch.object(bot, "flatten") as flat:
            bot.flatten_all_once(NOW)
        flat.assert_called_once_with("SPY")

    def test_intraday_caps_ignore_swing_positions(self):
        self.client.get_account.return_value = NS(equity="10000", last_equity="10000")
        self.client.get_all_positions.return_value = [
            NS(symbol="SPY", market_value="7000", asset_class=AssetClass.US_EQUITY),
            NS(symbol="MU", market_value="500", asset_class=AssetClass.US_EQUITY)]
        self.client.get_orders.return_value = []
        clock = NS(timestamp=NOW, next_close=NOW + timedelta(minutes=15), next_open=NOW + timedelta(days=3),
                   is_open=True)
        with mock.patch.object(bot, "swing_cycle") as cycle:
            bot.run_trading_cycle(clock, stocks=True)
        ctx = cycle.call_args.args[0]
        self.assertEqual((set(ctx["positions"]), ctx["swing_held"], ctx["exposure"]), ({"MU"}, {"SPY"}, 500.0))

    def test_inverse_fund_blocked_while_swing_holds_spy(self):
        ctx = make_ctx(NOW, swing_held={"SPY"})
        self.assertEqual(bot.try_enter("SH", NS(atr_14=1.0), ctx), "skip: opposite index fund held (would cancel out)")

    def test_cost_gate_blocks_btc_scalps(self):
        quote = NS(bid_price="69995", ask_price="70000")
        with mock.patch.object(bot, "latest_quote", return_value=quote), \
                mock.patch.object(bot, "asset_limits", return_value=(1e-9, 1e-4)), \
                mock.patch.object(config, "USE_LLM_VETO", False):
            result = bot.try_enter("BTC/USD", NS(atr_14=100.0), make_ctx(NOW, minutes_to_close=0))
        self.assertEqual(result, "skip: round-trip costs 2.33R > 0.25R of the risk")
        self.client.submit_order.assert_not_called()

    def test_config_checks(self):
        with mock.patch.object(config, "STRATEGY", "orb2"), self.assertRaises(SystemExit):
            bot.check_config()
        with mock.patch.object(config, "SWING_DIP_WEIGHT", 0.5), self.assertRaises(SystemExit):
            bot.check_config()
        bot.check_config()   # the defaults pass
        self.assertIn("swing ETFs (SPY trend 70%", bot.startup_banner())
        self.assertNotIn("BTC", bot.startup_banner())


if __name__ == "__main__":
    unittest.main()
