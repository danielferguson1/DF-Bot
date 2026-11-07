"""Momentum day-trading tests: rules (momentum.py), the backtest replay, and order handling in bot.py (Alpaca mocked).
Offline."""
import unittest
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace as NS
from unittest import mock

import numpy as np
import pandas as pd

import config
import momentum
from backtest import simulate_momentum
from test_bot import BotTestCase, bot, make_ctx  # sets dummy keys before importing bot
from alpaca.trading.enums import AssetClass, OrderSide, OrderStatus, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

NY = "America/New_York"
OPEN = pd.Timestamp("2026-11-06 09:30", tz=NY)   # a Friday; EST, so 14:30 UTC


def minute_bars(closes, open_time=OPEN, start=0, symbol=None, volume=1000.0):
    """Bars starting `start` minutes after the open: bar k opens at the previous close and closes at closes[k]."""
    idx = pd.DatetimeIndex([open_time + pd.Timedelta(minutes=start + i) for i in range(len(closes))]).tz_convert("UTC")
    c = np.asarray(closes, float)
    o = np.r_[c[0], c[:-1]]
    df = pd.DataFrame({"open": o, "high": np.maximum(o, c), "low": np.minimum(o, c), "close": c,
                       "volume": volume, "vwap": (o + c) / 2}, index=idx)
    if symbol:
        df.index = pd.MultiIndex.from_product([[symbol], df.index], names=["symbol", "timestamp"])
    return df


class Rules(unittest.TestCase):
    def test_check_times(self):
        self.assertEqual(momentum.check_minutes(390), list(range(30, 361, 30)))   # 10:00 .. 15:30
        self.assertEqual(momentum.check_minutes(210), list(range(30, 181, 30)))   # half day: 10:00 .. 12:30

    def test_minute_prices_vwap_and_open(self):
        bars = minute_bars([10.0, 11.0, 12.0, 13.0], start=0).drop(index=[OPEN + pd.Timedelta(minutes=2)])
        self.assertEqual(list(momentum.minute_prices(bars, OPEN, 6)[:5]), [10, 11, 11, 13, 13])  # gap carried
        v = momentum.vwap_by_minute(bars, OPEN, 6)
        self.assertAlmostEqual(v[1], (10 * 1000 + 10.5 * 1000) / 2000)
        self.assertEqual(momentum.session_open(bars, OPEN), 10.0)
        late = minute_bars([20.0, 21.0], start=3)
        self.assertTrue(np.isnan(momentum.minute_prices(late, OPEN, 6)[0]))
        self.assertEqual(momentum.session_open(late, OPEN), 20.0)       # no opening minute: its first close

    def test_sigma_is_the_average_move_and_needs_10_sessions(self):
        past = [(100.0, np.full(390, 100.0 + 0.1 * (i + 1))) for i in range(14)]   # moves 0.1% .. 1.4%
        s = momentum.sigma(past)
        self.assertAlmostEqual(s[29], np.mean([0.001 * (i + 1) for i in range(14)]))
        halfdays = [(100.0, np.full(210, 100.5)) for _ in range(5)] + past[:9]
        self.assertTrue(np.isnan(momentum.sigma(halfdays)[300]))        # only 9 sessions have 15:00
        self.assertFalse(np.isnan(momentum.sigma(halfdays)[100]))

    def test_bands_target_and_size(self):
        self.assertEqual(momentum.bands(100.0, 99.0, 0.01), (101.0, 98.01))   # gap up: lower band from the close
        self.assertEqual(momentum.target(102, vwap=101.5, upper=101, lower=98), 1)
        self.assertEqual(momentum.target(101.2, vwap=101.5, upper=101, lower=98), 0)   # above the band, below VWAP
        self.assertEqual(momentum.target(97, vwap=99, upper=101, lower=98), -1)
        self.assertEqual(momentum.target(97.5, vwap=97, upper=101, lower=98), 0)      # below the band, above VWAP
        self.assertEqual(momentum.shares(10_000, 656.10), 15)


class Backtest(unittest.TestCase):
    def test_trend_day(self):
        flat = np.full(390, 100.2)                                     # every past session moved +0.2% from 100
        sessions = [(date(2026, 10, d), 100.0, flat, flat, 390) for d in range(19, 31) if date(2026, 10, d).weekday() < 5]
        rise = 100.2 + 2.0 * np.arange(1, 391) / 390                    # today: a steady climb from yesterday's close
        vw = np.cumsum(rise) / np.arange(1, 391)
        sessions.append((date(2026, 11, 2), 100.2, rise, vw, 390))
        res = simulate_momentum(sessions, cost=0.0, equity=100_000)
        day = res.loc[pd.Timestamp("2026-11-02")]
        # 10:00: 100.35 < upper band 100.2 x 1.002 = 100.40, no trade; 10:30: 100.51 > band and VWAP -> long to 15:50
        n = int(100_000 / 100.2)
        self.assertAlmostEqual(day.ret * 100_000, n * (rise[379] - rise[59]), places=6)
        self.assertEqual(day.entries, 1)
        self.assertTrue((res.ret.iloc[:-1] == 0).all())               # flat days: no trades
        costly = simulate_momentum(sessions, cost="alpaca", equity=100_000).loc[pd.Timestamp("2026-11-02")]
        self.assertAlmostEqual((day.ret - costly.ret) * 100_000, 2 * n * 0.005 + n * rise[379] * 0.278e-4, places=6)


TODAY = date(2026, 11, 6)
AT_1000 = datetime(2026, 11, 6, 15, 0, 10, tzinfo=timezone.utc)     # 10:00:10 ET


class Cycle(BotTestCase):
    def setUp(self):
        super().setUp()
        self.state = {"day": TODAY, "open": OPEN, "length": 390, "sigma": np.full(390, 0.002), "prev": 100.0,
                      "done": set(), "shares": None}
        self.today = minute_bars(np.linspace(100.0, 101.0, 30), symbol="QQQ")   # 9:30 .. 9:59, up 1%
        for target, new in (("momentum_prepare", mock.Mock(side_effect=lambda now: self.state)),
                            ("data_client", mock.Mock(**{"get_stock_bars.side_effect":
                                                         lambda req: NS(df=self.today)}))):
            patcher = mock.patch.object(bot, target, new)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = mock.patch.multiple(config, STRATEGY="momentum", MOMENTUM_SYMBOL="QQQ")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.client.get_orders.return_value = []
        self.client.submit_order.side_effect = lambda req: NS(id=f"o-{req.side.value}")
        self.client.close_position.return_value = NS(id="close-1")
        self.client.get_order_by_id.return_value = NS(status=OrderStatus.FILLED, filled_qty="100")

    def cycle(self, now=AT_1000, held=0):
        positions = {"QQQ": NS(symbol="QQQ", qty=str(held))} if held else {}
        bot.momentum_cycle(make_ctx(now, positions, last_equity=10_000.0))

    def test_long_entry_once_per_check(self):
        self.cycle()   # 101 > upper band 100.2 and VWAP ~100.5
        req = self.client.submit_order.call_args.args[0]
        self.assertIsInstance(req, MarketOrderRequest)
        self.assertEqual((req.symbol, req.side, req.qty, req.time_in_force), ("QQQ", OrderSide.BUY, 100, TimeInForce.DAY))
        self.assertTrue(req.client_order_id.startswith("mom-20261106-"))
        logged = next(c.kwargs for c in bot.db.log.call_args_list if c.args == ("orders",))
        self.assertEqual(logged["note"], "entry: momentum long")
        self.assertIn("-> long 100 (BUY 100)", self.notes[-1])
        self.cycle(AT_1000 + timedelta(minutes=1), held=100)   # same check again: nothing
        self.cycle(AT_1000 + timedelta(minutes=12), held=100)  # between checks: nothing
        self.assertEqual(self.client.submit_order.call_count, 1)

    def test_exit_when_price_falls_back_under_vwap(self):
        self.state["done"].add(30)
        self.state["shares"] = 100
        self.today = minute_bars(np.r_[np.linspace(100.0, 101.0, 30), np.linspace(101.0, 100.3, 30)], symbol="QQQ")
        self.cycle(AT_1000 + timedelta(minutes=30), held=100)  # 10:30: 100.3 above the band but under VWAP ~100.6
        self.client.close_position.assert_called_once_with("QQQ")
        self.client.submit_order.assert_not_called()
        self.assertIn("-> flat (SELL 100 to close)", self.notes[-1])

    def test_flip_closes_first_then_shorts(self):
        self.state["done"].add(30)
        self.state["shares"] = 100
        self.today = minute_bars(np.r_[np.linspace(100.0, 101.0, 30), np.linspace(101.0, 99.0, 30)], symbol="QQQ")
        self.cycle(AT_1000 + timedelta(minutes=30), held=100)  # 99.0 under the lower band 99.8 and VWAP
        self.client.close_position.assert_called_once_with("QQQ")
        self.client.get_order_by_id.assert_called_with("close-1")   # waited for the close to fill
        req = self.client.submit_order.call_args.args[0]
        self.assertEqual((req.side, req.qty), (OrderSide.SELL, 100))
        self.assertIn("-> short 100 (SELL 100 to close, SELL 100)", self.notes[-1])

    def test_stale_bars_raise_and_the_check_stays_due(self):
        self.today = minute_bars(np.linspace(100.0, 101.0, 21), symbol="QQQ")   # last bar ended 9:51
        with self.assertRaisesRegex(RuntimeError, "too old to decide on"):
            self.cycle()
        self.assertNotIn(30, self.state["done"])
        self.client.submit_order.assert_not_called()

    def test_open_order_blocks_a_new_one(self):
        self.client.get_orders.return_value = [NS(symbol="QQQ", id="pending")]
        with self.assertRaisesRegex(RuntimeError, "still open"):
            self.cycle()
        self.assertNotIn(30, self.state["done"])

    def test_leftover_position_is_closed_before_the_first_check(self):
        with mock.patch.object(bot, "flatten") as flat:
            self.cycle(AT_1000 - timedelta(minutes=20), held=15)   # 9:40 with 15 shares from an earlier session
        flat.assert_called_once_with("QQQ")
        self.assertIn("left open from an earlier session", self.notes[-1])

    def test_off_unless_selected(self):
        with mock.patch.object(config, "STRATEGY", "none"):
            self.cycle()
        bot.momentum_prepare.assert_not_called()


class Prepare(BotTestCase):
    def setUp(self):
        super().setUp()
        bot._mom.clear()
        days = list(pd.bdate_range(end="2026-11-05", periods=14).date) + [TODAY]
        self.client.get_calendar.return_value = [NS(date=d, open=datetime(d.year, d.month, d.day, 9, 30),
                                                    close=datetime(d.year, d.month, d.day, 16, 0)) for d in days]
        frames = [minute_bars(np.full(390, 100.3), open_time=pd.Timestamp(d).tz_localize(NY) + pd.Timedelta("9h30min"),
                              symbol="QQQ") for d in days[:-1]]
        for f in frames:      # each session opens at 100, then trades at 100.3: a 0.3% move every minute
            f.iloc[0, f.columns.get_loc("open")] = 100.0
        self.bars = pd.concat(frames)
        patcher = mock.patch.object(bot, "data_client", mock.Mock(**{"get_stock_bars.side_effect":
                                                                     lambda req: NS(df=self.bars)}))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_noise_area_from_the_last_14_sessions(self):
        st = bot.momentum_prepare(AT_1000)
        self.assertAlmostEqual(st["sigma"][29], 0.003)
        self.assertEqual((st["prev"], st["length"], st["open"]), (100.3, 390, OPEN))
        req = bot.data_client.get_stock_bars.call_args.args[0]
        self.assertEqual(req.feed, bot.DataFeed.SIP)
        bot.momentum_prepare(AT_1000 + timedelta(minutes=30))   # cached for the day
        self.assertEqual(bot.data_client.get_stock_bars.call_count, 1)

    def test_patchy_history_raises(self):
        self.bars = self.bars.iloc[::2]
        with self.assertRaisesRegex(RuntimeError, "1-minute bars missing"):
            bot.momentum_prepare(AT_1000)


class Wiring(BotTestCase):
    def test_defaults_pass_and_a_shared_symbol_is_refused(self):
        bot.check_config()   # momentum on QQQ, swing dips in SPY/IWM/DIA
        self.assertNotIn("QQQ", bot.swing_symbols())
        self.assertIn("day trading QQQ (intraday momentum, 1x", bot.startup_banner())
        with mock.patch.object(config, "SWING_DIP_SYMBOLS", ("SPY", "QQQ")), self.assertRaises(SystemExit):
            bot.check_config()

    def market_open(self):
        self.client.get_account.return_value = NS(equity="10000", last_equity="10000")
        self.client.get_all_positions.return_value = []
        self.client.get_orders.return_value = []
        return NS(timestamp=AT_1000, next_close=AT_1000 + timedelta(hours=6), next_open=AT_1000 + timedelta(days=3),
                  is_open=True)

    def test_one_failing_step_does_not_stop_the_other(self):
        clock = self.market_open()
        with mock.patch.object(bot, "momentum_cycle", side_effect=RuntimeError("momentum: stale")), \
                mock.patch.object(bot, "swing_cycle") as sw, self.assertRaisesRegex(RuntimeError, "stale"):
            bot.run_trading_cycle(clock, stocks=True)
        sw.assert_called_once()

    def test_both_failing_report_both(self):
        # the main loop alerts once per distinct message: a day-long momentum error must not hide a swing failure
        clock = self.market_open()
        with mock.patch.object(bot, "momentum_cycle", side_effect=RuntimeError("momentum: stale")), \
                mock.patch.object(bot, "swing_cycle", side_effect=RuntimeError("swing: SPY: no fresh price")), \
                self.assertRaisesRegex(RuntimeError, "momentum: stale; RuntimeError: swing: SPY: no fresh price"):
            bot.run_trading_cycle(clock, stocks=True)

    def test_close_out_flattens_the_momentum_position_and_keeps_swing(self):
        self.client.get_orders.return_value = []
        self.client.get_all_positions.return_value = [NS(symbol="QQQ", asset_class=AssetClass.US_EQUITY),
                                                      NS(symbol="SPY", asset_class=AssetClass.US_EQUITY)]
        with mock.patch.object(bot, "flatten") as flat:
            bot.flatten_all_once(AT_1000 + timedelta(hours=5, minutes=50))
        flat.assert_called_once_with("QQQ")


class AllModeCosts(unittest.TestCase):
    def test_cost_flag_reaches_both_parts(self):
        import backtest
        days = pd.bdate_range("2026-01-05", periods=30)
        curve = pd.DataFrame({"equity": np.linspace(100_000, 101_000, 30), "exposure": 0.7}, index=days)
        closes = pd.DataFrame({"SPY": np.linspace(600, 610, 30)}, index=days)
        mom = pd.DataFrame({"equity": 1.0, "ret": 0.001, "entries": 1}, index=days)
        with mock.patch.object(backtest, "swing_curve", return_value=(curve, [], closes)) as sc, \
                mock.patch.object(backtest, "load_minute_sessions", return_value=[]), \
                mock.patch.object(backtest, "simulate_momentum", return_value=mom) as sm, \
                mock.patch("builtins.print"):
            backtest.all_main(None, "k", "s", NS(start="2026-01-01", cost_bps=1.0))
            backtest.all_main(None, "k", "s", NS(start="2026-01-01", cost_bps=None))
        self.assertEqual([c.args[2] for c in sc.call_args_list], [1e-4, 2e-4])        # swing: bps -> fraction
        self.assertEqual([c.args[1] for c in sm.call_args_list], [1.0, "alpaca"])     # momentum: bps or its default


if __name__ == "__main__":
    unittest.main()
