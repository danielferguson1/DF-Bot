"""python -m unittest -v"""
import unittest
from types import SimpleNamespace as Row

import pandas as pd

import config
import metrics
import strategy
from backtest import simulate


def bar(**kw):
    base = dict(sma_5=101, sma_20=100, rsi_14=45, rel_volume=1.5, atr_14=1.0, close=100, vwap=99)
    base.update(kw)
    return Row(**base)


def bars(closes, start="2026-11-02 10:00"):
    idx = pd.date_range(start, periods=len(closes), freq="5min", tz="America/New_York").tz_convert("UTC")
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({"open": c, "high": c + 0.5, "low": c - 0.5, "close": c, "volume": 1000.0})


class Indicators(unittest.TestCase):
    def test_rsi_flat_is_neutral_and_rising_is_100(self):
        self.assertEqual(strategy.add_indicators(bars([100] * 30))["rsi_14"].iloc[-1], 50)
        self.assertEqual(strategy.add_indicators(bars(range(100, 130)))["rsi_14"].iloc[-1], 100)

    def test_regular_hours_drops_premarket(self):
        self.assertEqual(len(strategy.regular_hours(bars([1] * 12, start="2026-11-02 09:00"))), 6)


class Signal(unittest.TestCase):
    def test_buy_when_all_conditions_met(self):
        self.assertEqual(strategy.signal(bar(rsi_14=35), bar(rsi_14=42), False, True), "BUY")

    def test_no_buy_against_regime_or_low_volume_or_nan(self):
        self.assertEqual(strategy.signal(bar(rsi_14=35), bar(rsi_14=42), False, False), "HOLD")
        self.assertEqual(strategy.signal(bar(rsi_14=35), bar(rsi_14=42, rel_volume=0.5), False, True), "HOLD")
        self.assertEqual(strategy.signal(bar(rsi_14=35), bar(sma_20=float("nan")), False, True), "HOLD")

    def test_sell_when_held_and_trend_breaks(self):
        self.assertEqual(strategy.signal(bar(), bar(sma_5=99), True, True), "SELL")
        self.assertEqual(strategy.signal(bar(), bar(), True, True), "HOLD")


class Freshness(unittest.TestCase):
    def test_yesterdays_last_bar_is_stale_at_the_open(self):
        ts = lambda s: pd.Timestamp(s, tz="America/New_York")
        self.assertFalse(strategy.bar_is_fresh(ts("2026-11-05 15:55"), ts("2026-11-06 09:31")))
        self.assertTrue(strategy.bar_is_fresh(ts("2026-11-06 09:30"), ts("2026-11-06 09:35:10")))
        self.assertFalse(strategy.bar_is_fresh(ts("2026-11-06 09:30"), ts("2026-11-06 09:46")))


class Sizing(unittest.TestCase):
    def test_risk_limited_and_cap_limited(self):
        # risk: 100k * 0.25% / (1.5 * $2 ATR) = 83 shares; cap: $2000 / $10 = 200 -> 83
        self.assertEqual(strategy.position_size(100_000, 10, 2.0), 83)
        # cap: $2000 / $500 = 4 shares
        self.assertEqual(strategy.position_size(100_000, 500, 2.0), 4)
        self.assertEqual(strategy.position_size(100_000, 10, 0), 0)

    def test_fractional_crypto_size(self):
        # risk: 250 / (1.5 * 500) = 0.333 BTC; cap: 2000 / 70000 = 0.0285714... -> cap wins, floored
        self.assertEqual(strategy.position_size(100_000, 70_000, 500, qty_increment=1e-9), 0.028571428)

    def test_round_to_increment(self):
        self.assertEqual(strategy.round_to_increment(69_151.537, 0.01), 69_151.53)
        self.assertEqual(strategy.round_to_increment(69_151.531, 0.01, up=True), 69_151.54)
        self.assertEqual(strategy.round_to_increment(0.3, 0.1), 0.3)  # no float noise


class Metrics(unittest.TestCase):
    def test_fifo_round_trip_per_order(self):
        fills = [("X", "buy", 10, 100, "a"), ("X", "buy", 10, 110, "b"),
                 ("X", "sell", 10, 120, "c"), ("X", "sell", 5, 120, "c")]
        self.assertEqual(metrics.round_trips(fills), [250.0])  # 10*20 + 5*10

    def test_drawdown(self):
        eq = pd.Series([100, 120, 90, 130], index=pd.date_range("2026-01-01", periods=4, tz="UTC"))
        self.assertAlmostEqual(metrics.summarize(eq, [])["max_drawdown"], -0.25)


class Backtest(unittest.TestCase):
    def test_entry_next_open_then_stop_out(self):
        df = pd.DataFrame({
            "open": [100] * 4, "high": [100] * 4, "low": [100, 100, 90, 95], "close": [100, 100, 95, 95],
            "sma_5": [101] * 4, "sma_20": [100] * 4, "rsi_14": [35, 42, 42, 42], "rel_volume": [1.5] * 4,
            "atr_14": [1.0] * 4, "regime_ok": [True] * 4,
        }, index=bars([0] * 4).index)
        trades = simulate(df, cost=0.0)
        # signal at bar 1 close, entry at bar 2 open = 100, stop = 98.5 hit by low 90 -> exit at 98.5
        qty = strategy.position_size(100_000, 100, 1.0)
        self.assertEqual(len(trades), 1)
        self.assertAlmostEqual(trades[0][1], qty * (98.5 - 100))

    def test_crypto_holds_overnight_and_takes_profit_at_close(self):
        # bar 1 signals BUY at 15:55 ET; a stock would be flattened at the day's last bar, crypto is not
        idx = pd.DatetimeIndex(["2026-11-02 15:50", "2026-11-02 15:55", "2026-11-02 16:00",
                                "2026-11-03 09:00"], tz="America/New_York").tz_convert("UTC")
        df = pd.DataFrame({
            "open": [100, 100, 100, 101], "high": [100, 100, 101, 104], "low": [100, 100, 99, 100],
            "close": [100, 100, 100.5, 103.5],
            "sma_5": [101] * 4, "sma_20": [100] * 4, "rsi_14": [35, 42, 42, 42], "rel_volume": [1.5] * 4,
            "atr_14": [1.0] * 4, "regime_ok": [True] * 4,
        }, index=idx)
        trades = simulate(df, cost=0.0, crypto=True)
        qty = strategy.position_size(100_000, 100, 1.0, qty_increment=1e-9)
        self.assertEqual(len(trades), 1)  # entry 100 at 16:00, target 103 hit at next day's close 103.5
        self.assertEqual(trades[0][0], idx[3])
        self.assertAlmostEqual(trades[0][1], qty * 3.5)


if __name__ == "__main__":
    unittest.main()
