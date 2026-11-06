"""python -m unittest -v"""
import unittest
from pathlib import Path
from types import SimpleNamespace as Row
from unittest import mock

import numpy as np
import pandas as pd

import config
import metrics
import strategy
import timemachine
from backtest import simulate

FIXTURE = Path(__file__).with_name("fixtures") / "AAPL-5m-iex.csv.gz"  # real IEX bars, 2026-07-29..11-05
ET = "America/New_York"


def real_bars() -> pd.DataFrame:
    return pd.read_csv(FIXTURE, index_col="timestamp", parse_dates=["timestamp"])


def ts(s):
    return pd.Timestamp(s, tz=ET)


def bar(**kw):
    base = dict(sma_5=101, sma_20=100, rsi_14=45, rel_volume=1.5, atr_14=1.0, close=100, vwap=99)
    base.update(kw)
    return Row(**base)


def bars(closes, start="2026-11-02 10:00"):
    idx = pd.date_range(start, periods=len(closes), freq="5min", tz="America/New_York").tz_convert("UTC")
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({"open": c, "high": c + 0.5, "low": c - 0.5, "close": c, "volume": 1000.0})


class Config(unittest.TestCase):
    def test_every_symbol_has_a_sector_and_inverse_is_watched(self):
        self.assertEqual(set(config.WATCHLIST + config.CRYPTO_WATCHLIST) - set(config.SECTORS), set())
        self.assertLessEqual(config.INVERSE, set(config.WATCHLIST))


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


class SessionAtr(unittest.TestCase):
    def test_overnight_gap_is_not_one_bar_of_range(self):
        day1 = bars([100] * 20, start="2026-11-02 14:20")         # 14:20-15:55
        day2 = bars([120] * 20, start="2026-11-03 09:30")         # gaps up $20 overnight
        df = pd.concat([day1, day2])
        with_gap = strategy.add_indicators(df, sessions=False)["atr_14"]
        per_session = strategy.add_indicators(df, sessions=True)["atr_14"]
        self.assertAlmostEqual(per_session.iloc[-1], 1.0)        # every bar's high-low is $1
        self.assertGreater(with_gap.iloc[25], 2.0)               # the $20 gap inflates the ATR

    def test_real_morning_atr_ignores_overnight_gap(self):
        df = strategy.add_indicators(real_bars())
        et = df.index.tz_convert(ET)
        for first in np.flatnonzero((et.hour == 9) & (et.minute == 30))[1:]:
            morning = df.iloc[first:first + 14]                  # ATR at 10:35 = the 14 bars of that morning
            prev = morning["close"].shift()                      # no previous close for the 9:30 bar
            tr = pd.concat([morning["high"] - morning["low"], (morning["high"] - prev).abs(),
                            (morning["low"] - prev).abs()], axis=1).max(axis=1)
            self.assertAlmostEqual(morning["atr_14"].iloc[-1], tr.mean())


class EntryRules(unittest.TestCase):
    def test_entry_window_after_open(self):
        self.assertFalse(strategy.in_entry_window(ts("2026-11-06 09:44")))
        self.assertTrue(strategy.in_entry_window(ts("2026-11-06 09:45")))

    def test_reward_risk(self):
        self.assertEqual(strategy.reward_risk(100, 98.5, 103), 2.0)
        self.assertEqual(strategy.reward_risk(100, 100, 103), 0.0)  # no risk below the entry = refuse

    def test_cooldown(self):
        self.assertTrue(strategy.cooled_down(None, ts("2026-11-06 10:00")))
        self.assertFalse(strategy.cooled_down(ts("2026-11-06 10:00"), ts("2026-11-06 10:29")))
        self.assertTrue(strategy.cooled_down(ts("2026-11-06 10:00"), ts("2026-11-06 10:30")))


def frame(n, start, **cols):
    base = dict(open=100.0, high=100.0, low=100.0, close=100.0, sma_5=101, sma_20=100,
                rsi_14=[35, 42] * (n // 2), rel_volume=1.5, atr_14=1.0, regime_ok=True)
    base.update(cols)
    idx = pd.date_range(start, periods=n, freq="5min", tz=ET).tz_convert("UTC")
    return pd.DataFrame({k: (v if isinstance(v, list) else [v] * n) for k, v in base.items()}, index=idx)


class BacktestRules(unittest.TestCase):
    def test_no_entry_in_first_minutes(self):
        # signals every other bar from 9:30; the first order allowed goes out at 9:45 (after the 9:40 bar)
        trades = simulate(frame(6, "2026-11-02 09:30", low=[100, 100, 100, 100, 100, 90]), cost=0.0)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0][0].tz_convert(ET).strftime("%H:%M"), "09:55")

    def test_cooldown_blocks_reentry_after_stop(self):
        low = [100, 100, 90] + [100] * 9 + [90] * 2               # stopped at bar 2, signals keep coming
        trades = simulate(frame(14, "2026-11-02 10:00", low=low), cost=0.0)
        exits = [t.tz_convert(ET).strftime("%H:%M") for t, _ in trades]
        self.assertEqual(exits, ["10:10", "11:00"])                # re-entry only 30 min after the 10:10 exit

    def test_min_reward_risk(self):
        with mock.patch.object(config, "MIN_REWARD_RISK", 2.5):   # the ATR plan is 2R
            self.assertEqual(simulate(frame(4, "2026-11-02 10:00", low=[100, 100, 90, 95]), cost=0.0), [])


def synthetic_days(n_days, drift):
    """n_days complete sessions; each bar closes `drift` above the last and ranges +-0.1%."""
    days = []
    for _ in range(n_days):
        c = 100 * (1 + drift) ** np.arange(timemachine.BARS_PER_DAY)
        o = np.concatenate([[100], c[:-1]])
        days.append(np.stack([o, np.maximum(o, c) * 1.001, np.minimum(o, c) * 0.999, c], axis=1))
    return np.array(days)


class TimeMachine(unittest.TestCase):
    def test_steady_uptrend_hits_target(self):
        days = synthetic_days(20, 0.001)
        hist = timemachine.history(days, 6, 0.005, 0.01)
        self.assertEqual((hist["n"], hist["target"]), (20, 1.0))
        mc = timemachine.replay(days, 6, 0.005, 0.01, replays=500)
        self.assertEqual(mc["target"], 1.0)
        self.assertAlmostEqual(mc["avg_r"], 2.0, places=1)

    def test_downtrend_hits_stop(self):
        mc = timemachine.replay(synthetic_days(20, -0.001), 6, 0.005, 0.01, replays=500)
        self.assertEqual(mc["stop"], 1.0)
        self.assertLess(mc["avg_r"], -0.99)

    def test_too_little_history_or_too_late(self):
        df = real_bars()
        self.assertIsNone(timemachine.run(df.iloc[:78 * 5], ts("2026-11-06 10:00"), 100, 99, 102))
        self.assertIsNone(timemachine.run(df, ts("2026-11-06 15:50"), 100, 99, 102))  # past the flatten
        self.assertIsNone(timemachine.run(df, ts("2026-11-06 10:00"), 100, 101, 102))  # stop above entry

    def test_real_aapl(self):
        df = real_bars()
        tm = timemachine.run(df, ts("2026-11-06 10:00:10"), 270.0, 268.5, 273.0, replays=2000)
        mc, hist = tm["mc"], tm["history"]
        self.assertEqual(hist["n"], config.TM_DAYS)               # the last 60 of the fixture's 71 full days
        for r in (mc, hist):
            self.assertAlmostEqual(r["target"] + r["stop"] + r["open"], 1.0)
            self.assertTrue(-1.5 < r["avg_r"] < 2.5)
        again = timemachine.run(df, ts("2026-11-06 10:00:10"), 270.0, 268.5, 273.0, replays=2000)
        self.assertEqual(again, tm)                                # fixed seed: repeatable
        self.assertIn("TM 60d: replays target", timemachine.describe(tm))

    def test_only_sessions_before_today_are_used(self):
        df = real_bars()
        days = timemachine.full_sessions(df, ts("2026-11-05 12:00").date())
        self.assertEqual(len(days), 70)                            # 11-05 itself is left out


if __name__ == "__main__":
    unittest.main()
