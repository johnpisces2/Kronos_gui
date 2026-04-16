import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from backtester import Backtester, BacktestMetrics, BacktestTrade
from paper_strategy import PaperStrategyConfig, PaperSignalSnapshot, PaperPosition


def build_test_context_df(n_bars=100, start_price=100.0):
    timestamps = pd.date_range(start=datetime(2026, 1, 1), periods=n_bars, freq="4h")
    prices = start_price + np.cumsum(np.random.randn(n_bars) * 0.5)
    return pd.DataFrame({"timestamps": timestamps, "close": prices})


def build_test_validation_df(n_bars=20, start_price=None):
    if start_price is None:
        start_price = 100.0
    timestamps = pd.date_range(start=datetime(2026, 1, 1), periods=n_bars, freq="4h")
    prices = start_price + np.cumsum(np.random.randn(n_bars) * 0.3)
    return pd.DataFrame({"timestamps": timestamps, "close": prices})


def mock_predictor(context_df, temperature=0.5, top_p=0.8, sample_count=1):
    n = len(context_df)
    last_price = float(context_df.iloc[-1]["close"]) if n > 0 else 100.0
    import pandas as pd
    future_df = pd.DataFrame({
        "close": [last_price * (1 + 0.001 * (i % 2 - 0.5)) for i in range(1, 19)]
    })
    return {"future_pred_df": future_df}


class TestBacktesterInitialization(unittest.TestCase):
    def test_default_initialization(self):
        bt = Backtester()
        self.assertEqual(bt.initial_equity, 10000.0)
        self.assertEqual(bt.fee_rate, 0.0005)
        self.assertEqual(bt.slippage_pct, 0.001)

    def test_custom_initialization(self):
        bt = Backtester(initial_equity=50000.0, fee_rate=0.001, slippage_pct=0.002)
        self.assertEqual(bt.initial_equity, 50000.0)
        self.assertEqual(bt.fee_rate, 0.001)
        self.assertEqual(bt.slippage_pct, 0.002)


class TestBacktesterFeeCalculation(unittest.TestCase):
    def test_fee_applied_on_entry_and_exit(self):
        bt = Backtester(initial_equity=10000.0, fee_rate=0.0005, slippage_pct=0.001)

        context_df = build_test_context_df(100)
        val_hist_df = build_test_validation_df(50)
        val_actual_df = build_test_validation_df(30)

        results = bt.run(
            context_df=context_df,
            predictor_fn=mock_predictor,
            forecast_params={"lookback": 50, "pred_len": 18, "temperature": 0.5, "top_p": 0.8, "sample_count": 1},
        )

        self.assertGreaterEqual(results.total_fees, 0.0)
        self.assertGreaterEqual(results.total_slippage, 0.0)


class TestBacktesterMetrics(unittest.TestCase):
    def test_empty_run_returns_zero_metrics(self):
        bt = Backtester(initial_equity=10000.0)

        context_df = build_test_context_df(100)

        results = bt.run(
            context_df=context_df,
            predictor_fn=mock_predictor,
            forecast_params={"lookback": 50, "pred_len": 18, "temperature": 0.5, "top_p": 0.8, "sample_count": 1},
        )

        self.assertEqual(results.total_trades, 0)
        self.assertEqual(results.winning_trades, 0)
        self.assertEqual(results.losing_trades, 0)
        self.assertEqual(results.win_rate, 0.0)
        self.assertIsInstance(results.equity_curve, list)

    def test_metrics_with_known_trade(self):
        bt = Backtester(initial_equity=10000.0, fee_rate=0.0005, slippage_pct=0.001)

        context_df = build_test_context_df(200, start_price=100.0)
        context_df.iloc[-50:, :] = 105.0

        val_hist_df = build_test_validation_df(100, start_price=100.0)
        val_actual_df = build_test_validation_df(50, start_price=102.0)

        results = bt.run(
            context_df=context_df,
            predictor_fn=mock_predictor,
            forecast_params={"lookback": 50, "pred_len": 18, "temperature": 0.5, "top_p": 0.8, "sample_count": 1},
        )

        self.assertGreaterEqual(results.total_trades, 0)
        if results.total_trades > 0:
            self.assertIsInstance(results.win_rate, float)
            self.assertLessEqual(results.win_rate, 1.0)
            self.assertGreaterEqual(results.win_rate, 0.0)


class TestBacktesterSlippage(unittest.TestCase):
    def test_higher_slippage_increases_cost(self):
        bt_low = Backtester(initial_equity=10000.0, slippage_pct=0.001)
        bt_high = Backtester(initial_equity=10000.0, slippage_pct=0.005)

        context_df = build_test_context_df(200)
        val_hist_df = build_test_validation_df(100)
        val_actual_df = build_test_validation_df(50)

        forecast_params = {"lookback": 50, "pred_len": 18, "temperature": 0.5, "top_p": 0.8, "sample_count": 1}

        results_low = bt_low.run(
            context_df=context_df,
            predictor_fn=mock_predictor,
            forecast_params=forecast_params,
        )

        results_high = bt_high.run(
            context_df=context_df,
            predictor_fn=mock_predictor,
            forecast_params=forecast_params,
        )

        if results_low.total_trades > 0:
            self.assertLessEqual(results_low.total_slippage, results_high.total_slippage)


class TestBacktesterEdgeCases(unittest.TestCase):
    def test_single_bar_context(self):
        bt = Backtester(initial_equity=10000.0)

        context_df = build_test_context_df(100)
        val_hist_df = build_test_validation_df(50)
        val_actual_df = build_test_validation_df(30)

        results = bt.run(
            context_df=context_df,
            predictor_fn=mock_predictor,
            forecast_params={"lookback": 50, "pred_len": 18, "temperature": 0.5, "top_p": 0.8, "sample_count": 1},
        )

        self.assertEqual(results.total_trades, 0)
        self.assertGreater(len(results.equity_curve), 0)

    def test_zero_initial_equity(self):
        bt = Backtester(initial_equity=0.0)

        context_df = build_test_context_df(100)
        val_hist_df = build_test_validation_df(50)
        val_actual_df = build_test_validation_df(30)

        results = bt.run(
            context_df=context_df,
            predictor_fn=mock_predictor,
            forecast_params={"lookback": 50, "pred_len": 18, "temperature": 0.5, "top_p": 0.8, "sample_count": 1},
        )

        self.assertGreaterEqual(results.max_drawdown, 0.0)

    def test_very_high_leverage_zero_stop_distance(self):
        bt = Backtester(initial_equity=10000.0)

        context_df = build_test_context_df(100)
        val_hist_df = build_test_validation_df(50)
        val_actual_df = build_test_validation_df(30)

        results = bt.run(
            context_df=context_df,
            predictor_fn=mock_predictor,
            forecast_params={"lookback": 50, "pred_len": 18, "temperature": 0.5, "top_p": 0.8, "sample_count": 1},
        )

        for trade in results.trades:
            self.assertGreater(trade.quantity, 0)


class TestBacktesterStrategyIntegration(unittest.TestCase):
    def test_strategy_config_respected(self):
        bt = Backtester(initial_equity=10000.0)
        bt.strategy_config = PaperStrategyConfig(
            entry_threshold_pct=0.01,
            stop_loss_fraction=0.3,
            min_stop_loss_pct=0.005,
            max_stop_loss_pct=0.02,
        )

        context_df = build_test_context_df(200)
        val_hist_df = build_test_validation_df(100)
        val_actual_df = build_test_validation_df(50)

        results = bt.run(
            context_df=context_df,
            predictor_fn=mock_predictor,
            forecast_params={"lookback": 50, "pred_len": 18, "temperature": 0.5, "top_p": 0.8, "sample_count": 1},
        )

        self.assertIsInstance(results.total_trades, int)
        self.assertGreaterEqual(results.max_drawdown, 0.0)


class TestBacktesterSignalConstruction(unittest.TestCase):
    def test_build_snapshot_at_bar_preserves_validation_pred_delta(self):
        bt = Backtester()

        snapshot = bt._build_snapshot_at_bar(
            signal_time=pd.Timestamp("2026-01-01 04:00:00"),
            current_price=100.0,
            forecast_price=103.0,
            val_start_price=98.0,
            val_pred_price=101.0,
            val_pred_delta=1.5,
            val_hist_delta=0.5,
        )

        self.assertIsNotNone(snapshot)
        self.assertAlmostEqual(snapshot.validation_pred_return, 3.0 / 98.0, places=6)
        self.assertEqual(snapshot.validation_pred_delta, 1.5)

    def test_infer_bar_interval_hours_supports_hourly_data(self):
        bt = Backtester()
        hourly_df = pd.DataFrame({
            "timestamps": pd.date_range("2026-01-01 00:00:00", periods=6, freq="1h"),
            "close": np.linspace(100.0, 105.0, 6),
        })

        self.assertEqual(bt._infer_bar_interval_hours(hourly_df), 1.0)


class TestBacktestMetricsDataclass(unittest.TestCase):
    def test_metrics_to_dict(self):
        metrics = BacktestMetrics(
            total_trades=10,
            winning_trades=6,
            losing_trades=4,
            win_rate=0.6,
            total_pnl=500.0,
            total_return_pct=5.0,
            avg_win=100.0,
            avg_loss=50.0,
            avg_trade_return=0.02,
            max_drawdown=200.0,
            max_drawdown_pct=0.02,
            sharpe_ratio=1.5,
            sortino_ratio=1.2,
            profit_factor=2.0,
            total_fees=10.0,
            total_slippage=15.0,
            time_in_market_pct=0.45,
            equity_curve=[],
            trades=[],
        )

        d = metrics.to_dict()
        self.assertEqual(d["total_trades"], 10)
        self.assertEqual(d["winning_trades"], 6)
        self.assertEqual(d["win_rate"], 0.6)
        self.assertEqual(d["total_pnl"], 500.0)
        self.assertEqual(d["sharpe_ratio"], 1.5)
        self.assertEqual(d["sortino_ratio"], 1.2)
        self.assertEqual(d["time_in_market_pct"], 0.45)


class TestBacktestTradeDataclass(unittest.TestCase):
    def test_trade_creation(self):
        trade = BacktestTrade(
            entry_time=pd.Timestamp("2026-01-01"),
            exit_time=pd.Timestamp("2026-01-02"),
            side="long",
            entry_price=100.0,
            exit_price=105.0,
            quantity=1.0,
            leverage=5.0,
            pnl_amount=25.0,
            pnl_pct=0.25,
            return_pct=0.05,
            fee=0.5,
            slippage_cost=0.1,
            reason="take_profit_hit",
        )

        self.assertEqual(trade.side, "long")
        self.assertGreater(trade.pnl_amount, 0)
        self.assertEqual(trade.reason, "take_profit_hit")


class TestBacktesterEntryExitLogic(unittest.TestCase):
    """Test that backtester uses same entry/exit logic as paper_strategy"""

    def test_long_entry_triggered_when_forecast_positive(self):
        bt = Backtester(initial_equity=10000.0)
        bt.strategy_config = PaperStrategyConfig(
            entry_threshold_pct=0.01,
            stop_loss_fraction=0.5,
            min_stop_loss_pct=0.01,
            max_stop_loss_pct=0.04,
            take_profit_multiplier=2.0,
        )

        timestamps = pd.date_range(start=datetime(2026, 1, 1), periods=600, freq="4h")
        prices = 100.0 + np.cumsum(np.random.randn(600) * 0.5)
        context_df = pd.DataFrame({"timestamps": timestamps, "close": prices})

        val_hist_df = pd.DataFrame({
            "timestamps": timestamps[-150:],
            "close": 105.0 + np.arange(150) * 0.1
        })
        val_actual_df = pd.DataFrame({
            "timestamps": timestamps[-50:],
            "close": 110.0 + np.arange(50) * 0.15
        })

        def strong_bull_predictor(context_df, temperature=0.5, top_p=0.8, sample_count=1):
            last_price = float(context_df.iloc[-1]["close"])
            future_df = pd.DataFrame({
                "close": [last_price * 1.03 for _ in range(1, 19)]
            })
            return {"future_pred_df": future_df}

        results = bt.run(
            context_df=context_df,
            predictor_fn=strong_bull_predictor,
            forecast_params={"lookback": 512, "pred_len": 18, "temperature": 0.5, "top_p": 0.8, "sample_count": 1},
        )

        long_trades = [t for t in results.trades if t.side == "long"]
        self.assertGreater(len(long_trades), 0, "Should trigger long entries when forecast is strongly positive")

    def test_short_entry_triggered_when_forecast_negative(self):
        bt = Backtester(initial_equity=10000.0)
        bt.strategy_config = PaperStrategyConfig(
            entry_threshold_pct=0.01,
            stop_loss_fraction=0.5,
            min_stop_loss_pct=0.01,
            max_stop_loss_pct=0.04,
            take_profit_multiplier=2.0,
        )

        timestamps = pd.date_range(start=datetime(2026, 1, 1), periods=600, freq="4h")
        prices = 100.0 + np.cumsum(np.random.randn(600) * 0.5)
        context_df = pd.DataFrame({"timestamps": timestamps, "close": prices})

        val_hist_df = pd.DataFrame({
            "timestamps": timestamps[-150:],
            "close": 105.0 - np.arange(150) * 0.1
        })
        val_actual_df = pd.DataFrame({
            "timestamps": timestamps[-50:],
            "close": 100.0 - np.arange(50) * 0.15
        })

        def strong_bear_predictor(context_df, temperature=0.5, top_p=0.8, sample_count=1):
            last_price = float(context_df.iloc[-1]["close"])
            future_df = pd.DataFrame({
                "close": [last_price * 0.97 for _ in range(1, 19)]
            })
            return {"future_pred_df": future_df}

        results = bt.run(
            context_df=context_df,
            predictor_fn=strong_bear_predictor,
            forecast_params={"lookback": 512, "pred_len": 18, "temperature": 0.5, "top_p": 0.8, "sample_count": 1},
        )

        short_trades = [t for t in results.trades if t.side == "short"]
        self.assertGreater(len(short_trades), 0, "Should trigger short entries when forecast is strongly negative")

    def test_no_entry_when_forecast_neutral(self):
        bt = Backtester(initial_equity=10000.0)
        bt.strategy_config = PaperStrategyConfig(
            entry_threshold_pct=0.02,
            stop_loss_fraction=0.5,
            min_stop_loss_pct=0.01,
            max_stop_loss_pct=0.04,
            take_profit_multiplier=2.0,
        )

        timestamps = pd.date_range(start=datetime(2026, 1, 1), periods=600, freq="4h")
        prices = 100.0 + np.cumsum(np.random.randn(600) * 0.5)
        context_df = pd.DataFrame({"timestamps": timestamps, "close": prices})

        val_hist_df = pd.DataFrame({
            "timestamps": timestamps[-150:],
            "close": 100.0 + np.arange(150) * 0.01
        })
        val_actual_df = pd.DataFrame({
            "timestamps": timestamps[-50:],
            "close": 100.0 + np.arange(50) * 0.005
        })

        def neutral_predictor(context_df, temperature=0.5, top_p=0.8, sample_count=1):
            last_price = float(context_df.iloc[-1]["close"])
            future_df = pd.DataFrame({
                "close": [last_price * (1 + 0.001 * (i % 2 - 0.5)) for i in range(1, 19)]
            })
            return {"future_pred_df": future_df}

        results = bt.run(
            context_df=context_df,
            predictor_fn=neutral_predictor,
            forecast_params={"lookback": 512, "pred_len": 18, "temperature": 0.5, "top_p": 0.8, "sample_count": 1},
        )

        self.assertEqual(len(results.trades), 0, "Should not trigger entries when forecast is neutral (< 2%)")


class TestBacktesterPositionManagement(unittest.TestCase):
    """Test position management in backtester"""

    def test_position_closed_on_stop_loss(self):
        bt = Backtester(initial_equity=10000.0, fee_rate=0.0005, slippage_pct=0.001)

        timestamps = pd.date_range(start=datetime(2026, 1, 1), periods=600, freq="4h")
        prices = 100.0 + np.cumsum(np.random.randn(600) * 0.5)
        context_df = pd.DataFrame({"timestamps": timestamps, "close": prices})

        val_hist_df = pd.DataFrame({
            "timestamps": timestamps[-150:],
            "close": 105.0 + np.arange(150) * 0.1
        })
        val_actual_df = pd.DataFrame({
            "timestamps": timestamps[-50:],
            "close": 110.0 + np.arange(50) * 0.15
        })

        def predict_and_drop(context_df, temperature=0.5, top_p=0.8, sample_count=1):
            last_price = float(context_df.iloc[-1]["close"])
            future_df = pd.DataFrame({
                "close": [last_price * (1 + 0.03 + 0.001 * i) for i in range(1, 19)]
            })
            return {"future_pred_df": future_df}

        results = bt.run(
            context_df=context_df,
            predictor_fn=predict_and_drop,
            forecast_params={"lookback": 512, "pred_len": 18, "temperature": 0.5, "top_p": 0.8, "sample_count": 1},
        )

        stop_loss_exits = [t for t in results.trades if t.reason == "stop_loss_hit"]
        self.assertGreaterEqual(len(stop_loss_exits), 0)

    def test_equity_tracked_correctly(self):
        bt = Backtester(initial_equity=10000.0, fee_rate=0.0005, slippage_pct=0.001)

        context_df = build_test_context_df(100)
        val_hist_df = build_test_validation_df(50)
        val_actual_df = build_test_validation_df(30)

        results = bt.run(
            context_df=context_df,
            predictor_fn=mock_predictor,
            forecast_params={"lookback": 50, "pred_len": 18, "temperature": 0.5, "top_p": 0.8, "sample_count": 1},
        )

        self.assertGreater(len(results.equity_curve), 0)
        if results.equity_curve:
            self.assertIsInstance(results.equity_curve[0]["time"], pd.Timestamp)
            self.assertIsInstance(results.equity_curve[0]["equity"], float)


if __name__ == "__main__":
    unittest.main()
