import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_sample_df(periods=24, start="2026-04-01 00:00:00", freq="4h"):
    timestamps = pd.date_range(start, periods=periods, freq=freq)
    base = np.arange(periods, dtype=float) + 70000.0
    return pd.DataFrame(
        {
            "timestamps": timestamps,
            "open": base,
            "high": base + 100,
            "low": base - 100,
            "close": base + 10,
            "volume": np.ones(periods),
            "amount": np.ones(periods) * 10,
        }
    )


def build_paper_entry_payload(
    current_price=100.0,
    forecast_price=104.0,
    validation_history=(95.0, 96.0, 98.0, 100.0),
    validation_pred=(101.0, 103.0),
    validation_actual=(100.5, 102.0),
):
    timestamps = pd.date_range("2026-04-10 00:00:00", periods=6, freq="4h")
    context_df = pd.DataFrame(
        {
            "timestamps": timestamps[:4],
            "close": [98.0, 99.0, 100.0, current_price],
        }
    )
    future_pred_df = pd.DataFrame(
        {
            "timestamps": timestamps[4:],
            "close": [102.0, forecast_price],
        }
    )
    validation_history_df = pd.DataFrame(
        {
            "timestamps": pd.date_range("2026-04-09 08:00:00", periods=4, freq="4h"),
            "close": list(validation_history),
        }
    )
    validation_pred_df = pd.DataFrame(
        {
            "timestamps": pd.date_range("2026-04-10 00:00:00", periods=2, freq="4h"),
            "close": list(validation_pred),
        }
    )
    validation_actual_df = pd.DataFrame(
        {
            "timestamps": pd.date_range("2026-04-10 00:00:00", periods=2, freq="4h"),
            "close": list(validation_actual),
        }
    )
    return {
        "summary_text": "done",
        "latest_close": f"${current_price}",
        "forecast_end": f"${forecast_price}",
        "forecast_move": f"+{(forecast_price/current_price - 1)*100:.2f}%",
        "mape_text": "1.25%",
        "subtitle": "Validation window: 2026-04-10 -> 2026-04-11",
        "df": pd.concat([validation_history_df, validation_actual_df], ignore_index=True),
        "backend_name": "Kronos-test",
        "symbol": "BTC/USDT",
        "timeframe": "4h",
        "context_df": context_df,
        "future_pred_df": future_pred_df,
        "forecast_interval_df": None,
        "validation_history_df": validation_history_df,
        "validation_pred_df": validation_pred_df,
        "validation_actual_df": validation_actual_df,
    }


def build_gui_stub_with_paper():
    from qtgui.kronos_gui import KronosGUI

    gui = KronosGUI.__new__(KronosGUI)
    gui.selected_model_key = "kronos-small-local"
    gui.auto_forecast_running = False
    gui.active_run_id = 1
    gui.active_run_mode = "manual"
    gui.results_text = MagicMock()
    gui.forecast_card = MagicMock()
    gui.move_card = MagicMock()
    gui.mape_card = MagicMock()
    gui.subtitle_label = MagicMock()
    gui.forecast_canvas = MagicMock()
    gui.validation_canvas = MagicMock()
    gui.status_bar = MagicMock()
    gui.update_badges = MagicMock()
    gui.set_busy = MagicMock()
    gui._save_forecast_results = MagicMock()
    gui._schedule_next_auto_forecast = MagicMock()
    gui.paper_trade_history = []
    gui.append_log = MagicMock()
    gui.show_error = MagicMock()
    gui.update_paper_mode = MagicMock()
    gui.paper_enabled = True
    gui.paper_latest_market_df = None
    gui.execution_mode = "paper"
    gui.paper_order_quantity = 0.01
    gui.paper_order_leverage = 5
    gui.paper_use_risk_fraction = True
    gui.paper_risk_fraction = 0.08
    gui.paper_initial_equity = 1000.0
    gui.paper_realized_equity = 1000.0
    gui.paper_equity_history = []
    gui.paper_position = None
    gui.paper_last_decision = None
    gui.paper_last_snapshot = None
    gui.paper_realized_pnl_pct = 0.0
    gui.paper_trade_history = []
    gui.paper_strategy_config = MagicMock()
    return gui


class TestRiskFractionCalculation(unittest.TestCase):
    """Test risk fraction position sizing logic."""

    def test_current_order_quantity_by_risk_handles_zero_stop_distance(self):
        from qtgui.kronos_gui import KronosGUI

        gui = KronosGUI.__new__(KronosGUI)
        gui.paper_risk_fraction = 0.08
        gui.paper_realized_equity = 1000.0
        gui.paper_initial_equity = 1000.0
        gui.paper_equity_history = []
        gui.paper_order_quantity = 0.01
        gui.paper_order_qty_spin = MagicMock()
        gui.paper_order_qty_spin.value.return_value = 0.01

        result = gui.current_order_quantity_by_risk(entry_price=100.0, stop_distance_pct=0.0)

        self.assertGreater(result, 0)
        self.assertEqual(result, 0.01)

    def test_current_order_quantity_by_risk_uses_risk_fraction_correctly(self):
        from qtgui.kronos_gui import KronosGUI

        gui = KronosGUI.__new__(KronosGUI)
        gui.paper_risk_fraction = 0.08
        gui.paper_realized_equity = 1000.0
        gui.paper_initial_equity = 1000.0
        gui.paper_equity_history = []
        gui.paper_order_quantity = 0.01
        gui.paper_order_qty_spin = MagicMock()
        gui.paper_order_qty_spin.value.return_value = 0.01

        entry_price = 100.0
        stop_distance_pct = 0.02
        expected_qty = (1000.0 * 0.08) / (100.0 * 0.02)

        result = gui.current_order_quantity_by_risk(entry_price, stop_distance_pct)

        self.assertAlmostEqual(result, expected_qty)

    def test_return_pct_calculation_with_zero_initial_equity_does_not_crash(self):
        from qtgui.kronos_gui import KronosGUI

        gui = KronosGUI.__new__(KronosGUI)
        gui.paper_realized_equity = 100.0
        gui.paper_initial_equity = 0.0

        result = gui.current_realized_return_pct()

        self.assertEqual(result, 0.0)


class TestPaperTradingLogic(unittest.TestCase):
    """Test paper trading entry/exit logic."""

    def test_paper_trading_active_requires_both_enabled_and_paper_mode(self):
        from qtgui.kronos_gui import KronosGUI

        gui = KronosGUI.__new__(KronosGUI)
        gui.paper_enabled = True
        gui.execution_mode = "paper"
        self.assertTrue(gui.paper_trading_active())

        gui.paper_enabled = False
        gui.execution_mode = "paper"
        self.assertFalse(gui.paper_trading_active())

        gui.paper_enabled = True
        gui.execution_mode = "testnet"
        self.assertFalse(gui.paper_trading_active())

    def test_current_paper_equity_with_unrealized_pnl(self):
        from qtgui.kronos_gui import KronosGUI

        gui = KronosGUI.__new__(KronosGUI)
        gui.paper_realized_equity = 1000.0

        self.assertEqual(gui.current_paper_equity(unrealized_amount=50.0), 1050.0)
        self.assertEqual(gui.current_paper_equity(unrealized_amount=-30.0), 970.0)

    def test_compute_position_return_pct_for_long_position(self):
        from qtgui.kronos_gui import KronosGUI
        from qtgui.paper_strategy import PaperPosition

        gui = KronosGUI.__new__(KronosGUI)

        position = PaperPosition(
            side="long",
            entry_price=100.0,
            stop_price=98.0,
            take_profit_price=104.0,
            entry_time=pd.Timestamp("2026-04-14 00:00:00"),
            quantity=1.0,
            leverage=1.0,
        )

        result = gui.compute_position_return_pct(position, current_price=102.0)
        self.assertAlmostEqual(result, 0.02)

    def test_compute_position_return_pct_for_short_position(self):
        from qtgui.kronos_gui import KronosGUI
        from qtgui.paper_strategy import PaperPosition

        gui = KronosGUI.__new__(KronosGUI)

        position = PaperPosition(
            side="short",
            entry_price=100.0,
            stop_price=102.0,
            take_profit_price=96.0,
            entry_time=pd.Timestamp("2026-04-14 00:00:00"),
            quantity=1.0,
            leverage=1.0,
        )

        result = gui.compute_position_return_pct(position, current_price=98.0)
        expected = (100.0 / 98.0) - 1.0
        self.assertAlmostEqual(result, expected)


class TestPaperStrategyEdgeCases(unittest.TestCase):
    """Test paper strategy edge cases."""

    def test_entry_decision_with_minimal_stop_distance(self):
        from qtgui.paper_strategy import (
            PaperStrategyConfig,
            build_entry_decision,
            build_signal_snapshot,
        )

        config = PaperStrategyConfig(
            entry_threshold_pct=0.02,
            stop_loss_fraction=0.5,
            min_stop_loss_pct=0.01,
            max_stop_loss_pct=0.04,
        )

        payload = build_paper_entry_payload(
            current_price=100.0,
            forecast_price=102.2,
            validation_history=(99.0, 99.5, 100.0, 100.5),
            validation_pred=(100.6, 100.8),
            validation_actual=(100.1, 100.2),
        )

        snapshot = build_signal_snapshot(payload)
        decision = build_entry_decision(snapshot, config)

        self.assertEqual(decision.action, "enter_long")
        self.assertIsNotNone(decision.stop_distance_pct)

    def test_entry_decision_with_max_stop_distance(self):
        from qtgui.paper_strategy import (
            PaperStrategyConfig,
            build_entry_decision,
            build_signal_snapshot,
        )

        config = PaperStrategyConfig(
            entry_threshold_pct=0.02,
            stop_loss_fraction=0.5,
            min_stop_loss_pct=0.01,
            max_stop_loss_pct=0.04,
        )

        payload = build_paper_entry_payload(
            current_price=100.0,
            forecast_price=110.0,
            validation_history=(95.0, 96.0, 98.0, 100.0),
            validation_pred=(101.0, 103.0),
            validation_actual=(100.5, 102.0),
        )

        snapshot = build_signal_snapshot(payload)
        decision = build_entry_decision(snapshot, config)

        self.assertEqual(decision.action, "enter_long")
        self.assertAlmostEqual(decision.stop_distance_pct, 0.04)


class TestGuiPaperTradingIntegration(unittest.TestCase):
    """Test GUI paper trading integration with proper Qt setup."""

    @classmethod
    def setUpClass(cls):
        try:
            from PyQt5.QtWidgets import QApplication
        except ImportError:
            from PyQt6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_paper_trading_defaults(self):
        from qtgui.kronos_gui import KronosGUI

        fake_model_configs = {
            "test-model": {
                "label": "Kronos-test",
                "backend": "pytorch",
                "context_length": 512,
                "model_path": "/tmp/kronos-model",
                "tokenizer_path": "/tmp/kronos-tokenizer",
                "description": "Test config",
            }
        }

        with patch.object(KronosGUI, "available_model_configs", return_value=fake_model_configs):
            window = KronosGUI()
            self.app.processEvents()

        try:
            self.assertTrue(window.paper_enabled)
            self.assertEqual(window.paper_initial_equity, 1000.0)
            self.assertEqual(window.paper_order_leverage, 5)
            self.assertTrue(window.paper_use_risk_fraction)
            self.assertEqual(window.paper_risk_fraction, 0.08)
        finally:
            window.close()
            self.app.processEvents()

    def test_paper_equity_card_exists_no_canvas(self):
        from qtgui.kronos_gui import KronosGUI

        fake_model_configs = {
            "test-model": {
                "label": "Kronos-test",
                "backend": "pytorch",
                "context_length": 512,
                "model_path": "/tmp/kronos-model",
                "tokenizer_path": "/tmp/kronos-tokenizer",
                "description": "Test config",
            }
        }

        with patch.object(KronosGUI, "available_model_configs", return_value=fake_model_configs):
            window = KronosGUI()
            self.app.processEvents()

        try:
            self.assertTrue(hasattr(window, "paper_equity_card"))
            self.assertFalse(hasattr(window, "paper_equity_canvas"))
        finally:
            window.close()
            self.app.processEvents()


class TestPaperTradingPnlEdgeCases(unittest.TestCase):
    """Test paper trading PnL edge cases."""

    def test_equity_can_become_zero(self):
        from qtgui.kronos_gui import KronosGUI

        gui = KronosGUI.__new__(KronosGUI)
        gui.paper_realized_equity = 0.0
        gui.paper_initial_equity = 1000.0
        gui.paper_equity_history = [{"equity": 100.0, "time": pd.Timestamp.now()}]

        equity = gui.current_paper_equity()
        self.assertEqual(equity, 0.0)

    def test_position_pnl_calculation_with_leverage(self):
        from qtgui.kronos_gui import KronosGUI
        from qtgui.paper_strategy import PaperPosition

        gui = KronosGUI.__new__(KronosGUI)

        position = PaperPosition(
            side="long",
            entry_price=100.0,
            stop_price=98.0,
            take_profit_price=104.0,
            entry_time=pd.Timestamp("2026-04-14 00:00:00"),
            quantity=2.0,
            leverage=5.0,
        )

        pnl = gui.compute_position_pnl_amount(position, current_price=102.0)
        expected_pnl = (102.0 - 100.0) * 2.0
        self.assertAlmostEqual(pnl, expected_pnl)


if __name__ == "__main__":
    unittest.main()
