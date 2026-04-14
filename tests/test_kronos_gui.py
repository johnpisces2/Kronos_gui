import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from PyQt5.QtCore import Qt
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    QT_LEFT_BUTTON = Qt.LeftButton
except ImportError:
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication

    QT_LEFT_BUTTON = Qt.MouseButton.LeftButton


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


def build_prediction_payload():
    df = build_sample_df(periods=6, start="2026-04-10 00:00:00")
    return {
        "summary_text": "done",
        "latest_close": "$70,000",
        "forecast_end": "$71,000",
        "forecast_move": "+1.43%",
        "mape_text": "1.25%",
        "subtitle": "Validation window: 2026-04-10 -> 2026-04-11",
        "df": df,
        "backend_name": "Kronos-test",
        "symbol": "BTC/USDT",
        "timeframe": "4h",
        "context_df": df.iloc[:4].copy(),
        "future_pred_df": df.iloc[4:].copy(),
        "forecast_interval_df": None,
        "validation_history_df": df.iloc[:4].copy(),
        "validation_pred_df": df.iloc[4:].copy(),
        "validation_actual_df": df.iloc[4:].copy(),
    }


def build_paper_entry_payload():
    timestamps = pd.date_range("2026-04-10 00:00:00", periods=6, freq="4h")
    context_df = pd.DataFrame(
        {
            "timestamps": timestamps[:4],
            "close": [98.0, 99.0, 100.0, 100.0],
        }
    )
    future_pred_df = pd.DataFrame(
        {
            "timestamps": timestamps[4:],
            "close": [102.0, 104.0],
        }
    )
    validation_history_df = pd.DataFrame(
        {
            "timestamps": pd.date_range("2026-04-09 08:00:00", periods=4, freq="4h"),
            "close": [95.0, 96.0, 98.0, 100.0],
        }
    )
    validation_pred_df = pd.DataFrame(
        {
            "timestamps": pd.date_range("2026-04-10 00:00:00", periods=2, freq="4h"),
            "close": [101.0, 103.0],
        }
    )
    validation_actual_df = pd.DataFrame(
        {
            "timestamps": pd.date_range("2026-04-10 00:00:00", periods=2, freq="4h"),
            "close": [100.5, 102.0],
        }
    )
    payload = build_prediction_payload()
    payload.update(
        {
            "df": pd.concat([validation_history_df, validation_actual_df], ignore_index=True),
            "context_df": context_df,
            "future_pred_df": future_pred_df,
            "validation_history_df": validation_history_df,
            "validation_pred_df": validation_pred_df,
            "validation_actual_df": validation_actual_df,
            "forecast_end": "$104",
            "forecast_move": "+4.00%",
        }
    )
    return payload


def build_gui_stub():
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
    gui.append_log = MagicMock()
    gui.show_error = MagicMock()
    gui.update_paper_mode = MagicMock()
    return gui


class TestTimeAndDataHelpers(unittest.TestCase):
    def test_utc_timestamp_series_converts_to_taipei_clock(self):
        from qtgui.kronos_gui import to_display_timestamp_series

        ts = pd.Series(pd.to_datetime(["2026-04-14 00:00:00"], utc=True))
        result = to_display_timestamp_series(ts)

        self.assertEqual(result.iloc[0], pd.Timestamp("2026-04-14 08:00:00"))
        self.assertIsNone(result.dt.tz)

    def test_format_display_timestamp_uses_taipei_clock(self):
        from qtgui.kronos_gui import format_display_timestamp

        result = format_display_timestamp(pd.Timestamp("2026-04-14 00:05:06", tz="UTC"))

        self.assertEqual(result, "20260414_080506")

    def test_build_future_timestamps_steps_forward(self):
        from qtgui.kronos_gui import build_future_timestamps

        future = build_future_timestamps(
            pd.Timestamp("2026-04-14 04:00:00"),
            pd.Timedelta(hours=4),
            3,
        )

        self.assertEqual(list(future), list(pd.to_datetime(
            ["2026-04-14 08:00:00", "2026-04-14 12:00:00", "2026-04-14 16:00:00"]
        )))

    def test_build_validation_segments_drops_latest_bar(self):
        from qtgui.kronos_gui import build_validation_segments

        df = build_sample_df(periods=24)
        history_df, actual_df = build_validation_segments(df, lookback=10, pred_len=4)

        self.assertEqual(history_df["timestamps"].iloc[0], df["timestamps"].iloc[-15])
        self.assertEqual(actual_df["timestamps"].iloc[-1], df["timestamps"].iloc[-2])

    def test_build_validation_segments_advances_with_new_candle(self):
        from qtgui.kronos_gui import build_validation_segments

        df = build_sample_df(periods=24)
        next_row = df.iloc[[-1]].copy()
        next_row["timestamps"] = next_row["timestamps"] + pd.Timedelta(hours=4)
        df_plus_one = pd.concat([df, next_row], ignore_index=True)

        history_df, actual_df = build_validation_segments(df, lookback=10, pred_len=4)
        history_df_next, actual_df_next = build_validation_segments(df_plus_one, lookback=10, pred_len=4)

        self.assertEqual(
            history_df_next["timestamps"].iloc[0] - history_df["timestamps"].iloc[0],
            pd.Timedelta(hours=4),
        )
        self.assertEqual(
            actual_df_next["timestamps"].iloc[0] - actual_df["timestamps"].iloc[0],
            pd.Timedelta(hours=4),
        )


class TestGuiControllerMethods(unittest.TestCase):
    def test_get_interval_seconds_mapping(self):
        from qtgui.kronos_gui import KronosGUI

        gui = KronosGUI.__new__(KronosGUI)

        self.assertEqual(gui._get_interval_seconds("5m"), 300)
        self.assertEqual(gui._get_interval_seconds("15m"), 900)
        self.assertEqual(gui._get_interval_seconds("1h"), 3600)
        self.assertEqual(gui._get_interval_seconds("4h"), 14400)
        self.assertEqual(gui._get_interval_seconds("1d"), 86400)

    def test_get_next_candle_time_4h_at_2pm(self):
        from qtgui.kronos_gui import KronosGUI

        gui = KronosGUI.__new__(KronosGUI)
        next_time = gui._get_next_candle_time(pd.Timestamp("2026-04-14 14:30:00"), "4h")

        self.assertEqual(next_time, pd.Timestamp("2026-04-14 16:00:00"))

    def test_set_busy_disables_controls_for_manual_run(self):
        from qtgui.kronos_gui import KronosGUI

        gui = KronosGUI.__new__(KronosGUI)
        gui.auto_forecast_running = False
        gui.busy = False
        gui.model_combo = MagicMock()
        gui.symbol_combo = MagicMock()
        gui.tf_combo = MagicMock()
        gui.limit_spin = MagicMock()
        gui.lookback_spin = MagicMock()
        gui.predlen_spin = MagicMock()
        gui.temp_spin = MagicMock()
        gui.topp_spin = MagicMock()
        gui.sample_spin = MagicMock()
        gui.predict_btn = MagicMock()
        gui.auto_forecast_btn = MagicMock()
        gui.progress = MagicMock()

        gui.set_busy(predicting=True)

        self.assertTrue(gui.busy)
        gui.model_combo.setEnabled.assert_called_with(False)
        gui.auto_forecast_btn.setEnabled.assert_called_with(False)
        gui.progress.setVisible.assert_called_with(True)

    def test_stop_auto_forecast_invalidates_active_run_and_resets_ui(self):
        from qtgui.kronos_gui import KronosGUI

        gui = KronosGUI.__new__(KronosGUI)
        gui.auto_forecast_running = True
        gui.active_run_id = 7
        gui.active_run_mode = "auto"
        timer = MagicMock()
        gui.auto_forecast_timer = timer
        gui.auto_forecast_btn = MagicMock()
        gui.auto_status_label = MagicMock()
        gui.status_bar = MagicMock()
        gui.update_badges = MagicMock()
        gui.append_log = MagicMock()
        gui._set_controls_enabled = MagicMock()
        gui.set_busy = MagicMock()

        gui.stop_auto_forecast()

        self.assertFalse(gui.auto_forecast_running)
        self.assertEqual(gui.active_run_id, 8)
        timer.cancel.assert_called_once()
        gui.auto_status_label.setText.assert_called_with("Auto Forecast: Stopped")
        gui.set_busy.assert_called_once()

    def test_on_prediction_ready_ignores_stale_auto_callback(self):
        gui = build_gui_stub()
        payload = build_prediction_payload()
        gui.active_run_id = 3

        gui.on_prediction_ready(payload, predictor=MagicMock(), run_id=2, auto_run=True)

        gui.results_text.setPlainText.assert_not_called()
        gui._save_forecast_results.assert_not_called()
        gui.set_busy.assert_not_called()

    def test_on_prediction_ready_updates_manual_run_without_auto_flag(self):
        gui = build_gui_stub()
        payload = build_prediction_payload()
        gui.active_run_id = 4

        gui.on_prediction_ready(payload, predictor=MagicMock(), run_id=4, auto_run=False)

        gui.results_text.setPlainText.assert_called_once_with("done")
        gui.forecast_canvas.plot_forecast.assert_called_once()
        gui.validation_canvas.plot_validation.assert_called_once()
        gui.update_paper_mode.assert_called_once_with(payload)
        gui._save_forecast_results.assert_not_called()
        gui.set_busy.assert_called_once()


class TestQtGuiIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from qtgui.kronos_gui import KronosGUI

        self.fake_model_configs = {
            "test-model": {
                "label": "Kronos-test",
                "backend": "pytorch",
                "context_length": 512,
                "model_path": "/tmp/kronos-model",
                "tokenizer_path": "/tmp/kronos-tokenizer",
                "description": "Test config for Qt integration.",
            }
        }
        self.model_patcher = patch.object(
            KronosGUI,
            "available_model_configs",
            return_value=self.fake_model_configs,
        )
        self.model_patcher.start()
        self.window = KronosGUI()
        self.app.processEvents()

    def tearDown(self):
        if hasattr(self, "window"):
            self.window.close()
            self.app.processEvents()
        if hasattr(self, "model_patcher"):
            self.model_patcher.stop()

    def test_window_builds_expected_tabs(self):
        tab_labels = [
            self.window.main_tabs.tabText(i)
            for i in range(self.window.main_tabs.count())
        ]

        self.assertEqual(tab_labels, ["Forecast", "Validation", "Paper Mode", "Log"])

    def test_auto_forecast_button_click_toggles_running_and_stopped_state(self):
        self.window._run_auto_forecast = MagicMock()

        QTest.mouseClick(self.window.auto_forecast_btn, QT_LEFT_BUTTON)
        self.app.processEvents()

        self.assertTrue(self.window.auto_forecast_running)
        self.assertEqual(self.window.auto_forecast_btn.text(), "Stop Auto Forecast")
        self.assertEqual(self.window.auto_status_label.text(), "Auto Forecast: Running")
        self.window._run_auto_forecast.assert_called_once()

        timer = MagicMock()
        self.window.auto_forecast_timer = timer

        QTest.mouseClick(self.window.auto_forecast_btn, QT_LEFT_BUTTON)
        self.app.processEvents()

        self.assertFalse(self.window.auto_forecast_running)
        self.assertEqual(self.window.auto_forecast_btn.text(), "Run Auto Forecast")
        self.assertEqual(self.window.auto_status_label.text(), "Auto Forecast: Stopped")
        self.assertEqual(self.window.statusBar().currentMessage(), "Auto forecast stopped")
        timer.cancel.assert_called_once()

    def test_stale_auto_callback_after_stop_does_not_mutate_live_widgets(self):
        self.window._run_auto_forecast = MagicMock()

        QTest.mouseClick(self.window.auto_forecast_btn, QT_LEFT_BUTTON)
        self.app.processEvents()

        stale_run_id = self.window.active_run_id
        self.window.auto_forecast_timer = MagicMock()

        QTest.mouseClick(self.window.auto_forecast_btn, QT_LEFT_BUTTON)
        self.app.processEvents()

        payload = build_prediction_payload()
        before_text = self.window.results_text.toPlainText()
        before_status = self.window.auto_status_label.text()
        before_message = self.window.statusBar().currentMessage()

        self.window.on_prediction_ready(payload, predictor=MagicMock(), run_id=stale_run_id, auto_run=True)
        self.app.processEvents()

        self.assertEqual(self.window.results_text.toPlainText(), before_text)
        self.assertEqual(self.window.auto_status_label.text(), before_status)
        self.assertEqual(self.window.statusBar().currentMessage(), before_message)
        self.assertNotEqual(self.window.results_text.toPlainText(), payload["summary_text"])

    def test_forecast_and_validation_charts_show_crosshair_time_labels(self):
        payload = build_prediction_payload()

        self.window.forecast_canvas.plot_forecast(
            symbol=payload["symbol"],
            timeframe=payload["timeframe"],
            context_df=payload["context_df"],
            future_pred_df=payload["future_pred_df"],
            forecast_interval_df=payload["forecast_interval_df"],
        )
        self.window.validation_canvas.plot_validation(
            validation_history_df=payload["validation_history_df"],
            validation_pred_df=payload["validation_pred_df"],
            validation_actual_df=payload["validation_actual_df"],
        )
        self.app.processEvents()

        forecast_texts = [text.get_text() for text in self.window.forecast_canvas.price_ax.texts]
        validation_texts = [text.get_text() for text in self.window.validation_canvas.validation_ax.texts]

        self.assertIn("04-10 12:00", forecast_texts)
        self.assertIn("04-10 12:00", validation_texts)

    def test_prediction_ready_updates_paper_mode_tab(self):
        payload = build_paper_entry_payload()
        self.window.active_run_id = 1

        self.window.on_prediction_ready(payload, predictor=MagicMock(), run_id=1, auto_run=False)
        self.app.processEvents()

        self.assertEqual(self.window.paper_position.side, "long")
        self.assertIn("ENTER LONG", self.window.paper_signal_card.value_label.text())
        self.assertIn("LONG @", self.window.paper_position_card.value_label.text())
        self.assertIn("ENTER LONG", self.window.paper_trade_log.toPlainText())
        self.assertEqual(self.window.paper_trade_table.rowCount(), 1)
        self.assertEqual(self.window.paper_position_table.item(0, 0).text(), "LONG")
        self.assertGreaterEqual(len(self.window.paper_equity_history), 1)
        self.assertGreaterEqual(len(self.window.paper_trade_chart_canvas.trade_ax.collections), 1)

    def test_disable_paper_trading_blocks_new_local_entries(self):
        payload = build_paper_entry_payload()

        self.window.paper_enable_checkbox.setChecked(False)
        self.app.processEvents()
        self.window.active_run_id = 1
        self.window.on_prediction_ready(payload, predictor=MagicMock(), run_id=1, auto_run=False)
        self.app.processEvents()

        self.assertIsNone(self.window.paper_position)
        self.assertEqual(self.window.paper_trade_table.rowCount(), 0)
        self.assertEqual(self.window.paper_signal_card.value_label.text(), "DISABLED")

    def test_reset_paper_account_clears_tables_and_history(self):
        payload = build_paper_entry_payload()
        self.window.active_run_id = 1
        self.window.on_prediction_ready(payload, predictor=MagicMock(), run_id=1, auto_run=False)
        self.app.processEvents()

        QTest.mouseClick(self.window.paper_reset_btn, QT_LEFT_BUTTON)
        self.app.processEvents()

        self.assertIsNone(self.window.paper_position)
        self.assertEqual(self.window.paper_trade_table.rowCount(), 0)
        self.assertEqual(len(self.window.paper_trade_history), 0)
        self.assertEqual(len(self.window.paper_equity_history), 0)
        self.assertEqual(self.window.paper_position_table.item(0, 0).text(), "--")

    def test_changing_initial_equity_resets_paper_account(self):
        payload = build_paper_entry_payload()
        self.window.active_run_id = 1
        self.window.on_prediction_ready(payload, predictor=MagicMock(), run_id=1, auto_run=False)
        self.app.processEvents()

        self.window.paper_initial_equity_spin.setValue(250000.0)
        self.app.processEvents()

        self.assertEqual(self.window.paper_initial_equity, 250000.0)
        self.assertIsNone(self.window.paper_position)
        self.assertEqual(self.window.paper_trade_table.rowCount(), 0)
        self.assertIn("250,000", self.window.paper_status_label.text())


if __name__ == "__main__":
    unittest.main()
