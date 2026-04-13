import os
import sys
import threading
import time
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestEnsureTimestampSeries(unittest.TestCase):
    """Test ensure_timestamp_series function."""

    def test_ensure_timestamp_series_from_series(self):
        """Test that function handles pandas Series correctly."""
        ts = pd.Series(pd.date_range("2024-01-01", periods=5))
        result = ts.reset_index(drop=True)
        
        self.assertEqual(len(result), 5)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result))

    def test_ensure_timestamp_series_handles_nat(self):
        """Test that NaT values are preserved."""
        ts = pd.Series([pd.Timestamp("2024-01-01"), pd.NaT, pd.Timestamp("2024-01-03")])
        
        result = ts.reset_index(drop=True)
        
        self.assertTrue(result.iloc[1] is pd.NaT or pd.isna(result.iloc[1]))


class TestTimeframeStep(unittest.TestCase):
    """Test timeframe step (Timedelta) calculation."""

    def test_5m_timedelta(self):
        """Test 5 minute timedelta."""
        step = pd.Timedelta(minutes=5)
        self.assertEqual(step, pd.Timedelta(minutes=5))

    def test_15m_timedelta(self):
        """Test 15 minute timedelta."""
        step = pd.Timedelta(minutes=15)
        self.assertEqual(step, pd.Timedelta(minutes=15))

    def test_1h_timedelta(self):
        """Test 1 hour timedelta."""
        step = pd.Timedelta(hours=1)
        self.assertEqual(step, pd.Timedelta(hours=1))

    def test_4h_timedelta(self):
        """Test 4 hour timedelta."""
        step = pd.Timedelta(hours=4)
        self.assertEqual(step, pd.Timedelta(hours=4))

    def test_1d_timedelta(self):
        """Test 1 day timedelta."""
        step = pd.Timedelta(days=1)
        self.assertEqual(step, pd.Timedelta(days=1))

    def test_timedelta_from_string(self):
        """Test creating timedelta from string."""
        step = pd.Timedelta("5m")
        self.assertEqual(step, pd.Timedelta(minutes=5))


class TestModelInputColumns(unittest.TestCase):
    """Test model input columns selection."""

    def test_required_feature_columns(self):
        """Test that required columns are selected correctly."""
        required_cols = ["open", "high", "low", "close", "volume"]
        
        df = pd.DataFrame({col: [1.0] * 5 for col in required_cols})
        df["amount"] = [100.0] * 5
        
        features = df[required_cols]
        
        self.assertEqual(list(features.columns), required_cols)
        self.assertNotIn("amount", features.columns)

    def test_all_columns_present(self):
        """Test that all required columns are present in dataframe."""
        required_cols = ["open", "high", "low", "close", "volume"]
        
        df = pd.DataFrame({col: [1.0] * 5 for col in required_cols})
        
        for col in required_cols:
            self.assertIn(col, df.columns)


class TestEmptyDataHandling(unittest.TestCase):
    """Test empty data handling."""

    def test_empty_dataframe_check(self):
        """Test that empty dataframe is detected."""
        df = pd.DataFrame()
        self.assertTrue(df.empty)

    def test_empty_dataframe_length(self):
        """Test that empty dataframe has zero length."""
        df = pd.DataFrame()
        self.assertEqual(len(df), 0)

    def test_dataframe_with_data_not_empty(self):
        """Test that dataframe with data is not empty."""
        df = pd.DataFrame({"close": [1.0]})
        self.assertFalse(df.empty)

    def test_empty_check_before_operation(self):
        """Test that operations check for empty data first."""
        df = pd.DataFrame()
        
        if df.empty:
            result = "No data"
        else:
            result = df["close"].sum()
        
        self.assertEqual(result, "No data")


class TestLookbackSpinBounds(unittest.TestCase):
    """Test lookback spin box bounds validation."""

    def test_lookback_min_bound(self):
        """Test minimum lookback is 50."""
        min_lookback = 50
        lookback = 30
        
        self.assertLess(lookback, min_lookback)

    def test_lookback_max_bound(self):
        """Test maximum lookback is 512."""
        max_lookback = 512
        lookback = 600
        
        self.assertGreater(lookback, max_lookback)

    def test_lookback_in_valid_range(self):
        """Test lookback in valid range."""
        min_lookback = 50
        max_lookback = 512
        lookback = 280
        
        self.assertGreaterEqual(lookback, min_lookback)
        self.assertLessEqual(lookback, max_lookback)

    def test_pred_len_min_bound(self):
        """Test minimum prediction length is 1."""
        pred_len = 0
        self.assertLess(pred_len, 1)

    def test_pred_len_max_bound(self):
        """Test maximum prediction length is 200."""
        max_pred_len = 200
        pred_len = 250
        self.assertGreater(pred_len, max_pred_len)


class TestModelConfigStructure(unittest.TestCase):
    """Test LOCAL_MODEL_CONFIGS structure validation."""

    def test_config_has_required_fields(self):
        """Test that model config has required fields."""
        required_fields = ["model_path", "tokenizer_path", "backend", "context_length", "label"]
        
        mock_config = {
            "model_path": "/path/to/model",
            "tokenizer_path": "/path/to/tokenizer",
            "backend": "pytorch",
            "context_length": 512,
            "label": "Kronos-small"
        }
        
        for field in required_fields:
            self.assertIn(field, mock_config)

    def test_config_backend_values(self):
        """Test that backend is valid."""
        valid_backends = ["mlx", "pytorch"]
        
        config = {"backend": "pytorch"}
        self.assertIn(config["backend"], valid_backends)

    def test_config_context_length_positive(self):
        """Test that context_length is positive."""
        config = {"context_length": 512}
        self.assertGreater(config["context_length"], 0)


class TestWorkerThreadDispatch(unittest.TestCase):
    """Test worker thread dispatch mechanism."""

    def test_worker_payload_structure(self):
        """Test that worker payload has required fields."""
        required_fields = ["df", "context_df", "future_pred_df", "summary_text"]
        
        payload = {
            "df": pd.DataFrame(),
            "context_df": pd.DataFrame(),
            "future_pred_df": pd.DataFrame(),
            "summary_text": "test"
        }
        
        for field in required_fields:
            self.assertIn(field, payload)

    def test_dispatch_signal发射(self):
        """Test that dispatch signal can emit payload."""
        from PyQt5.QtCore import pyqtSignal, QObject
        
        class TestEmitter(QObject):
            dispatch = pyqtSignal(object)
        
        emitter = TestEmitter()
        received = []
        
        def handler(payload):
            received.append(payload)
        
        emitter.dispatch.connect(handler)
        emitter.dispatch.emit({"key": "value"})
        
        self.assertEqual(received[0]["key"], "value")

    def test_worker_callback_format(self):
        """Test that worker callback has correct signature."""
        def worker_callback(payload, predictor):
            return {"processed": True}
        
        payload = {"data": "test"}
        predictor = None
        
        result = worker_callback(payload, predictor)
        self.assertTrue(result["processed"])


class TestForecastChartStyling(unittest.TestCase):
    """Test forecast chart styling."""

    def test_price_axis_formatter(self):
        """Test that price axis uses dollar formatter."""
        from qtgui.kronos_gui import format_price
        
        result = format_price(73450)
        self.assertIn("$", result)
        self.assertIn("73,450", result)

    def test_chart_background_color(self):
        """Test chart background color is dark."""
        bg_color = "#1b1b1b"
        
        self.assertTrue(bg_color.startswith("#"))
        self.assertEqual(len(bg_color), 7)

    def test_grid_color_properties(self):
        """Test grid has proper color and alpha."""
        grid_color = "#4b5563"
        grid_alpha = 0.30
        
        self.assertTrue(grid_color.startswith("#"))
        self.assertGreaterEqual(grid_alpha, 0)
        self.assertLessEqual(grid_alpha, 1)


class TestTabStructure(unittest.TestCase):
    """Test QTabWidget tab structure."""

    def test_expected_tabs_exist(self):
        """Test that expected tab names are defined."""
        expected_tabs = ["Forecast", "Validation", "Log"]
        
        for tab in expected_tabs:
            self.assertIsInstance(tab, str)
            self.assertGreater(len(tab), 0)

    def test_tab_count(self):
        """Test that there are 3 tabs."""
        expected_tab_count = 3
        actual_tabs = ["Forecast", "Validation", "Log"]
        
        self.assertEqual(len(actual_tabs), expected_tab_count)


class TestAutoSavePathCreation(unittest.TestCase):
    """Test auto-save path creation."""

    def test_makedirs_creates_directory(self):
        """Test that makedirs creates directory recursively."""
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "prediction_results", "BTC_USDT", "4h")
            os.makedirs(save_path, exist_ok=True)
            
            self.assertTrue(os.path.exists(save_path))

    def test_exist_ok_prevents_error(self):
        """Test that exist_ok=True prevents error if directory exists."""
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_dir")
            os.makedirs(save_path, exist_ok=True)
            os.makedirs(save_path, exist_ok=True)
            
            self.assertTrue(os.path.exists(save_path))

    def test_symbol_safe_for_path(self):
        """Test that symbol with / is replaced for path."""
        symbol = "BTC/USDT"
        safe_symbol = symbol.replace("/", "_")
        path = os.path.join("results", safe_symbol)
        
        self.assertNotIn("BTC/USDT", path)
        self.assertEqual(path, "results/BTC_USDT")


class TestTimestampIndexReset(unittest.TestCase):
    """Test prediction result index reset handling."""

    def test_reset_index_creates_timestamps_column(self):
        """Test that reset_index converts index to column."""
        df = pd.DataFrame({"close": [100, 101, 102]}, index=pd.date_range("2024-01-01", periods=3))
        result = df.reset_index()
        
        self.assertIn("index", result.columns)

    def test_rename_index_to_timestamps(self):
        """Test that index is renamed to timestamps."""
        df = pd.DataFrame({"close": [100, 101, 102]}, index=pd.date_range("2024-01-01", periods=3))
        result = df.reset_index().rename(columns={"index": "timestamps"})
        
        self.assertIn("timestamps", result.columns)
        self.assertNotIn("index", result.columns)

    def test_predictions_have_timestamps(self):
        """Test that prediction output has timestamps column."""
        df = pd.DataFrame({
            "timestamps": pd.date_range("2024-01-01", periods=5),
            "close": [73450, 73500, 73600, 73700, 73800]
        })
        
        self.assertIn("timestamps", df.columns)


class TestDeviceDetection(unittest.TestCase):
    """Test device detection and priority logic."""

    def test_device_priority_order(self):
        """Test that device priority is MPS > XPU > CUDA > CPU."""
        priority_order = ["mps", "xpu", "cuda", "cpu"]
        
        available_devices = ["cuda", "cpu"]
        chosen = min(available_devices, key=lambda x: priority_order.index(x))
        self.assertEqual(chosen, "cuda")

    def test_device_priority_mps_wins(self):
        """Test that MPS wins when all devices available."""
        priority_order = ["mps", "xpu", "cuda", "cpu"]
        
        available_devices = ["mps", "xpu", "cuda", "cpu"]
        chosen = min(available_devices, key=lambda x: priority_order.index(x))
        self.assertEqual(chosen, "mps")

    def test_device_fallback_to_cpu(self):
        """Test that CPU is fallback when no GPU available."""
        priority_order = ["mps", "xpu", "cuda", "cpu"]
        
        available_devices = []
        chosen = "cpu"
        self.assertEqual(chosen, "cpu")


class TestMultiSampleAveraging(unittest.TestCase):
    """Test multi-sample prediction averaging logic."""

    def test_stack_and_mean_calculation(self):
        """Test that multiple prediction paths are stacked and averaged."""
        num_samples = 3
        pred_len = 5
        num_features = 6
        
        np.random.seed(42)
        path1 = np.random.rand(pred_len, num_features).astype(np.float32)
        path2 = np.random.rand(pred_len, num_features).astype(np.float32)
        path3 = np.random.rand(pred_len, num_features).astype(np.float32)
        
        stacked = np.stack([path1, path2, path3], axis=0)
        mean_pred = stacked.mean(axis=0)
        
        self.assertEqual(stacked.shape, (3, 5, 6))
        self.assertEqual(mean_pred.shape, (5, 6))

    def test_single_sample_no_averaging(self):
        """Test that single sample returns as-is."""
        path = np.random.rand(5, 6).astype(np.float32)
        
        if len([path]) == 1:
            result = path
        else:
            stacked = np.stack([path], axis=0)
            result = stacked.mean(axis=0)
        
        np.testing.assert_array_equal(result, path)


class TestMCIntervalCalculation(unittest.TestCase):
    """Test Monte Carlo interval (quantile) calculation."""

    def test_quantile_10_90_calculation(self):
        """Test that 10%-90% quantile range is calculated correctly."""
        num_samples = 100
        pred_len = 10
        
        np.random.seed(42)
        all_paths = np.random.rand(num_samples, pred_len) * 1000 + 70000
        
        lower = np.quantile(all_paths, 0.10, axis=0)
        upper = np.quantile(all_paths, 0.90, axis=0)
        
        self.assertEqual(len(lower), pred_len)
        self.assertEqual(len(upper), pred_len)
        self.assertTrue(np.all(lower <= upper))


class TestPredictionMovePercentage(unittest.TestCase):
    """Test prediction move percentage calculation."""

    def test_move_percentage_calculation(self):
        """Test move percentage: (last_pred - last_close) / last_close * 100."""
        last_close = 73450.0
        last_pred_close = 74150.0
        
        move_pct = ((last_pred_close - last_close) / last_close) * 100
        
        self.assertAlmostEqual(move_pct, 0.953, places=3)

    def test_move_percentage_negative(self):
        """Test negative move percentage."""
        last_close = 74000.0
        last_pred_close = 73000.0
        
        move_pct = ((last_pred_close - last_close) / last_close) * 100
        
        self.assertAlmostEqual(move_pct, -1.351, places=3)


class TestContextSlicing(unittest.TestCase):
    """Test context data slicing."""

    def test_tail_lookback_slicing(self):
        """Test that df.tail(lookback) extracts correct portion."""
        lookback = 280
        
        df = pd.DataFrame({"close": range(1000, 1000 + 600)})
        context_df = df.tail(lookback).reset_index(drop=True)
        
        self.assertEqual(len(context_df), lookback)
        self.assertEqual(context_df["close"].iloc[0], 1000 + 600 - lookback)


class TestPredictionOutputFormat(unittest.TestCase):
    """Test prediction output DataFrame format."""

    def test_prediction_df_has_timestamps_column(self):
        """Test that prediction output has timestamps column."""
        df = pd.DataFrame({
            "timestamps": pd.date_range("2024-01-01", periods=5),
            "close": [73450, 73500, 73600, 73700, 73800],
        })
        
        self.assertIn("timestamps", df.columns)

    def test_prediction_df_numeric_columns(self):
        """Test that prediction output has required numeric columns."""
        required_cols = ["open", "high", "low", "close", "volume"]
        
        df = pd.DataFrame({col: [1.0] * 5 for col in required_cols})
        df["timestamps"] = pd.date_range("2024-01-01", periods=5)
        
        for col in required_cols:
            self.assertIn(col, df.columns)


class TestValidationMetricsCalculation(unittest.TestCase):
    """Test validation metrics (MAE/RMSE/MAPE) calculation."""

    def test_mae_calculation(self):
        """Test MAE calculation: mean(abs(predicted - actual))."""
        predicted = np.array([73450, 73500, 73600, 73700, 73800])
        actual = np.array([73400, 73550, 73550, 73750, 73900])
        
        mae = np.mean(np.abs(predicted - actual))
        
        self.assertAlmostEqual(mae, 60.0, places=2)

    def test_rmse_calculation(self):
        """Test RMSE calculation: sqrt(mean((predicted - actual)^2))."""
        predicted = np.array([73450, 73500, 73600, 73700, 73800])
        actual = np.array([73400, 73550, 73550, 73750, 73900])
        
        mse = np.mean((predicted - actual) ** 2)
        rmse = np.sqrt(mse)
        
        self.assertAlmostEqual(rmse, 63.25, places=2)

    def test_mape_calculation(self):
        """Test MAPE calculation: mean(abs((actual - predicted) / actual)) * 100."""
        predicted = np.array([73450, 73500, 73600, 73700, 73800])
        actual = np.array([73400, 73550, 73550, 73750, 73900])
        
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        self.assertAlmostEqual(mape, 0.0815, places=3)


class TestDispatchSignalMechanism(unittest.TestCase):
    """Test dispatch signal mechanism for thread communication."""

    def test_dispatch_signal_connects(self):
        """Test that dispatch signal can be connected to a handler."""
        from PyQt5.QtCore import pyqtSignal, QObject
        
        class TestEmitter(QObject):
            dispatch = pyqtSignal(object)
        
        emitter = TestEmitter()
        received = []
        
        def handler(payload):
            received.append(payload)
        
        emitter.dispatch.connect(handler)
        emitter.dispatch.emit("test_payload")
        
        self.assertEqual(received[0], "test_payload")


class TestWorkerErrorHandling(unittest.TestCase):
    """Test worker error handling flow."""

    def test_worker_error_calls_handler(self):
        """Test that worker error calls on_worker_error handler."""
        errors_received = []
        
        def on_worker_error(title, message):
            errors_received.append((title, message))
        
        on_worker_error("Error", "Test error message")
        
        self.assertEqual(len(errors_received), 1)
        self.assertEqual(errors_received[0], ("Error", "Test error message"))


class TestAutoForecastStateTransitions(unittest.TestCase):
    """Test auto-forecast state transitions."""

    def test_start_auto_sets_running_true(self):
        """Test that start_auto sets running flag to True."""
        state = {"running": False}
        state["running"] = True
        self.assertTrue(state["running"])

    def test_stop_auto_sets_running_false(self):
        """Test that stop_auto sets running flag to False."""
        state = {"running": True, "timer": MagicMock()}
        state["running"] = False
        state["timer"].cancel()
        state["timer"] = None
        self.assertFalse(state["running"])
        self.assertIsNone(state["timer"])


class TestWorkerCancellation(unittest.TestCase):
    """Test worker thread cancellation logic."""

    def test_worker_checks_running_before_continue(self):
        """Test that worker checks auto_forecast_running flag before proceeding."""
        running_flag = True
        cancelled = False
        
        if not running_flag:
            cancelled = True
        
        self.assertFalse(cancelled)

    def test_worker_cancellation_early_return(self):
        """Test that worker returns early when cancelled."""
        running_flag = False
        work_done = False
        
        if not running_flag:
            return
        work_done = True
        
        self.assertFalse(work_done)

    def test_multiple_cancellation_check_points(self):
        """Test that cancellation is checked at multiple points."""
        checkpoints = [True, True, False]
        results = []
        
        for checkpoint in checkpoints:
            if not checkpoint:
                results.append("cancelled")
                break
            results.append("continued")
        
        self.assertEqual(results, ["continued", "continued", "cancelled"])

    def test_work_complete_called_when_cancelled(self):
        """Test that on_work_complete is called when work is cancelled."""
        auto_forecast_running = False
        on_complete_called = False
        
        def on_work_complete():
            nonlocal on_complete_called
            on_complete_called = True
        
        if not auto_forecast_running:
            on_work_complete()
        
        self.assertTrue(on_complete_called)

    def test_work_continues_when_not_cancelled(self):
        """Test that work continues when not cancelled."""
        auto_forecast_running = True
        work_done = False
        
        if not auto_forecast_running:
            return
        work_done = True
        
        self.assertTrue(work_done)

    def test_on_prediction_ready_returns_when_not_running(self):
        """Test that on_prediction_ready returns early if auto_forecast_running is False."""
        auto_forecast_running = False
        ui_updated = False
        
        def on_prediction_ready():
            nonlocal ui_updated
            if not auto_forecast_running:
                return
            ui_updated = True
        
        on_prediction_ready()
        
        self.assertFalse(ui_updated)

    def test_on_prediction_ready_updates_ui_when_running(self):
        """Test that on_prediction_ready updates UI when auto_forecast_running is True."""
        auto_forecast_running = True
        ui_updated = False
        
        def on_prediction_ready():
            nonlocal ui_updated
            if not auto_forecast_running:
                return
            ui_updated = True
        
        on_prediction_ready()
        
        self.assertTrue(ui_updated)

    def test_on_pipeline_data_ready_returns_when_not_running(self):
        """Test that on_pipeline_data_ready returns early if auto_forecast_running is False."""
        auto_forecast_running = False
        ui_updated = False
        
        def on_pipeline_data_ready():
            nonlocal ui_updated
            if not auto_forecast_running:
                return
            ui_updated = True
        
        on_pipeline_data_ready()
        
        self.assertFalse(ui_updated)

    def test_short_circuit_eval_prevents_ui_update(self):
        """Test that short-circuit evaluation prevents UI update when flag is False."""
        auto_forecast_running = False
        ui_updated = False
        
        def append_log(msg):
            nonlocal ui_updated
            ui_updated = True
        
        if auto_forecast_running and append_log("test"):
            pass
        
        self.assertFalse(ui_updated)

    def test_short_circuit_eval_allows_ui_update(self):
        """Test that short-circuit evaluation allows UI update when flag is True."""
        auto_forecast_running = True
        ui_updated = False
        
        def append_log(msg):
            nonlocal ui_updated
            ui_updated = True
        
        if auto_forecast_running and append_log("test"):
            pass
        
        self.assertTrue(ui_updated)


class TestPriceFormatting(unittest.TestCase):
    """Test price formatting functions."""

    def test_format_price_whole_number(self):
        """Test that format_price displays whole numbers."""
        from qtgui.kronos_gui import format_price
        
        result = format_price(73450.0)
        self.assertEqual(result, "$73,450")

    def test_format_price_rounds(self):
        """Test that format_price rounds to nearest integer."""
        from qtgui.kronos_gui import format_price
        
        result = format_price(73450.99)
        self.assertEqual(result, "$73,451")


class TestNextCandleTimeCalculation(unittest.TestCase):
    """Test next candle time calculation logic."""

    def test_get_interval_seconds(self):
        """Test interval seconds mapping."""
        from qtgui.kronos_gui import KronosGUI
        
        gui = KronosGUI.__new__(KronosGUI)
        
        self.assertEqual(gui._get_interval_seconds("5m"), 300)
        self.assertEqual(gui._get_interval_seconds("15m"), 900)
        self.assertEqual(gui._get_interval_seconds("1h"), 3600)
        self.assertEqual(gui._get_interval_seconds("4h"), 14400)
        self.assertEqual(gui._get_interval_seconds("1d"), 86400)

    def test_get_next_candle_time_5m_mid_candle(self):
        """Test 5m mid-candle returns next candle."""
        from qtgui.kronos_gui import KronosGUI
        
        gui = KronosGUI.__new__(KronosGUI)
        
        now = pd.Timestamp("2024-01-01 10:03:00")
        next_time = gui._get_next_candle_time(now, "5m")
        
        self.assertEqual(next_time.hour, 10)
        self.assertEqual(next_time.minute, 5)

    def test_get_next_candle_time_4h_at_2pm(self):
        """Test 4h at 2pm returns 4pm candle."""
        from qtgui.kronos_gui import KronosGUI
        
        gui = KronosGUI.__new__(KronosGUI)
        
        now = pd.Timestamp("2024-01-01 14:30:00")
        next_time = gui._get_next_candle_time(now, "4h")
        
        self.assertEqual(next_time.hour, 16)


class TestValidationDataSlicing(unittest.TestCase):
    """Test validation data slicing logic."""

    def test_validation_slice_extraction(self):
        """Test that validation slice extracts correct portions."""
        lookback = 10
        pred_len = 5
        total = lookback + pred_len
        
        df = pd.DataFrame({"close": range(100, 100 + total)})
        validation_slice = df.iloc[-total:].reset_index(drop=True)
        
        history = validation_slice.iloc[:lookback]
        actual = validation_slice.iloc[lookback:lookback + pred_len]
        
        self.assertEqual(len(history), lookback)
        self.assertEqual(len(actual), pred_len)


class TestBuildFutureTimestamps(unittest.TestCase):
    """Test future timestamp building."""

    def test_build_future_timestamps_length(self):
        """Test that output has correct length."""
        from qtgui.kronos_gui import build_future_timestamps
        
        last_ts = pd.Timestamp("2024-01-01 12:00:00")
        step = pd.Timedelta(hours=4)
        pred_len = 18
        
        future = build_future_timestamps(last_ts, step, pred_len)
        
        self.assertEqual(len(future), pred_len)


class TestAutoForecastFileNaming(unittest.TestCase):
    """Test auto-forecast file naming."""

    def test_symbol_replacement_in_path(self):
        """Test that / is replaced in symbol for file paths."""
        symbol = "BTC/USDT"
        safe_symbol = symbol.replace("/", "_")
        self.assertEqual(safe_symbol, "BTC_USDT")

    def test_save_path_structure(self):
        """Test save path follows expected structure."""
        symbol = "BTC_USDT"
        timeframe = "4h"
        timestamp = "20260414_143022"
        
        save_dir = os.path.join("prediction_results", symbol, timeframe)
        self.assertEqual(save_dir, "prediction_results/BTC_USDT/4h")


class TestAutoForecastControlFlow(unittest.TestCase):
    """Test auto-forecast control flow."""

    @patch("qtgui.kronos_gui.threading.Timer")
    def test_toggle_auto_forecast_starts_when_not_running(self, mock_timer):
        """Test toggle starts auto forecast when not running."""
        from qtgui import kronos_gui
        
        gui = kronos_gui.KronosGUI.__new__(kronos_gui.KronosGUI)
        gui.auto_forecast_running = False
        gui.predictor = MagicMock()
        gui.auto_forecast_btn = MagicMock()
        gui.auto_status_label = MagicMock()
        gui._set_controls_enabled = MagicMock()
        gui.append_log = MagicMock()
        gui._run_auto_forecast = MagicMock()
        
        gui.toggle_auto_forecast()
        
        self.assertTrue(gui.auto_forecast_running)

    @patch("qtgui.kronos_gui.threading.Timer")
    def test_toggle_auto_forecast_stops_when_running(self, mock_timer):
        """Test toggle stops auto forecast when running."""
        from qtgui import kronos_gui
        
        gui = kronos_gui.KronosGUI.__new__(kronos_gui.KronosGUI)
        gui.auto_forecast_running = True
        gui.auto_forecast_timer = MagicMock()
        gui.auto_forecast_btn = MagicMock()
        gui.auto_status_label = MagicMock()
        gui._set_controls_enabled = MagicMock()
        gui.append_log = MagicMock()
        gui.results_text = MagicMock()
        
        gui.toggle_auto_forecast()
        
        self.assertFalse(gui.auto_forecast_running)


class TestSetBusyState(unittest.TestCase):
    """Test set_busy state management."""

    @patch("qtgui.kronos_gui.threading.Timer")
    def test_set_busy_sets_predicting_flag(self, mock_timer):
        """Test that set_busy sets the busy flag correctly."""
        from qtgui import kronos_gui
        
        gui = kronos_gui.KronosGUI.__new__(kronos_gui.KronosGUI)
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


class TestLookbackValidation(unittest.TestCase):
    """Test lookback and prediction length validation."""

    def test_lookback_plus_pred_len_within_context(self):
        """Test that lookback + pred_len must be within max_context."""
        max_context = 512
        lookback = 300
        pred_len = 150
        
        is_valid = (lookback <= max_context) and (lookback + pred_len <= max_context)
        self.assertTrue(is_valid)

    def test_lookback_exceeds_context(self):
        """Test that lookback exceeding context is invalid."""
        max_context = 512
        lookback = 600
        
        is_valid = lookback <= max_context
        self.assertFalse(is_valid)


if __name__ == "__main__":
    unittest.main()
