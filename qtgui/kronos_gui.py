#!/usr/bin/env python3
"""Kronos BTC prediction GUI for Binance futures data."""

import os
import sys
import threading

import ccxt
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if sys.platform == "darwin":
    os.environ.setdefault("QT_MAC_WANTS_LAYER", "1")
    os.environ.setdefault("QT_QPA_PLATFORM", "cocoa")

try:
    from PyQt5.QtCore import Qt, pyqtSignal
    from PyQt5.QtWidgets import (
        QAbstractItemView,
        QApplication,
        QComboBox,
        QDoubleSpinBox,
        QFrame,
        QHeaderView,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QProgressBar,
        QPushButton,
        QCheckBox,
        QSpinBox,
        QStatusBar,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
    QT_API = "PyQt5"
except ImportError:
    from PyQt6.QtCore import Qt, pyqtSignal
    from PyQt6.QtWidgets import (
        QAbstractItemView,
        QApplication,
        QComboBox,
        QDoubleSpinBox,
        QFrame,
        QHeaderView,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QProgressBar,
        QPushButton,
        QCheckBox,
        QSpinBox,
        QStatusBar,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
    QT_API = "PyQt6"

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
except ImportError:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter

try:
    from kronos_mlx import Kronos as MLXKronos
    from kronos_mlx import KronosPredictor as MLXPredictor
    from kronos_mlx import KronosTokenizer as MLXTokenizer

    MLX_AVAILABLE = True
    MLX_IMPORT_ERROR = None
except ImportError as exc:
    MLX_AVAILABLE = False
    MLX_IMPORT_ERROR = exc

try:
    from model import Kronos as TorchKronos
    from model import KronosPredictor as TorchPredictor
    from model import KronosTokenizer as TorchTokenizer

    TORCH_AVAILABLE = True
    TORCH_IMPORT_ERROR = None
except ImportError as exc:
    TORCH_AVAILABLE = False
    TORCH_IMPORT_ERROR = exc

from qtgui.paper_strategy import (
    PaperPosition,
    PaperStrategyConfig,
    build_entry_decision,
    build_exit_decision,
    build_signal_snapshot,
)

LOCAL_MLX_MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "kronos-mlx-base")
LOCAL_MLX_TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "model", "kronos-mlx-tokenizer-base")
LOCAL_SMALL_MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "kronos-small")
LOCAL_SMALL_TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "model", "kronos-tokenizer-base")
LOCAL_MINI_MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "kronos-mini")
LOCAL_MINI_TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "model", "kronos-tokenizer-2k")
DISPLAY_TIMEZONE = "Asia/Taipei"
HEADER_STRETCH = QHeaderView.Stretch if hasattr(QHeaderView, "Stretch") else QHeaderView.ResizeMode.Stretch

LOCAL_MODEL_CONFIGS = {
    "kronos-mlx-base-8bit": {
        "label": "Kronos-base (MLX 8-bit, local)",
        "backend": "mlx",
        "context_length": 512,
        "model_path": LOCAL_MLX_MODEL_PATH,
        "tokenizer_path": LOCAL_MLX_TOKENIZER_PATH,
        "description": "Apple Silicon optimized local MLX model.",
    },
    "kronos-small-local": {
        "label": "Kronos-small (PyTorch, local)",
        "backend": "pytorch",
        "context_length": 512,
        "model_path": LOCAL_SMALL_MODEL_PATH,
        "tokenizer_path": LOCAL_SMALL_TOKENIZER_PATH,
        "description": "Balanced quality and speed. Good default local PyTorch model.",
    },
    "kronos-mini-local": {
        "label": "Kronos-mini (PyTorch, local)",
        "backend": "pytorch",
        "context_length": 2048,
        "model_path": LOCAL_MINI_MODEL_PATH,
        "tokenizer_path": LOCAL_MINI_TOKENIZER_PATH,
        "description": "Fastest local model with the longest context window.",
    },
}


def to_display_timestamp_series(values, *, assume_utc=False):
    series = pd.Series(pd.to_datetime(values)).reset_index(drop=True)
    tz = getattr(series.dt, "tz", None)
    if tz is not None:
        series = series.dt.tz_convert(DISPLAY_TIMEZONE).dt.tz_localize(None)
    elif assume_utc:
        series = series.dt.tz_localize("UTC").dt.tz_convert(DISPLAY_TIMEZONE).dt.tz_localize(None)
    series.name = "timestamps"
    return series


def ensure_timestamp_series(values):
    return to_display_timestamp_series(values)


def current_display_time():
    return pd.Timestamp.now(tz=DISPLAY_TIMEZONE).tz_localize(None)


def format_display_timestamp(timestamp=None):
    ts = current_display_time() if timestamp is None else to_display_timestamp_series([timestamp], assume_utc=True).iloc[0]
    return ts.strftime("%Y%m%d_%H%M%S")


def infer_time_delta(timestamps):
    ts = ensure_timestamp_series(timestamps)
    diffs = ts.diff().dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]
    if diffs.empty:
        return pd.Timedelta(hours=1)
    return diffs.median()


def build_future_timestamps(last_timestamp, step, pred_len):
    future = [last_timestamp + step * idx for idx in range(1, pred_len + 1)]
    return ensure_timestamp_series(future)


def format_price(value):
    return f"${value:,.0f}"


def format_signed_pct(value):
    return f"{value * 100:+.2f}%"


def default_lookback_for_context(context_length, pred_len):
    return max(50, context_length - pred_len)


def build_validation_segments(df, lookback, pred_len, *, drop_latest_bar=True):
    total = lookback + pred_len
    source_df = df.reset_index(drop=True)

    if drop_latest_bar and len(source_df) > total:
        source_df = source_df.iloc[:-1].reset_index(drop=True)

    if len(source_df) < total:
        return None

    validation_slice = source_df.tail(total).reset_index(drop=True)
    validation_history_df = validation_slice.iloc[:lookback].copy()
    validation_actual_df = validation_slice.iloc[lookback : lookback + pred_len].copy().reset_index(drop=True)
    validation_actual_df["timestamps"] = ensure_timestamp_series(validation_actual_df["timestamps"])
    return validation_history_df, validation_actual_df


class MetricCard(QFrame):
    def __init__(self, title, accent):
        super().__init__()
        self.setObjectName("metricCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(2)

        title_label = QLabel(title)
        title_label.setObjectName("metricTitle")
        title_label.setStyleSheet(f"color: {accent};")
        layout.addWidget(title_label)

        self.value_label = QLabel("--")
        self.value_label.setObjectName("metricValue")
        layout.addWidget(self.value_label)

        self.detail_label = QLabel("")
        self.detail_label.setObjectName("metricDetail")
        self.detail_label.setWordWrap(True)
        layout.addWidget(self.detail_label)

    def set_content(self, value, detail=""):
        self.value_label.setText(value)
        self.detail_label.setText(detail)


class TerminalBadge(QLabel):
    def __init__(self, label):
        super().__init__(label)
        self.setObjectName("tagLabel")
        self.setAlignment(Qt.AlignCenter)


class PriceChartCanvas(FigureCanvas):
    def __init__(self, chart_type="price"):
        self.chart_type = chart_type
        self.figure = Figure(figsize=(8.8, 6.4), facecolor="#151515")
        super().__init__(self.figure)
        self.setStyleSheet("background: transparent;")
        if chart_type == "price":
            self.draw_empty("Load data to see historical prices and forecast.")
        else:
            self.draw_empty("Load data to see validation results.")

    def draw_empty(self, message=None):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor("#1b1b1b")
        ax.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            fontsize=12,
            color="#d4d4d8",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        self.draw_idle()

    def _format_marker_time(self, timestamp):
        ts = pd.Timestamp(timestamp)
        return ts.strftime("%m-%d %H:%M")

    def _annotate_x_time(self, ax, timestamp, color):
        ax.annotate(
            self._format_marker_time(timestamp),
            xy=(pd.Timestamp(timestamp), 0),
            xycoords=("data", "axes fraction"),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color=color,
            fontsize=8,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.20",
                "facecolor": "#18181b",
                "edgecolor": color,
                "alpha": 0.95,
            },
            clip_on=False,
            zorder=10,
        )

    def plot_history(self, df, symbol, timeframe):
        if df is None or df.empty:
            self.draw_empty()
            return

        self.figure.clear()
        self.price_ax = self.figure.add_subplot(111)
        self.price_ax.set_facecolor("#1b1b1b")
        self.price_ax.grid(True, color="#4b5563", alpha=0.30, linewidth=0.8)
        self.price_ax.tick_params(colors="#d4d4d8", labelsize=9)
        self.price_ax.spines["bottom"].set_color("#52525b")
        self.price_ax.spines["left"].set_color("#52525b")
        self.price_ax.spines["top"].set_visible(False)
        self.price_ax.spines["right"].set_visible(False)
        self.price_ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))

        chart_df = df.tail(min(len(df), 240)).copy()
        self.price_ax.plot(
            chart_df["timestamps"],
            chart_df["close"],
            color="#71717a",
            linewidth=1.8,
            label="Historical close",
        )
        self.price_ax.fill_between(
            chart_df["timestamps"],
            chart_df["close"],
            chart_df["close"].min(),
            color="#93c5fd",
            alpha=0.08,
        )
        self.price_ax.set_title(
            f"{symbol} {timeframe} historical close",
            color="#fafafa",
            fontsize=12,
            loc="left",
        )
        self.price_ax.legend(
            loc="upper left",
            facecolor="#18181b",
            edgecolor="#3f3f46",
            labelcolor="#e5e7eb",
        )
        self.figure.tight_layout()
        self.draw_idle()

    def plot_forecast(
        self,
        *,
        symbol,
        timeframe,
        context_df,
        future_pred_df,
        forecast_interval_df=None,
    ):
        self.figure.clear()
        self.price_ax = self.figure.add_subplot(111)
        self.price_ax.set_facecolor("#1b1b1b")
        self.price_ax.grid(True, color="#4b5563", alpha=0.30, linewidth=0.8)
        self.price_ax.tick_params(colors="#d4d4d8", labelsize=9)
        self.price_ax.spines["bottom"].set_color("#52525b")
        self.price_ax.spines["left"].set_color("#52525b")
        self.price_ax.spines["top"].set_visible(False)
        self.price_ax.spines["right"].set_visible(False)
        self.price_ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))

        history_chart_df = context_df.tail(min(len(context_df), 240)).copy()
        forecast_df = future_pred_df.copy()
        if "timestamps" not in forecast_df.columns:
            forecast_df = forecast_df.reset_index().rename(columns={"index": "timestamps"})

        self.price_ax.plot(
            history_chart_df["timestamps"],
            history_chart_df["close"],
            color="#71717a",
            linewidth=1.8,
            label="Historical close",
        )
        if forecast_interval_df is not None and not forecast_interval_df.empty:
            self.price_ax.fill_between(
                forecast_interval_df["timestamps"],
                forecast_interval_df["lower_close"],
                forecast_interval_df["upper_close"],
                color="#fde68a",
                alpha=0.20,
                label="MC interval",
                zorder=1,
            )
        self.price_ax.plot(
            forecast_df["timestamps"],
            forecast_df["close"],
            color="#f59e0b",
            linewidth=2.0,
            marker="o",
            markersize=3,
            label="Forecast close",
            zorder=3,
        )
        self.price_ax.axvline(
            history_chart_df["timestamps"].iloc[-1],
            color="#a1a1aa",
            linewidth=1.0,
            linestyle="--",
            alpha=0.8,
        )
        self.price_ax.set_title(
            f"{symbol} {timeframe} history plus forward forecast",
            color="#fafafa",
            fontsize=12,
            loc="left",
        )
        self.price_ax.legend(
            loc="upper left",
            facecolor="#18181b",
            edgecolor="#3f3f46",
            labelcolor="#e5e7eb",
        )

        price_latest_x = history_chart_df["timestamps"].iloc[-1]
        price_latest_y = history_chart_df["close"].iloc[-1]
        self.price_ax.axvline(price_latest_x, color="#00d4ff", linewidth=1.0, linestyle="--", alpha=0.8)
        self.price_ax.axhline(price_latest_y, color="#00d4ff", linewidth=1.0, linestyle="--", alpha=0.8)
        self.price_ax.annotate(
            f" {price_latest_y:,.0f}",
            xy=(price_latest_x, price_latest_y),
            xytext=(5, 0),
            textcoords="offset points",
            color="#00d4ff",
            fontsize=10,
            fontweight="bold",
            va="center",
        )
        self._annotate_x_time(self.price_ax, price_latest_x, "#00d4ff")

        self.figure.tight_layout()
        self.draw_idle()

    def plot_validation(
        self,
        *,
        validation_history_df=None,
        validation_pred_df=None,
        validation_actual_df=None,
    ):
        self.figure.clear()
        self.validation_ax = self.figure.add_subplot(111)
        self.validation_ax.set_facecolor("#1b1b1b")
        self.validation_ax.grid(True, color="#4b5563", alpha=0.30, linewidth=0.8)
        self.validation_ax.tick_params(colors="#d4d4d8", labelsize=9)
        self.validation_ax.spines["bottom"].set_color("#52525b")
        self.validation_ax.spines["left"].set_color("#52525b")
        self.validation_ax.spines["top"].set_visible(False)
        self.validation_ax.spines["right"].set_visible(False)
        self.validation_ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))

        if (
            validation_history_df is not None
            and validation_pred_df is not None
            and validation_actual_df is not None
            and not validation_pred_df.empty
            and not validation_actual_df.empty
        ):
            val_hist_df = validation_history_df.tail(min(len(validation_history_df), 120)).copy()
            val_pred_df = validation_pred_df.copy()
            val_actual_df = validation_actual_df.copy()

            self.validation_ax.plot(
                val_hist_df["timestamps"],
                val_hist_df["close"],
                color="#71717a",
                linewidth=1.5,
                label="Validation history",
            )
            self.validation_ax.plot(
                val_pred_df["timestamps"],
                val_pred_df["close"],
                color="#c084fc",
                linewidth=1.8,
                label="Predicted close",
            )
            self.validation_ax.plot(
                val_actual_df["timestamps"],
                val_actual_df["close"],
                color="#34d399",
                linewidth=1.8,
                label="Actual close",
            )
            self.validation_ax.axvline(
                val_hist_df["timestamps"].iloc[-1],
                color="#a1a1aa",
                linewidth=1.0,
                linestyle="--",
                alpha=0.8,
            )
            start_price = val_hist_df["close"].iloc[-1]
            start_x = val_hist_df["timestamps"].iloc[-1]
            self.validation_ax.axhline(start_price, color="#71717a", linewidth=1.0, linestyle="--", alpha=0.8)
            self.validation_ax.annotate(
                f" {start_price:,.0f}",
                xy=(start_x, start_price),
                xytext=(5, 0),
                textcoords="offset points",
                color="#71717a",
                fontsize=10,
                fontweight="bold",
                va="center",
            )
            self._annotate_x_time(self.validation_ax, start_x, "#a1a1aa")
            self.validation_ax.set_title(
                "Recent validation on latest closed candles",
                color="#fafafa",
                fontsize=11,
                loc="left",
            )
            self.validation_ax.legend(
                loc="upper left",
                facecolor="#18181b",
                edgecolor="#3f3f46",
                labelcolor="#e5e7eb",
            )

            pred_latest_x = val_pred_df["timestamps"].iloc[-1]
            pred_latest_y = val_pred_df["close"].iloc[-1]
            self.validation_ax.annotate(
                f" {pred_latest_y:,.0f}",
                xy=(pred_latest_x, pred_latest_y),
                xytext=(5, 0),
                textcoords="offset points",
                color="#c084fc",
                fontsize=10,
                fontweight="bold",
                va="center",
            )

            valid_latest_x = val_actual_df["timestamps"].iloc[-1]
            valid_latest_y = val_actual_df["close"].iloc[-1]
            self.validation_ax.annotate(
                f" {valid_latest_y:,.0f}",
                xy=(valid_latest_x, valid_latest_y),
                xytext=(5, 0),
                textcoords="offset points",
                color="#34d399",
                fontsize=10,
                fontweight="bold",
                va="center",
            )
        else:
            self.validation_ax.text(
                0.5,
                0.5,
                "Not enough data for recent validation.",
                ha="center",
                va="center",
                color="#d4d4d8",
                fontsize=10,
                transform=self.validation_ax.transAxes,
            )
            self.validation_ax.set_xticks([])
            self.validation_ax.set_yticks([])

        self.figure.tight_layout()
        self.draw_idle()


class EquityCurveCanvas(FigureCanvas):
    def __init__(self):
        self.figure = Figure(figsize=(8.6, 3.0), facecolor="#151515")
        super().__init__(self.figure)
        self.setStyleSheet("background: transparent;")
        self.draw_empty()

    def draw_empty(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.trade_ax = ax
        ax.set_facecolor("#1b1b1b")
        ax.text(
            0.5,
            0.5,
            "Paper equity curve will appear after the first forecast update.",
            ha="center",
            va="center",
            fontsize=10,
            color="#d4d4d8",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        self.draw_idle()

    def plot_equity(self, equity_history):
        if not equity_history:
            self.draw_empty()
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor("#1b1b1b")
        ax.grid(True, color="#4b5563", alpha=0.30, linewidth=0.8)
        ax.tick_params(colors="#d4d4d8", labelsize=8)
        ax.spines["bottom"].set_color("#52525b")
        ax.spines["left"].set_color("#52525b")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))

        curve_df = pd.DataFrame(equity_history)
        ax.plot(
            curve_df["time"],
            curve_df["equity"],
            color="#22c55e",
            linewidth=2.0,
            marker="o",
            markersize=3,
        )
        ax.fill_between(
            curve_df["time"],
            curve_df["equity"],
            curve_df["equity"].min(),
            color="#22c55e",
            alpha=0.10,
        )
        latest_time = curve_df["time"].iloc[-1]
        latest_equity = float(curve_df["equity"].iloc[-1])
        ax.axvline(latest_time, color="#22c55e", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.axhline(latest_equity, color="#22c55e", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.annotate(
            f" {latest_equity:,.0f}",
            xy=(latest_time, latest_equity),
            xytext=(5, 0),
            textcoords="offset points",
            color="#22c55e",
            fontsize=9,
            fontweight="bold",
            va="center",
        )
        ax.set_title(
            "Normalized paper equity curve",
            color="#fafafa",
            fontsize=11,
            loc="left",
        )
        self.figure.tight_layout()
        self.draw_idle()


class PaperTradeChartCanvas(FigureCanvas):
    def __init__(self):
        self.figure = Figure(figsize=(8.6, 3.6), facecolor="#151515")
        super().__init__(self.figure)
        self.setStyleSheet("background: transparent;")
        self.draw_empty()

    def draw_empty(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor("#1b1b1b")
        ax.text(
            0.5,
            0.5,
            "Paper trades will be marked here after signal evaluation.",
            ha="center",
            va="center",
            fontsize=10,
            color="#d4d4d8",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        self.draw_idle()

    def plot_market_trades(self, *, df, trades, symbol, timeframe):
        if df is None or df.empty:
            self.draw_empty()
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.trade_ax = ax
        ax.set_facecolor("#1b1b1b")
        ax.grid(True, color="#4b5563", alpha=0.30, linewidth=0.8)
        ax.tick_params(colors="#d4d4d8", labelsize=8)
        ax.spines["bottom"].set_color("#52525b")
        ax.spines["left"].set_color("#52525b")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))

        chart_df = df.tail(min(len(df), 240)).copy()
        ax.plot(
            chart_df["timestamps"],
            chart_df["close"],
            color="#71717a",
            linewidth=1.7,
            label="Close",
        )

        filtered_trades = [
            trade for trade in trades
            if trade.get("symbol") == symbol and trade.get("timeframe") == timeframe
        ]
        markers = {
            "enter_long": {"marker": "^", "color": "#22c55e", "label": "Enter long"},
            "enter_short": {"marker": "v", "color": "#ef4444", "label": "Enter short"},
            "exit_long": {"marker": "X", "color": "#f59e0b", "label": "Exit long"},
            "exit_short": {"marker": "X", "color": "#38bdf8", "label": "Exit short"},
        }

        seen_labels = set()
        for action, style in markers.items():
            action_trades = [trade for trade in filtered_trades if trade["action"] == action]
            if not action_trades:
                continue
            xs = [trade["time"] for trade in action_trades]
            ys = [trade["price"] for trade in action_trades]
            label = style["label"] if action not in seen_labels else None
            ax.scatter(
                xs,
                ys,
                marker=style["marker"],
                s=64,
                color=style["color"],
                edgecolors="#fafafa",
                linewidths=0.6,
                label=label,
                zorder=5,
            )
            seen_labels.add(action)

        ax.set_title(
            f"{symbol} {timeframe} local paper trades",
            color="#fafafa",
            fontsize=11,
            loc="left",
        )
        if filtered_trades:
            ax.legend(
                loc="upper left",
                facecolor="#18181b",
                edgecolor="#3f3f46",
                labelcolor="#e5e7eb",
                fontsize=8,
            )
        self.figure.tight_layout()
        self.draw_idle()


class KronosGUI(QMainWindow):
    dispatch = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.predictor = None
        self.backend_name = "Not loaded"
        self.loaded_model_key = None
        self.model_configs = self.available_model_configs()
        self.selected_model_key = next(iter(self.model_configs))
        self.df = None
        self.current_symbol = "BTC/USDT"
        self.current_timeframe = "4h"
        self.dispatch.connect(self._run_on_ui_thread)
        self.auto_forecast_running = False
        self.auto_forecast_timer = None
        self.last_auto_forecast_time = None
        self.busy = False
        self.active_run_id = 0
        self.active_run_mode = None
        self.paper_strategy_config = PaperStrategyConfig()
        self.paper_position = None
        self.paper_trade_history = []
        self.paper_equity_history = []
        self.paper_last_snapshot = None
        self.paper_last_decision = None
        self.paper_realized_pnl_pct = 0.0
        self.paper_initial_equity = 100000.0
        self.paper_enabled = True
        self.paper_latest_market_df = None
        self.init_ui()
        self.show()

    def available_model_configs(self):
        configs = {}
        for key, config in LOCAL_MODEL_CONFIGS.items():
            if not (os.path.isdir(config["model_path"]) and os.path.isdir(config["tokenizer_path"])):
                continue
            if config["backend"] == "mlx" and not MLX_AVAILABLE:
                continue
            if config["backend"] == "pytorch" and not TORCH_AVAILABLE:
                continue
            configs[key] = config

        if not configs:
            raise RuntimeError(
                "No usable local Kronos models were found. "
                "Check the model folders and installed backend dependencies."
            )
        return configs

    def current_model_config(self):
        return self.model_configs[self.selected_model_key]

    def _run_on_ui_thread(self, callback):
        callback()

    def init_ui(self):
        self.setWindowTitle("Kronos BTC Forecast Terminal")
        self.setGeometry(90, 90, 1380, 900)
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #141414;
                color: #e4e4e7;
                font-size: 12px;
            }
            QGroupBox {
                border: 1px solid #303036;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: 600;
                background: #1c1c1f;
            }
            QGroupBox::title {
                left: 10px;
                padding: 0 6px;
                color: #f4f4f5;
            }
            QPushButton {
                background: #3f3f46;
                border: none;
                border-radius: 6px;
                padding: 8px 12px;
                color: #fafafa;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #52525b;
            }
            QPushButton:disabled {
                background: #27272a;
                color: #71717a;
            }
            QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit {
                background: #232326;
                border: 1px solid #3f3f46;
                border-radius: 6px;
                padding: 5px 7px;
                color: #f5f5f5;
                min-height: 20px;
            }
            QSpinBox::up-button, QSpinBox::down-button,
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                width: 24px;
                background: #3f3f46;
                border: none;
            }
            QSpinBox::up-button { border-radius: 0 6px 0 0; }
            QSpinBox::down-button { border-radius: 0 0 6px 0; }
            QDoubleSpinBox::up-button { border-radius: 0 6px 0 0; }
            QDoubleSpinBox::down-button { border-radius: 0 0 6px 0; }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover,
            QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {
                background: #52525b;
            }
            QSpinBox::up-arrow { image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAAdgAAAHYBTnsmCAAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAABVSURBVDiNY2AYBaMBMDIyMvzHwD8YkIGBgYHhjyoeEG9gYPj3H5mHgYGB4T8Dw38Ghv8MDP8Z/jP8Z2D4z8Dwn4HhPwPDfwaG/wwM/xkY/jMw/Gdg+M/A8B8DA8MAJgwMAADdJQr9b9HqPwAAAABJRU5ErkJggg==); }
            QSpinBox::down-arrow { image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAAdgAAAHYBTnsmCAAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAAA6SURBVDiNY2AYBaNgGABGRkbG/xgY/jMw/Gdg+M/A8J+B4T8Dw38Ghv8MDP8ZGP4zMPxnYPjPwPCfgeE/AwAj6Q0F9LZR8AAAAABJRU5ErkJggg==); }
            QDoubleSpinBox::up-arrow { image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAAdgAAAHYBTnsmCAAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAABVSURBVDiNY2AYBaMBMDIyMvzHwD8YkIGBgYHhjyoeEG9gYPj3H5mHgYGB4T8Dw38Ghv8MDP8Z/jP8Z2D4z8Dwn4HhPwPDfwaG/wwM/xkY/jMw/Gdg+M/A8B8DA8MAJgwMAADdJQr9b9HqPwAAAABJRU5ErkJggg==); }
            QDoubleSpinBox::down-arrow { image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAAdgAAAHYBTnsmCAAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAAA6SURBVDiNY2AYBaNgGABGRkbG/xgY/jMw/Gdg+M/A8J+B4T8Dw38Ghv8MDP8ZGP4zMPxnYPjPwPCfgeE/AwAj6Q0F9LZR8AAAAABJRU5ErkJggg==); }
            QTextEdit {
                padding: 6px;
                font-family: Menlo, Consolas, monospace;
                font-size: 11px;
            }
            QLabel#titleLabel {
                color: #fafafa;
                font-size: 22px;
                font-weight: 700;
            }
            QLabel#subtitleLabel {
                color: #a1a1aa;
                font-size: 11px;
            }
            QLabel#metricTitle {
                font-size: 10px;
                font-weight: 700;
            }
            QLabel#metricValue {
                color: #f8fafc;
                font-size: 18px;
                font-weight: 700;
            }
            QLabel#metricDetail {
                color: #a1a1aa;
                font-size: 10px;
            }
            QFrame#metricCard {
                background: #18181b;
                border: 1px solid #323238;
                border-radius: 6px;
            }
            QLabel#tagLabel {
                background: #232326;
                border: 1px solid #3f3f46;
                border-radius: 5px;
                color: #d4d4d8;
                padding: 4px 8px;
                font-size: 10px;
                font-weight: 600;
            }
            QStatusBar {
                background: #111111;
                color: #a1a1aa;
            }
            """
        )

        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)
        main_layout.addWidget(self.create_left_panel(), 0)
        main_layout.addWidget(self.create_right_panel(), 1)
        self.update_badges(feed_status="IDLE")

        self.auto_controls = [
            self.symbol_combo,
            self.tf_combo,
            self.limit_spin,
            self.lookback_spin,
            self.predlen_spin,
            self.temp_spin,
            self.topp_spin,
            self.sample_spin,
            self.model_combo,
            self.predict_btn,
        ]

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - load model, fetch Binance futures data, then forecast")

        self.raise_()
        self.activateWindow()

    def create_left_panel(self):
        group = QGroupBox("Terminal")
        group.setMinimumWidth(330)
        group.setMaximumWidth(360)
        layout = QVBoxLayout()
        layout.setSpacing(8)

        compact_header = QLabel("One click runs local Kronos load, Binance futures sync, and forecast.")
        compact_header.setWordWrap(True)
        compact_header.setStyleSheet("color: #a1a1aa;")
        layout.addWidget(compact_header)

        model_group = QGroupBox("Model Engine")
        model_layout = QGridLayout(model_group)
        model_layout.setContentsMargins(10, 10, 10, 10)
        model_layout.setHorizontalSpacing(8)
        model_layout.setVerticalSpacing(8)

        model_layout.addWidget(QLabel("Selection"), 0, 0)
        self.model_combo = QComboBox()
        for key, config in self.model_configs.items():
            self.model_combo.addItem(config["label"], key)
        self.model_combo.currentIndexChanged.connect(self.on_model_change)
        model_layout.addWidget(self.model_combo, 0, 1)

        self.model_info_label = QLabel("")
        self.model_info_label.setWordWrap(True)
        self.model_info_label.setStyleSheet("color: #a1a1aa;")
        model_layout.addWidget(self.model_info_label, 1, 0, 1, 2)
        layout.addWidget(model_group)

        market_group = QGroupBox("Market Feed")
        market_layout = QGridLayout(market_group)
        market_layout.setContentsMargins(10, 10, 10, 10)
        market_layout.setHorizontalSpacing(8)
        market_layout.setVerticalSpacing(8)

        market_layout.addWidget(QLabel("Symbol"), 0, 0)
        self.symbol_combo = QComboBox()
        self.symbol_combo.addItems(["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT"])
        self.symbol_combo.setCurrentText("BTC/USDT")
        market_layout.addWidget(self.symbol_combo, 0, 1)

        market_layout.addWidget(QLabel("Timeframe"), 1, 0)
        self.tf_combo = QComboBox()
        self.tf_combo.addItems(["5m", "15m", "1h", "4h", "1d"])
        self.tf_combo.setCurrentText("4h")
        self.tf_combo.currentTextChanged.connect(self.on_tf_change)
        market_layout.addWidget(self.tf_combo, 1, 1)

        market_layout.addWidget(QLabel("Limit"), 2, 0)
        self.limit_spin = QSpinBox()
        self.limit_spin.setRange(120, 2048)
        self.limit_spin.setValue(600)
        market_layout.addWidget(self.limit_spin, 2, 1)
        layout.addWidget(market_group)

        params = QGroupBox("Forecast Engine")
        params_layout = QGridLayout(params)
        params_layout.setContentsMargins(10, 10, 10, 10)
        params_layout.setHorizontalSpacing(8)
        params_layout.setVerticalSpacing(8)

        params_layout.addWidget(QLabel("Lookback"), 0, 0)
        self.lookback_spin = QSpinBox()
        self.lookback_spin.setRange(50, 512)
        self.lookback_spin.setValue(280)
        params_layout.addWidget(self.lookback_spin, 0, 1)

        params_layout.addWidget(QLabel("Forecast"), 1, 0)
        self.predlen_spin = QSpinBox()
        self.predlen_spin.setRange(1, 200)
        self.predlen_spin.setValue(18)
        params_layout.addWidget(self.predlen_spin, 1, 1)

        params_layout.addWidget(QLabel("Temp"), 2, 0)
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.1, 2.0)
        self.temp_spin.setSingleStep(0.1)
        self.temp_spin.setValue(0.5)
        params_layout.addWidget(self.temp_spin, 2, 1)

        params_layout.addWidget(QLabel("Top P"), 3, 0)
        self.topp_spin = QDoubleSpinBox()
        self.topp_spin.setRange(0.1, 1.0)
        self.topp_spin.setSingleStep(0.05)
        self.topp_spin.setValue(0.8)
        params_layout.addWidget(self.topp_spin, 3, 1)

        params_layout.addWidget(QLabel("Samples"), 4, 0)
        self.sample_spin = QSpinBox()
        self.sample_spin.setRange(1, 100)
        self.sample_spin.setValue(3)
        params_layout.addWidget(self.sample_spin, 4, 1)

        layout.addWidget(params)

        self.predict_btn = QPushButton("Run Single Forecast")
        self.predict_btn.clicked.connect(self.start_prediction)
        layout.addWidget(self.predict_btn)

        self.auto_forecast_btn = QPushButton("Run Auto Forecast")
        self.auto_forecast_btn.setStyleSheet("""
            QPushButton {
                background: #16a34a;
            }
            QPushButton:hover {
                background: #15803d;
            }
            QPushButton:disabled {
                background: #27272a;
                color: #71717a;
            }
        """)
        self.auto_forecast_btn.clicked.connect(self.toggle_auto_forecast)
        layout.addWidget(self.auto_forecast_btn)

        self.auto_status_label = QLabel("")
        self.auto_status_label.setStyleSheet("color: #16a34a; font-weight: bold;")
        layout.addWidget(self.auto_status_label)

        self.progress = QProgressBar()

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setRange(0, 0)
        layout.addWidget(self.progress)

        note = QLabel(
            "History + forecast terminal. Market orders remain disabled."
        )
        note.setWordWrap(True)
        note.setStyleSheet(
            "background: #202024; border: 1px solid #3a3a40; border-radius: 8px; padding: 8px; color: #a1a1aa;"
        )
        layout.addWidget(note)

        layout.addStretch()
        group.setLayout(layout)
        self.on_model_change(self.model_combo.currentIndex())
        return group

    def create_right_panel(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        header = QFrame()
        header.setObjectName("metricCard")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(14, 12, 14, 12)
        header_layout.setSpacing(6)

        self.title_label = QLabel("Kronos BTC Forecast Terminal")
        self.title_label.setObjectName("titleLabel")
        header_layout.addWidget(self.title_label)

        self.subtitle_label = QLabel("Load an engine, pull the latest futures candles, and inspect the next forecast window.")
        self.subtitle_label.setObjectName("subtitleLabel")
        self.subtitle_label.setWordWrap(True)
        header_layout.addWidget(self.subtitle_label)

        badge_row = QHBoxLayout()
        badge_row.setContentsMargins(0, 0, 0, 0)
        badge_row.setSpacing(8)
        self.symbol_badge = TerminalBadge("PAIR  BTC/USDT")
        self.timeframe_badge = TerminalBadge("TF  4H")
        self.model_badge = TerminalBadge("ENGINE  --")
        self.feed_badge = TerminalBadge("FEED  IDLE")
        badge_row.addWidget(self.symbol_badge)
        badge_row.addWidget(self.timeframe_badge)
        badge_row.addWidget(self.model_badge)
        badge_row.addWidget(self.feed_badge)
        badge_row.addStretch()
        header_layout.addLayout(badge_row)
        layout.addWidget(header)

        metrics_frame = QFrame()
        metrics_layout = QGridLayout(metrics_frame)
        metrics_layout.setContentsMargins(0, 0, 0, 0)
        metrics_layout.setHorizontalSpacing(10)
        metrics_layout.setVerticalSpacing(10)

        self.latest_card = MetricCard("Latest Close", "#38bdf8")
        self.forecast_card = MetricCard("Forecast End", "#f59e0b")
        self.move_card = MetricCard("Forecast Move", "#22c55e")
        self.mape_card = MetricCard("Recent MAPE", "#a78bfa")

        metrics_layout.addWidget(self.latest_card, 0, 0)
        metrics_layout.addWidget(self.forecast_card, 0, 1)
        metrics_layout.addWidget(self.move_card, 1, 0)
        metrics_layout.addWidget(self.mape_card, 1, 1)
        layout.addWidget(metrics_frame)

        self.main_tabs = QTabWidget()
        self.main_tabs.setStyleSheet("QTabWidget::pane { border: none; }")
        
        forecast_tab = QWidget()
        forecast_layout = QVBoxLayout(forecast_tab)
        forecast_layout.setContentsMargins(10, 10, 10, 10)
        forecast_layout.setSpacing(8)
        self.forecast_canvas = PriceChartCanvas(chart_type="price")
        forecast_layout.addWidget(self.forecast_canvas)
        self.main_tabs.addTab(forecast_tab, "Forecast")
        
        validation_tab = QWidget()
        validation_layout = QVBoxLayout(validation_tab)
        validation_layout.setContentsMargins(10, 10, 10, 10)
        validation_layout.setSpacing(8)
        self.validation_canvas = PriceChartCanvas(chart_type="validation")
        validation_layout.addWidget(self.validation_canvas)
        self.main_tabs.addTab(validation_tab, "Validation")

        paper_tab = self.create_paper_tab()
        self.main_tabs.addTab(paper_tab, "Paper Mode")

        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        log_layout.setContentsMargins(10, 10, 10, 10)
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setPlaceholderText(
            "Forecast details and run log will appear here."
        )
        log_layout.addWidget(self.results_text)
        self.main_tabs.addTab(log_tab, "Log")
        
        layout.addWidget(self.main_tabs)

        self.set_metric_defaults()
        return container

    def set_metric_defaults(self):
        self.latest_card.set_content("--", "Latest market close")
        self.forecast_card.set_content("--", "Forecast final close")
        self.move_card.set_content("--", "Predicted direction")
        self.mape_card.set_content("--", "Recent validation")

    def create_paper_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        hint_label = QLabel("Local paper mode evaluates one signal per closed candle and simulates entries, exits, and stop loss on-device.")
        hint_label.setWordWrap(True)
        hint_label.setStyleSheet("color: #a1a1aa;")
        layout.addWidget(hint_label)

        controls_frame = QFrame()
        controls_frame.setObjectName("metricCard")
        controls_layout = QGridLayout(controls_frame)
        controls_layout.setContentsMargins(12, 10, 12, 10)
        controls_layout.setHorizontalSpacing(10)
        controls_layout.setVerticalSpacing(8)

        self.paper_enable_checkbox = QCheckBox("Enable Paper Trading")
        self.paper_enable_checkbox.setChecked(self.paper_enabled)
        self.paper_enable_checkbox.toggled.connect(self.on_paper_trading_toggled)
        controls_layout.addWidget(self.paper_enable_checkbox, 0, 0)

        controls_layout.addWidget(QLabel("Initial Equity"), 0, 1)
        self.paper_initial_equity_spin = QDoubleSpinBox()
        self.paper_initial_equity_spin.setRange(1000.0, 100000000.0)
        self.paper_initial_equity_spin.setDecimals(0)
        self.paper_initial_equity_spin.setSingleStep(1000.0)
        self.paper_initial_equity_spin.setValue(self.paper_initial_equity)
        self.paper_initial_equity_spin.valueChanged.connect(self.on_paper_initial_equity_changed)
        controls_layout.addWidget(self.paper_initial_equity_spin, 0, 2)

        self.paper_reset_btn = QPushButton("Reset Paper Account")
        self.paper_reset_btn.clicked.connect(self.reset_paper_account)
        controls_layout.addWidget(self.paper_reset_btn, 0, 3)
        layout.addWidget(controls_frame)

        metrics_frame = QFrame()
        metrics_layout = QGridLayout(metrics_frame)
        metrics_layout.setContentsMargins(0, 0, 0, 0)
        metrics_layout.setHorizontalSpacing(10)
        metrics_layout.setVerticalSpacing(10)

        self.paper_signal_card = MetricCard("Signal", "#38bdf8")
        self.paper_position_card = MetricCard("Position", "#f59e0b")
        self.paper_stop_card = MetricCard("Stop", "#ef4444")
        self.paper_pnl_card = MetricCard("PnL", "#22c55e")

        metrics_layout.addWidget(self.paper_signal_card, 0, 0)
        metrics_layout.addWidget(self.paper_position_card, 0, 1)
        metrics_layout.addWidget(self.paper_stop_card, 1, 0)
        metrics_layout.addWidget(self.paper_pnl_card, 1, 1)
        layout.addWidget(metrics_frame)

        self.paper_status_label = QLabel("")
        self.paper_status_label.setWordWrap(True)
        self.paper_status_label.setStyleSheet("color: #d4d4d8;")
        layout.addWidget(self.paper_status_label)

        self.paper_snapshot_text = QTextEdit()
        self.paper_snapshot_text.setReadOnly(True)
        self.paper_snapshot_text.setPlaceholderText("Latest paper mode signal snapshot will appear here.")
        
        top_grid = QGridLayout()
        top_grid.setContentsMargins(0, 0, 0, 0)
        top_grid.setHorizontalSpacing(10)
        top_grid.setVerticalSpacing(10)

        position_group = QGroupBox("Position Watch")
        position_layout = QVBoxLayout(position_group)
        position_layout.setContentsMargins(10, 10, 10, 10)
        position_layout.setSpacing(8)
        self.paper_position_table = QTableWidget(1, 6)
        self.paper_position_table.setHorizontalHeaderLabels(
            ["Side", "Entry Time", "Entry", "Mark", "Stop", "Unrealized"]
        )
        self.paper_position_table.verticalHeader().setVisible(False)
        self.paper_position_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.paper_position_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.paper_position_table.horizontalHeader().setSectionResizeMode(HEADER_STRETCH)
        position_layout.addWidget(self.paper_position_table)
        position_layout.addWidget(self.paper_snapshot_text)
        top_grid.addWidget(position_group, 0, 0)

        equity_group = QGroupBox("Equity Curve")
        equity_layout = QVBoxLayout(equity_group)
        equity_layout.setContentsMargins(10, 10, 10, 10)
        equity_layout.setSpacing(8)
        self.paper_equity_canvas = EquityCurveCanvas()
        equity_layout.addWidget(self.paper_equity_canvas)
        top_grid.addWidget(equity_group, 0, 1)

        layout.addLayout(top_grid, 1)

        trade_chart_group = QGroupBox("Trade Overlay")
        trade_chart_layout = QVBoxLayout(trade_chart_group)
        trade_chart_layout.setContentsMargins(10, 10, 10, 10)
        trade_chart_layout.setSpacing(8)
        self.paper_trade_chart_canvas = PaperTradeChartCanvas()
        trade_chart_layout.addWidget(self.paper_trade_chart_canvas)
        layout.addWidget(trade_chart_group, 1)

        trade_group = QGroupBox("Trade Ledger")
        trade_layout = QVBoxLayout(trade_group)
        trade_layout.setContentsMargins(10, 10, 10, 10)
        trade_layout.setSpacing(8)
        self.paper_trade_table = QTableWidget(0, 8)
        self.paper_trade_table.setHorizontalHeaderLabels(
            ["Time", "Action", "Price", "Stop", "Forecast", "Val Pred", "Val Actual", "PnL"]
        )
        self.paper_trade_table.verticalHeader().setVisible(False)
        self.paper_trade_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.paper_trade_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.paper_trade_table.horizontalHeader().setSectionResizeMode(HEADER_STRETCH)
        trade_layout.addWidget(self.paper_trade_table)

        self.paper_trade_log = QTextEdit()
        self.paper_trade_log.setReadOnly(True)
        self.paper_trade_log.setPlaceholderText("Local paper trades will appear here.")
        trade_layout.addWidget(self.paper_trade_log)
        layout.addWidget(trade_group, 1)

        self.set_paper_defaults()
        return tab

    def set_paper_defaults(self):
        if hasattr(self, "paper_signal_card"):
            self.paper_signal_card.set_content("WAIT", "Waiting for a completed forecast")
        if hasattr(self, "paper_position_card"):
            self.paper_position_card.set_content("FLAT", "No local paper position")
        if hasattr(self, "paper_stop_card"):
            self.paper_stop_card.set_content("--", "Stop price will arm on entry")
        if hasattr(self, "paper_pnl_card"):
            self.paper_pnl_card.set_content("$0", "Realized +0.00% | Unrealized +0.00%")
        if hasattr(self, "paper_status_label"):
            self.paper_status_label.setText("Local paper mode is idle until forecast and validation data are both ready.")
        if hasattr(self, "paper_snapshot_text"):
            self.paper_snapshot_text.setPlainText("")
        if hasattr(self, "paper_trade_log"):
            self.paper_trade_log.setPlainText("")
        if hasattr(self, "paper_position_table"):
            for col in range(self.paper_position_table.columnCount()):
                self.paper_position_table.setItem(0, col, QTableWidgetItem("--"))
        if hasattr(self, "paper_trade_table"):
            self.paper_trade_table.setRowCount(0)
        if hasattr(self, "paper_equity_canvas"):
            self.paper_equity_canvas.draw_empty()
        if hasattr(self, "paper_trade_chart_canvas"):
            self.paper_trade_chart_canvas.draw_empty()

    def on_paper_trading_toggled(self, checked):
        self.paper_enabled = bool(checked)
        if self.paper_enabled:
            self.paper_status_label.setText("Paper trading enabled. The next completed forecast candle can open or close local positions.")
        else:
            self.paper_last_decision = None
            self.paper_status_label.setText("Paper trading disabled. Signals are visible, but no local entries or exits will be simulated.")
        self.render_paper_mode()

    def on_paper_initial_equity_changed(self, value):
        value = float(value)
        if value <= 0:
            return
        if abs(value - self.paper_initial_equity) < 1e-9:
            return
        self.paper_initial_equity = value
        self.reset_paper_account(reason=f"Paper account reset with initial equity {format_price(value)}.")

    def reset_paper_account(self, checked=False, *, reason=None):
        self.paper_position = None
        self.paper_trade_history = []
        self.paper_equity_history = []
        self.paper_realized_pnl_pct = 0.0
        self.paper_last_decision = None
        self.paper_last_snapshot = None
        self.paper_latest_market_df = None
        self.set_paper_defaults()
        if hasattr(self, "paper_status_label"):
            self.paper_status_label.setText(
                reason or f"Paper account reset. Starting equity {format_price(self.paper_initial_equity)}."
            )

    def update_badges(self, *, feed_status=None):
        symbol = getattr(self, "current_symbol", "BTC/USDT")
        timeframe = getattr(self, "current_timeframe", "4h")
        model_name = self.current_model_config()["label"] if hasattr(self, "model_configs") else "--"

        if hasattr(self, "symbol_badge"):
            self.symbol_badge.setText(f"PAIR  {symbol}")
        if hasattr(self, "timeframe_badge"):
            self.timeframe_badge.setText(f"TF  {timeframe.upper()}")
        if hasattr(self, "model_badge"):
            engine_text = self.backend_name if self.backend_name != "Not loaded" else model_name
            self.model_badge.setText(f"ENGINE  {engine_text}")
        if hasattr(self, "feed_badge") and feed_status is not None:
            self.feed_badge.setText(f"FEED  {feed_status}")

    def on_tf_change(self, tf):
        presets = {
            "5m": 36,
            "15m": 24,
            "1h": 48,
            "4h": 18,
            "1d": 7,
        }
        pred_len = presets.get(tf, 18)
        self.predlen_spin.setValue(pred_len)
        context_length = self.current_model_config()["context_length"]
        self.lookback_spin.setValue(default_lookback_for_context(context_length, pred_len))

    def on_model_change(self, index):
        if index < 0:
            return

        self.selected_model_key = self.model_combo.itemData(index)
        config = self.current_model_config()
        context_length = config["context_length"]
        self.model_info_label.setText(
            f"{config['description']}\nContext length: {context_length}"
        )

        self.lookback_spin.setMaximum(context_length)
        self.predlen_spin.setMaximum(min(400, context_length - 1))

        if self.predlen_spin.value() >= context_length:
            self.predlen_spin.setValue(max(1, min(self.predlen_spin.value(), context_length - 1)))
        self.lookback_spin.setValue(
            default_lookback_for_context(context_length, self.predlen_spin.value())
        )

        if self.predictor is not None:
            self.predictor = None
            self.backend_name = "Not loaded"
            self.loaded_model_key = None
            self.append_log(f"[INFO] Model selection changed to {config['label']}. Reload model to apply.")
        if hasattr(self, "title_label"):
            self.title_label.setText(f"{self.symbol_combo.currentText()} futures forecast")
        if hasattr(self, "subtitle_label"):
            self.subtitle_label.setText(
                f"Ready to run {config['label']} | context length {context_length}"
            )
        self.update_badges(feed_status="IDLE")
        self.set_busy()

    def append_log(self, message):
        self.results_text.append(message)

    def set_busy(self, *, loading=False, fetching=False, predicting=False):
        busy = loading or fetching or predicting
        self.busy = busy
        if self.auto_forecast_running and not predicting:
            return
        self.model_combo.setEnabled(not busy)
        self.symbol_combo.setEnabled(not busy)
        self.tf_combo.setEnabled(not busy)
        self.limit_spin.setEnabled(not busy)
        self.lookback_spin.setEnabled(not busy)
        self.predlen_spin.setEnabled(not busy)
        self.temp_spin.setEnabled(not busy)
        self.topp_spin.setEnabled(not busy)
        self.sample_spin.setEnabled(not busy)
        self.predict_btn.setEnabled(not busy)
        if not self.auto_forecast_running:
            self.auto_forecast_btn.setEnabled(not busy)
        self.progress.setVisible(busy)

    def describe_paper_action(self, action):
        labels = {
            "enter_long": "ENTER LONG",
            "enter_short": "ENTER SHORT",
            "exit_long": "EXIT LONG",
            "exit_short": "EXIT SHORT",
            "hold": "HOLD",
            "no_action": "NO ACTION",
        }
        return labels.get(action, action.upper())

    def describe_paper_reason(self, reason):
        labels = {
            "forecast_upside_and_validation_trend_confirm_long": "Forecast upside and validation trend both confirm a long entry.",
            "forecast_downside_and_validation_trend_confirm_short": "Forecast downside and validation trend both confirm a short entry.",
            "entry_conditions_not_met": "Forecast edge or validation alignment is not strong enough yet.",
            "stop_loss_hit": "Current price crossed the local stop boundary.",
            "long_signal_invalidated": "Long setup lost support from forecast or validation trend.",
            "short_signal_invalidated": "Short setup lost support from forecast or validation trend.",
            "long_position_still_valid": "Long setup still agrees with forecast and validation trend.",
            "short_position_still_valid": "Short setup still agrees with forecast and validation trend.",
            "missing_validation_data": "Validation data is not ready, so paper mode is waiting.",
        }
        return labels.get(reason, reason.replace("_", " ").capitalize())

    def compute_position_return_pct(self, position, current_price):
        if position.side == "long":
            return (current_price / position.entry_price) - 1.0
        return (position.entry_price / current_price) - 1.0

    def current_paper_equity(self, unrealized_pct=0.0):
        return self.paper_initial_equity * (1.0 + self.paper_realized_pnl_pct + unrealized_pct)

    def append_paper_equity_point(self, snapshot, unrealized_pct):
        point = {
            "time": pd.Timestamp(snapshot.signal_time),
            "equity": self.current_paper_equity(unrealized_pct),
        }
        if self.paper_equity_history and self.paper_equity_history[-1]["time"] == point["time"]:
            self.paper_equity_history[-1] = point
            return
        self.paper_equity_history.append(point)
        self.paper_equity_history = self.paper_equity_history[-240:]

    def set_table_item(self, table, row, column, value):
        table.setItem(row, column, QTableWidgetItem(value))

    def record_paper_trade(self, *, action, snapshot, reason, stop_price=None, pnl_pct=None):
        trade = {
            "time": pd.Timestamp(snapshot.signal_time),
            "action": action,
            "price": float(snapshot.current_price),
            "reason": reason,
            "stop_price": stop_price,
            "forecast_return": float(snapshot.forecast_return),
            "validation_pred_return": float(snapshot.validation_pred_return),
            "validation_actual_return": float(snapshot.validation_actual_return),
            "pnl_pct": pnl_pct,
            "symbol": getattr(self, "current_symbol", "BTC/USDT"),
            "timeframe": getattr(self, "current_timeframe", "4h"),
        }
        self.paper_trade_history.append(trade)
        self.paper_trade_history = self.paper_trade_history[-40:]
        self.append_log(
            f"[PAPER] {self.describe_paper_action(action)} @ {format_price(trade['price'])} | "
            f"forecast {format_signed_pct(trade['forecast_return'])} | "
            f"reason: {self.describe_paper_reason(reason)}"
        )

    def render_paper_mode(self):
        snapshot = self.paper_last_snapshot
        decision = self.paper_last_decision
        position = self.paper_position

        if snapshot is None:
            self.set_paper_defaults()
            return

        disabled_detail = "Paper trading disabled. Forecast and validation are visible in monitor mode."
        if not self.paper_enabled:
            signal_value = "DISABLED"
            signal_detail = disabled_detail
        else:
            signal_value = self.describe_paper_action(decision.action if decision else "no_action")
            signal_detail = self.describe_paper_reason(decision.reason if decision else "entry_conditions_not_met")
        self.paper_signal_card.set_content(signal_value, signal_detail)

        if position is None:
            self.paper_position_card.set_content("FLAT", "No local paper position")
            self.paper_stop_card.set_content("--", "Stop price inactive while flat")
            unrealized_pct = 0.0
        else:
            self.paper_position_card.set_content(
                f"{position.side.upper()} @ {format_price(position.entry_price)}",
                f"Entry {position.entry_time.strftime('%Y-%m-%d %H:%M')}",
            )
            stop_distance = abs(position.stop_price / position.entry_price - 1.0)
            self.paper_stop_card.set_content(
                format_price(position.stop_price),
                f"Armed at {format_signed_pct(stop_distance)} from entry",
            )
            unrealized_pct = self.compute_position_return_pct(position, snapshot.current_price)

        self.append_paper_equity_point(snapshot, unrealized_pct)
        self.paper_pnl_card.set_content(
            format_signed_pct(unrealized_pct),
            f"Realized {format_signed_pct(self.paper_realized_pnl_pct)} | Equity {format_price(self.current_paper_equity(unrealized_pct))}",
        )

        snapshot_lines = [
            f"Signal time: {snapshot.signal_time.strftime('%Y-%m-%d %H:%M')}",
            f"Current close: {format_price(snapshot.current_price)}",
            f"Forecast close: {format_price(snapshot.forecast_price)} ({format_signed_pct(snapshot.forecast_return)})",
            f"Validation start: {format_price(snapshot.validation_start_price)}",
            f"Validation predicted end: {format_price(snapshot.validation_pred_price)} ({format_signed_pct(snapshot.validation_pred_return)})",
            f"Validation actual end: {format_price(snapshot.validation_actual_price)} ({format_signed_pct(snapshot.validation_actual_return)})",
            f"Validation history delta: {format_price(snapshot.validation_history_delta)}",
            f"Validation actual delta: {format_price(snapshot.validation_actual_delta)}",
        ]
        if decision and decision.stop_price is not None:
            snapshot_lines.append(
                f"Suggested stop: {format_price(decision.stop_price)} ({format_signed_pct(decision.stop_distance_pct)})"
            )
        self.paper_snapshot_text.setPlainText("\n".join(snapshot_lines))

        position_values = ["FLAT", "--", "--", format_price(snapshot.current_price), "--", "+0.00%"]
        if position is not None:
            position_values = [
                position.side.upper(),
                position.entry_time.strftime("%Y-%m-%d %H:%M"),
                format_price(position.entry_price),
                format_price(snapshot.current_price),
                format_price(position.stop_price),
                format_signed_pct(unrealized_pct),
            ]
        for idx, value in enumerate(position_values):
            self.set_table_item(self.paper_position_table, 0, idx, value)

        trade_lines = []
        self.paper_trade_table.setRowCount(len(self.paper_trade_history))
        for trade in reversed(self.paper_trade_history):
            line = (
                f"{trade['time'].strftime('%Y-%m-%d %H:%M')} | "
                f"{self.describe_paper_action(trade['action'])} | "
                f"{format_price(trade['price'])} | "
                f"forecast {format_signed_pct(trade['forecast_return'])}"
            )
            if trade["stop_price"] is not None:
                line += f" | stop {format_price(trade['stop_price'])}"
            if trade["pnl_pct"] is not None:
                line += f" | pnl {format_signed_pct(trade['pnl_pct'])}"
            line += f" | {self.describe_paper_reason(trade['reason'])}"
            trade_lines.append(line)
        for row, trade in enumerate(reversed(self.paper_trade_history)):
            values = [
                trade["time"].strftime("%Y-%m-%d %H:%M"),
                self.describe_paper_action(trade["action"]),
                format_price(trade["price"]),
                format_price(trade["stop_price"]) if trade["stop_price"] is not None else "--",
                format_signed_pct(trade["forecast_return"]),
                format_signed_pct(trade["validation_pred_return"]),
                format_signed_pct(trade["validation_actual_return"]),
                format_signed_pct(trade["pnl_pct"]) if trade["pnl_pct"] is not None else "--",
            ]
            for col, value in enumerate(values):
                self.set_table_item(self.paper_trade_table, row, col, value)
        self.paper_trade_log.setPlainText("\n".join(trade_lines))
        self.paper_equity_canvas.plot_equity(self.paper_equity_history)
        self.paper_trade_chart_canvas.plot_market_trades(
            df=self.paper_latest_market_df,
            trades=self.paper_trade_history,
            symbol=getattr(self, "current_symbol", "BTC/USDT"),
            timeframe=getattr(self, "current_timeframe", "4h"),
        )

        if not self.paper_enabled:
            position_text = "Paper trading disabled."
        else:
            position_text = "Paper position active." if position is not None else "Paper mode is flat."
        self.paper_status_label.setText(
            f"{position_text} Latest decision: {signal_value}. {signal_detail}"
        )

    def update_paper_mode(self, payload):
        self.paper_latest_market_df = payload.get("df")
        try:
            snapshot = build_signal_snapshot(payload)
        except ValueError:
            self.paper_last_snapshot = None
            self.paper_last_decision = None
            self.set_paper_defaults()
            self.paper_status_label.setText(self.describe_paper_reason("missing_validation_data"))
            return

        self.paper_last_snapshot = snapshot
        if not self.paper_enabled:
            self.paper_last_decision = None
            self.render_paper_mode()
            return

        if self.paper_position is None:
            decision = build_entry_decision(snapshot, self.paper_strategy_config)
            self.paper_last_decision = decision
            if decision.action in ("enter_long", "enter_short"):
                side = "long" if decision.action == "enter_long" else "short"
                self.paper_position = PaperPosition(
                    side=side,
                    entry_price=snapshot.current_price,
                    stop_price=decision.stop_price,
                    entry_time=snapshot.signal_time,
                )
                self.record_paper_trade(
                    action=decision.action,
                    snapshot=snapshot,
                    reason=decision.reason,
                    stop_price=decision.stop_price,
                )
        else:
            decision = build_exit_decision(self.paper_position, snapshot, self.paper_strategy_config)
            self.paper_last_decision = decision
            if decision.action in ("exit_long", "exit_short"):
                pnl_pct = self.compute_position_return_pct(self.paper_position, snapshot.current_price)
                self.paper_realized_pnl_pct += pnl_pct
                self.record_paper_trade(
                    action=decision.action,
                    snapshot=snapshot,
                    reason=decision.reason,
                    stop_price=self.paper_position.stop_price,
                    pnl_pct=pnl_pct,
                )
                self.paper_position = None

        self.render_paper_mode()

    def show_error(self, title, message):
        self.append_log(f"[ERROR] {message}")
        QMessageBox.critical(self, title, message)

    def create_predictor(self):
        config = self.current_model_config()

        if config["backend"] == "mlx":
            if not MLX_AVAILABLE:
                raise RuntimeError(f"MLX backend unavailable: {MLX_IMPORT_ERROR}")
            tokenizer = MLXTokenizer.from_pretrained(config["tokenizer_path"])
            model = MLXKronos.from_pretrained(config["model_path"], bits=8)
            return MLXPredictor(model, tokenizer, max_context=config["context_length"]), config["label"]

        if config["backend"] == "pytorch":
            if not TORCH_AVAILABLE:
                raise RuntimeError(f"PyTorch backend unavailable: {TORCH_IMPORT_ERROR}")
            tokenizer = TorchTokenizer.from_pretrained(config["tokenizer_path"])
            model = TorchKronos.from_pretrained(config["model_path"])
            return TorchPredictor(model, tokenizer, max_context=config["context_length"]), config["label"]

        raise RuntimeError(f"Unsupported backend: {config['backend']}")

    def fetch_market_data_sync(self, symbol, timeframe, limit):
        exchange = ccxt.binance(
            {
                "enableRateLimit": True,
                "options": {"defaultType": "future"},
            }
        )
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(
            ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamps"] = to_display_timestamp_series(
            pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        )
        df = df.drop(columns=["timestamp"]).reset_index(drop=True)
        df["amount"] = df["volume"] * df["close"] * 0.0001
        return df

    def ensure_predictor_loaded(self):
        config = self.current_model_config()
        if self.predictor is not None and self.loaded_model_key == self.selected_model_key:
            return self.predictor, self.backend_name

        predictor, backend_name = self.create_predictor()
        return predictor, backend_name

    def build_loaded_data_summary(self, df, symbol, timeframe):
        start = df["timestamps"].min().strftime("%Y-%m-%d %H:%M")
        end = df["timestamps"].max().strftime("%Y-%m-%d %H:%M")
        return {
            "start": start,
            "end": end,
            "last_close": float(df["close"].iloc[-1]),
            "rows": len(df),
            "symbol": symbol,
            "timeframe": timeframe,
        }

    def toggle_auto_forecast(self):
        if self.auto_forecast_running:
            self.stop_auto_forecast()
        else:
            self.start_auto_forecast()

    def start_auto_forecast(self):
        self.auto_forecast_running = True
        self.active_run_mode = "auto"
        self.auto_forecast_btn.setText("Stop Auto Forecast")
        self._set_controls_enabled(False)
        self.auto_status_label.setText("Auto Forecast: Running")
        self.status_bar.showMessage("Auto forecast running")
        self.append_log("[AUTO] Auto forecast started")
        self._run_auto_forecast()

    def stop_auto_forecast(self):
        self.auto_forecast_running = False
        self.active_run_id += 1
        self.active_run_mode = None
        if self.auto_forecast_timer:
            self.auto_forecast_timer.cancel()
            self.auto_forecast_timer = None
        self.auto_forecast_btn.setText("Run Auto Forecast")
        self.auto_forecast_btn.setEnabled(True)
        self.auto_status_label.setText("Auto Forecast: Stopped")
        self.status_bar.showMessage("Auto forecast stopped")
        self.update_badges(feed_status="IDLE")
        self.append_log("[AUTO] Auto forecast stopped")
        self._set_controls_enabled(True)
        self.set_busy()

    def _set_controls_enabled(self, enabled):
        for ctrl in self.auto_controls:
            ctrl.setEnabled(enabled)

    def _begin_run(self, auto_run):
        self.active_run_id += 1
        self.active_run_mode = "auto" if auto_run else "manual"
        return self.active_run_id

    def _is_run_active(self, run_id, auto_run):
        if run_id != self.active_run_id:
            return False
        if auto_run and not self.auto_forecast_running:
            return False
        return True

    def _run_auto_forecast(self):
        if not self.auto_forecast_running:
            return
        if hasattr(self, 'busy') and self.busy:
            self.append_log("[AUTO] Previous forecast still running, skipping this cycle")
            return
        self.append_log(f"[AUTO] Running scheduled forecast at {pd.Timestamp.now()}")
        self.start_prediction(auto_run=True)

    def _schedule_next_auto_forecast(self):
        if not self.auto_forecast_running:
            return
        timeframe = self.current_timeframe
        now = pd.Timestamp.now()
        next_run = self._get_next_candle_time(now, timeframe)
        wait_seconds = (next_run - now).total_seconds()
        if wait_seconds <= 0:
            wait_seconds = self._get_interval_seconds(timeframe)
        self.auto_forecast_timer = threading.Timer(wait_seconds, self._on_timer_elapsed)
        self.auto_forecast_timer.daemon = True
        self.auto_forecast_timer.start()
        self.last_auto_forecast_time = now
        self.auto_status_label.setText(f"Next: {next_run.strftime('%H:%M')}")

    def _get_interval_seconds(self, timeframe):
        intervals = {
            "5m": 5 * 60,
            "15m": 15 * 60,
            "1h": 60 * 60,
            "4h": 4 * 60 * 60,
            "1d": 24 * 60 * 60,
        }
        return intervals.get(timeframe, 3600)

    def _get_next_candle_time(self, now, timeframe):
        tf_minutes = {
            "5m": 5,
            "15m": 15,
            "1h": 60,
            "4h": 240,
            "1d": 1440,
        }
        mins = tf_minutes.get(timeframe, 60)
        freq = pd.DateOffset(minutes=mins)
        if timeframe == "1d":
            current_candle = now.replace(hour=0, minute=0, second=0, microsecond=0)
            if now.hour > 0:
                current_candle += pd.DateOffset(days=1)
        else:
            minutes_since_midnight = now.hour * 60 + now.minute
            current_candle_num = (minutes_since_midnight // mins) * mins
            current_candle = now.replace(hour=0, minute=0, second=0, microsecond=0)
            current_candle += pd.DateOffset(minutes=current_candle_num)
        next_candle = current_candle + freq
        if next_candle <= now:
            next_candle += freq
        return next_candle

    def _on_timer_elapsed(self):
        self.dispatch.emit(self._run_auto_forecast)

    def _save_forecast_results(self, payload):
        symbol = payload.get("symbol", "UNKNOWN").replace("/", "_")
        timeframe = payload.get("timeframe", "UNKNOWN")
        timestamp = format_display_timestamp()
        save_dir = os.path.join(PROJECT_ROOT, "prediction_results", symbol, timeframe)
        os.makedirs(save_dir, exist_ok=True)

        log_file = os.path.join(save_dir, f"log_{timestamp}.txt")
        with open(log_file, "w") as f:
            f.write(payload.get("summary_text", ""))

        try:
            forecast_path = os.path.join(save_dir, f"forecast_{timestamp}.png")
            self.forecast_canvas.figure.savefig(forecast_path, dpi=150, facecolor="#151515")
            validation_path = os.path.join(save_dir, f"validation_{timestamp}.png")
            self.validation_canvas.figure.savefig(validation_path, dpi=150, facecolor="#151515")
            self.append_log(f"[AUTO] Saved: forecast/validation/log files at {timestamp}")
        except Exception as e:
            self.append_log(f"[AUTO] Warning: Could not save charts: {e}")

    def start_prediction(self, checked=False, *, auto_run=False):
        symbol = self.symbol_combo.currentText()
        timeframe = self.tf_combo.currentText()
        limit = self.limit_spin.value()
        lookback = self.lookback_spin.value()
        pred_len = self.predlen_spin.value()
        temperature = self.temp_spin.value()
        top_p = self.topp_spin.value()
        sample_count = self.sample_spin.value()
        max_context = self.current_model_config()["context_length"]

        if lookback > max_context or lookback + pred_len > max_context:
            QMessageBox.warning(
                self,
                "Warning",
                f"lookback + pred_len must stay within {max_context} (current: {lookback + pred_len})",
            )
            return

        run_id = self._begin_run(auto_run)
        self.set_busy(predicting=True)
        self.current_symbol = symbol
        self.current_timeframe = timeframe
        self.title_label.setText(f"{symbol} futures forecast")
        self.subtitle_label.setText(
            f"Running local forecast pipeline | {symbol} {timeframe} | lookback={lookback} | pred_len={pred_len}"
        )
        self.update_badges(feed_status="FORECAST")
        self.append_log(
            f"[INFO] Run requested: load model, fetch {symbol} {timeframe}, then forecast."
        )

        def work():
            try:
                if not self._is_run_active(run_id, auto_run):
                    return

                self.dispatch.emit(
                    lambda run_id=run_id, auto_run=auto_run: self._is_run_active(run_id, auto_run)
                    and self.append_log("[INFO] Loading selected model...")
                )
                predictor, backend_name = self.ensure_predictor_loaded()

                if not self._is_run_active(run_id, auto_run):
                    return

                self.dispatch.emit(
                    lambda run_id=run_id, auto_run=auto_run: self._is_run_active(run_id, auto_run)
                    and self.update_badges(feed_status="SYNC")
                )
                self.dispatch.emit(
                    lambda run_id=run_id, auto_run=auto_run: self._is_run_active(run_id, auto_run)
                    and self.append_log("[INFO] Syncing latest Binance futures candles...")
                )
                df = self.fetch_market_data_sync(symbol, timeframe, limit)

                if not self._is_run_active(run_id, auto_run):
                    return

                if len(df) < lookback:
                    raise ValueError(f"Insufficient data. Need at least {lookback} bars, have {len(df)}.")

                data_summary = self.build_loaded_data_summary(df, symbol, timeframe)
                self.dispatch.emit(
                    lambda backend_name=backend_name, data_summary=data_summary, run_id=run_id, auto_run=auto_run: self.on_pipeline_data_ready(
                        backend_name,
                        data_summary,
                        run_id,
                        auto_run=auto_run,
                    )
                )

                self.dispatch.emit(
                    lambda run_id=run_id, auto_run=auto_run: self._is_run_active(run_id, auto_run)
                    and self.update_badges(feed_status="FORECAST")
                )
                self.dispatch.emit(
                    lambda run_id=run_id, auto_run=auto_run: self._is_run_active(run_id, auto_run)
                    and self.append_log(
                        f"[INFO] Forecasting with T={temperature}, top_p={top_p}, samples={sample_count}..."
                    )
                )
                payload = self.run_prediction(
                    df=df,
                    predictor=predictor,
                    backend_name=backend_name,
                    symbol=symbol,
                    timeframe=timeframe,
                    lookback=lookback,
                    pred_len=pred_len,
                    temperature=temperature,
                    top_p=top_p,
                    sample_count=sample_count,
                )
                self.dispatch.emit(
                    lambda payload=payload, predictor=predictor, run_id=run_id, auto_run=auto_run: self.on_prediction_ready(
                        payload,
                        predictor,
                        run_id,
                        auto_run=auto_run,
                    )
                )
            except Exception as exc:
                self.dispatch.emit(
                    lambda exc=exc, run_id=run_id, auto_run=auto_run: self.on_worker_error(
                        "Forecast Pipeline Error",
                        str(exc),
                        run_id,
                        auto_run=auto_run,
                    )
                )

        threading.Thread(target=work, daemon=True).start()

    def generate_sampled_forecast(
        self,
        *,
        predictor,
        x_df,
        x_ts,
        future_ts,
        pred_len,
        temperature,
        top_p,
        sample_count,
    ):
        path_dfs = []
        numeric_cols = ["open", "high", "low", "close", "volume", "amount"]

        for _ in range(max(1, sample_count)):
            pred_df = predictor.predict(
                df=x_df,
                x_timestamp=x_ts,
                y_timestamp=future_ts,
                pred_len=pred_len,
                T=temperature,
                top_p=top_p,
                sample_count=1,
                verbose=False,
            ).reset_index().rename(columns={"index": "timestamps"})
            path_dfs.append(pred_df)

        if len(path_dfs) == 1:
            return path_dfs[0], None

        stacked = np.stack(
            [path_df[numeric_cols].to_numpy(dtype=np.float32) for path_df in path_dfs],
            axis=0,
        )
        mean_df = pd.DataFrame(stacked.mean(axis=0), columns=numeric_cols)
        mean_df["timestamps"] = path_dfs[0]["timestamps"].values

        close_paths = stacked[:, :, numeric_cols.index("close")]
        interval_df = pd.DataFrame(
            {
                "timestamps": path_dfs[0]["timestamps"].values,
                "lower_close": np.quantile(close_paths, 0.10, axis=0),
                "upper_close": np.quantile(close_paths, 0.90, axis=0),
            }
        )
        return mean_df, interval_df

    def run_prediction(
        self,
        *,
        df,
        predictor,
        backend_name,
        symbol,
        timeframe,
        lookback,
        pred_len,
        temperature,
        top_p,
        sample_count,
    ):
        columns = ["open", "high", "low", "close", "volume"]
        context_df = df.tail(lookback).reset_index(drop=True)
        x_df = context_df[columns].copy()
        x_ts = ensure_timestamp_series(context_df["timestamps"])

        step = infer_time_delta(df["timestamps"])
        future_ts = build_future_timestamps(x_ts.iloc[-1], step, pred_len)

        future_pred_df, forecast_interval_df = self.generate_sampled_forecast(
            predictor=predictor,
            x_df=x_df,
            x_ts=x_ts,
            future_ts=future_ts,
            pred_len=pred_len,
            temperature=temperature,
            top_p=top_p,
            sample_count=sample_count,
        )

        last_close = float(x_df["close"].iloc[-1])
        first_pred_close = float(future_pred_df["close"].iloc[0])
        last_pred_close = float(future_pred_df["close"].iloc[-1])
        move_pct = ((last_pred_close - last_close) / last_close) * 100 if last_close else 0.0

        summary_lines = [
            "=== Kronos Futures Forecast ===",
            "",
            f"Model: {backend_name}",
            f"Symbol: {symbol}",
            f"Timeframe: {timeframe}",
            f"Latest candle: {x_ts.iloc[-1]}",
            f"Context candles: {lookback}",
            f"Forecast candles: {pred_len}",
            f"Sampling: T={temperature}, top_p={top_p}, sample_count={sample_count}",
            "",
            "Forward forecast:",
            f"  Future window: {future_ts.iloc[0]} -> {future_ts.iloc[-1]}",
            f"  Last close: {format_price(last_close)}",
            f"  Next close forecast: {format_price(first_pred_close)}",
            f"  Final close forecast: {format_price(last_pred_close)}",
            f"  Forecast move: {move_pct:+.2f}%",
            f"  Forecast low/high range: {format_price(float(future_pred_df['low'].min()))} ~ {format_price(float(future_pred_df['high'].max()))}",
        ]

        if forecast_interval_df is not None:
            summary_lines.extend(
                [
                    f"  MC 10%-90% close band: {format_price(float(forecast_interval_df['lower_close'].min()))} ~ {format_price(float(forecast_interval_df['upper_close'].max()))}",
                    f"  MC 10%-90% final close range: {format_price(float(forecast_interval_df['lower_close'].iloc[-1]))} ~ {format_price(float(forecast_interval_df['upper_close'].iloc[-1]))}",
                ]
            )

        summary_lines.extend(
            [
                "",
                "First forecast closes:",
                future_pred_df[["timestamps", "close"]].head(8).round(2).to_string(index=False),
            ]
        )

        validation_history_df = None
        validation_pred_df = None
        validation_actual_df = None
        mape_text = "N/A"
        validation_subtitle = "Recent validation unavailable"

        validation_segments = build_validation_segments(df, lookback, pred_len, drop_latest_bar=True)
        if validation_segments is not None:
            validation_history_df, validation_actual_df = validation_segments
            val_x_df = validation_history_df[columns].copy()
            val_x_ts = ensure_timestamp_series(validation_history_df["timestamps"])
            val_y_ts = ensure_timestamp_series(validation_actual_df["timestamps"])

            validation_pred_df = predictor.predict(
                df=val_x_df,
                x_timestamp=val_x_ts,
                y_timestamp=val_y_ts,
                pred_len=pred_len,
                T=temperature,
                top_p=top_p,
                sample_count=sample_count,
                verbose=False,
            )
            validation_pred_df = validation_pred_df.reset_index().rename(columns={"index": "timestamps"})

            pred_prices = validation_pred_df["close"].to_numpy()
            actual_prices = validation_actual_df["close"].to_numpy()
            mae = float(np.mean(np.abs(pred_prices - actual_prices)))
            rmse = float(np.sqrt(np.mean((pred_prices - actual_prices) ** 2)))
            mape = float(np.mean(np.abs((pred_prices - actual_prices) / actual_prices)) * 100)
            pred_change = ((pred_prices[-1] - pred_prices[0]) / pred_prices[0]) * 100
            actual_change = ((actual_prices[-1] - actual_prices[0]) / actual_prices[0]) * 100
            mape_text = f"{mape:.2f}%"
            validation_subtitle = (
                f"Validation window: {val_y_ts.iloc[0]} -> {val_y_ts.iloc[-1]} | "
                f"MAE {format_price(mae)} | RMSE {format_price(rmse)}"
            )

            summary_lines.extend(
                [
                    "",
                    "Recent validation on latest closed candles:",
                    f"  Window: {val_y_ts.iloc[0]} -> {val_y_ts.iloc[-1]}",
                    f"  MAE: {format_price(mae)}",
                    f"  RMSE: {format_price(rmse)}",
                    f"  MAPE: {mape:.2f}%",
                    f"  Predicted move: {pred_change:+.2f}%",
                    f"  Actual move: {actual_change:+.2f}%",
                ]
            )
        else:
            summary_lines.extend(
                [
                    "",
                    "Recent validation:",
                    "  Skipped because loaded bars are not enough for lookback + prediction length.",
            ]
        )

        return {
            "summary_text": "\n".join(summary_lines),
            "latest_close": format_price(last_close),
            "forecast_end": format_price(last_pred_close),
            "forecast_move": f"{move_pct:+.2f}%",
            "mape_text": mape_text,
            "subtitle": validation_subtitle,
            "df": df,
            "backend_name": backend_name,
            "symbol": symbol,
            "timeframe": timeframe,
            "context_df": context_df,
            "future_pred_df": future_pred_df,
            "forecast_interval_df": forecast_interval_df,
            "validation_history_df": validation_history_df,
            "validation_pred_df": validation_pred_df,
            "validation_actual_df": validation_actual_df,
        }

    def on_pipeline_data_ready(self, backend_name, data_summary, run_id, *, auto_run=False):
        if not self._is_run_active(run_id, auto_run):
            return
        self.backend_name = backend_name
        self.title_label.setText(f"{data_summary['symbol']} futures forecast")
        self.subtitle_label.setText(
            f"Loaded {data_summary['rows']} candles from Binance futures | {data_summary['timeframe']} | {data_summary['start']} -> {data_summary['end']}"
        )
        self.latest_card.set_content(format_price(data_summary["last_close"]), f"{data_summary['symbol']} latest close")
        self.forecast_card.set_content("--", "Forecast final close")
        self.move_card.set_content("--", "Model is running")
        self.mape_card.set_content("--", "Recent validation")
        self.update_badges(feed_status="LIVE")
        self.append_log(f"[SUCCESS] Model ready via {backend_name}")
        self.append_log(
            f"[SUCCESS] Loaded {data_summary['rows']} bars of {data_summary['symbol']} {data_summary['timeframe']} futures data"
        )

    def on_prediction_ready(self, payload, predictor, run_id, *, auto_run=False):
        if not self._is_run_active(run_id, auto_run):
            return
        self.predictor = predictor
        self.loaded_model_key = self.selected_model_key
        self.df = payload["df"]
        self.backend_name = payload["backend_name"]
        self.current_symbol = payload["symbol"]
        self.current_timeframe = payload["timeframe"]
        self.results_text.setPlainText(payload["summary_text"])
        self.forecast_card.set_content(payload["forecast_end"], "Forecast final close")
        self.move_card.set_content(payload["forecast_move"], "Predicted move from latest close")
        self.mape_card.set_content(payload["mape_text"], payload["subtitle"])
        self.subtitle_label.setText(payload["subtitle"])
        self.forecast_canvas.plot_forecast(
            symbol=self.current_symbol,
            timeframe=self.current_timeframe,
            context_df=payload["context_df"],
            future_pred_df=payload["future_pred_df"],
            forecast_interval_df=payload["forecast_interval_df"],
        )
        self.validation_canvas.plot_validation(
            validation_history_df=payload["validation_history_df"],
            validation_pred_df=payload["validation_pred_df"],
            validation_actual_df=payload["validation_actual_df"],
        )
        self.update_paper_mode(payload)
        self.status_bar.showMessage("Forecast complete")
        self.update_badges(feed_status="READY")
        if auto_run and self.auto_forecast_running:
            self._save_forecast_results(payload)
            self._schedule_next_auto_forecast()
        self.set_busy()

    def on_worker_error(self, title, message, run_id, *, auto_run=False):
        if not self._is_run_active(run_id, auto_run):
            return
        self.status_bar.showMessage(title)
        self.update_badges(feed_status="ALERT")
        self.show_error(title, message)
        self.set_busy()

    def on_work_complete(self):
        if not self.auto_forecast_running:
            self.update_badges(feed_status="IDLE")
            self.set_busy()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("KronosGUI")
    window = KronosGUI()
    window.show()
    window.raise_()
    window.activateWindow()
    if hasattr(app, "exec"):
        sys.exit(app.exec())
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
