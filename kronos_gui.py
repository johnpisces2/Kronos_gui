#!/usr/bin/env python3
"""Kronos BTC prediction GUI for Binance futures data."""

import os
import sys
import threading

import ccxt
import numpy as np
import pandas as pd
from typing import Optional

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if sys.platform == "darwin":
    os.environ.setdefault("QT_MAC_WANTS_LAYER", "1")
    os.environ.setdefault("QT_QPA_PLATFORM", "cocoa")

try:
    from PyQt5.QtCore import Qt, pyqtSignal, QDate
    from PyQt5.QtGui import QColor
    from PyQt5.QtWidgets import (
        QAbstractItemView,
        QApplication,
        QComboBox,
        QDateEdit,
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
        QSplitter,
        QSizePolicy,
        QStatusBar,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
        QScrollArea,
    )
    QT_API = "PyQt5"
except ImportError:
    from PyQt6.QtCore import Qt, pyqtSignal, QDate
    from PyQt6.QtGui import QColor
    from PyQt6.QtWidgets import (
        QAbstractItemView,
        QApplication,
        QComboBox,
        QDateEdit,
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
        QSplitter,
        QSizePolicy,
        QStatusBar,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
        QScrollArea,
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

from paper_strategy import (
    PaperPosition,
    PaperStrategyConfig,
    build_entry_decision,
    build_exit_decision,
    build_signal_snapshot,
)
from execution import ExecutionEngine
from backtester import Backtester

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


def format_quantity(value):
    return f"{value:,.4f}".rstrip("0").rstrip(".")


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
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(1)

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
        self.figure = Figure(figsize=(8.8, 4.8), facecolor="#151515")
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
        paper_trades=None,
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
        filtered_trades = [
            trade for trade in (paper_trades or [])
            if trade.get("symbol") == symbol and trade.get("timeframe") == timeframe
        ]
        trade_styles = {
            "enter_long": {"marker": "^", "color": "#22c55e", "label": "Enter long"},
            "enter_short": {"marker": "v", "color": "#ef4444", "label": "Enter short"},
            "exit_long": {"marker": "X", "color": "#f59e0b", "label": "Exit long"},
            "exit_short": {"marker": "X", "color": "#38bdf8", "label": "Exit short"},
        }
        for action, style in trade_styles.items():
            action_trades = [trade for trade in filtered_trades if trade["action"] == action]
            if not action_trades:
                continue
            self.price_ax.scatter(
                [trade["time"] for trade in action_trades],
                [trade["price"] for trade in action_trades],
                marker=style["marker"],
                s=58,
                color=style["color"],
                edgecolors="#fafafa",
                linewidths=0.6,
                label=style["label"],
                zorder=6,
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
        self.figure = Figure(figsize=(8.6, 2.4), facecolor="#151515")
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
        self.figure = Figure(figsize=(8.6, 2.8), facecolor="#151515")
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

    _PAPER_ACTION_LABELS = {
        "enter_long": "ENTER LONG",
        "enter_short": "ENTER SHORT",
        "exit_long": "EXIT LONG",
        "exit_short": "EXIT SHORT",
        "hold": "HOLD",
        "no_action": "NO ACTION",
    }

    _PAPER_REASON_LABELS = {
        "long_signal_confirmed": "Long entry: forecast & validation trend confirm.",
        "short_signal_confirmed": "Short entry: forecast & validation trend confirm.",
        "entry_conditions_not_met": "Forecast edge or validation not aligned.",
        "stop_loss_hit": "Stop loss triggered.",
        "take_profit_hit": "Take profit triggered!",
        "long_signal_invalidated": "Long invalidated: trend reversed.",
        "short_signal_invalidated": "Short invalidated: trend reversed.",
        "long_position_still_valid": "Long valid: trend intact.",
        "short_position_still_valid": "Short valid: trend intact.",
        "missing_validation_data": "Validation data not ready.",
        "unknown_position_side": "Unknown position side.",
    }

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
        self.execution = ExecutionEngine()
        self.execution_mode = "paper"
        self.latest_market_df = None

        self.init_ui()
        self.show()

    @property
    def paper_position(self):
        return self.execution.position

    @property
    def paper_initial_equity(self):
        return self.execution.initial_equity

    @paper_initial_equity.setter
    def paper_initial_equity(self, value):
        self.execution.initial_equity = value

    @property
    def paper_realized_equity(self):
        return self.execution.realized_equity

    @paper_realized_equity.setter
    def paper_realized_equity(self, value):
        self.execution.realized_equity = value

    @property
    def paper_trade_history(self):
        return self.execution.trade_history

    @property
    def paper_equity_history(self):
        return self.execution.equity_history

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
        self.setGeometry(80, 60, 1600, 1000)
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
                font-size: 18px;
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
                font-size: 15px;
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
                border-radius: 4px;
                color: #d4d4d8;
                padding: 2px 6px;
                font-size: 9px;
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
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(10, 8, 10, 8)
        header_layout.setSpacing(12)

        left_panel = QVBoxLayout()
        left_panel.setSpacing(2)
        left_panel.setContentsMargins(0, 0, 0, 0)

        self.title_label = QLabel("Kronos BTC Forecast Terminal")
        self.title_label.setObjectName("titleLabel")
        left_panel.addWidget(self.title_label)

        self.subtitle_label = QLabel("Load engine → fetch Binance futures → forecast")
        self.subtitle_label.setObjectName("subtitleLabel")
        self.subtitle_label.setStyleSheet("color: #71717a; font-size: 10px;")
        left_panel.addWidget(self.subtitle_label)

        badge_row = QHBoxLayout()
        badge_row.setSpacing(6)
        self.symbol_badge = TerminalBadge("BTC/USDT")
        self.timeframe_badge = TerminalBadge("4H")
        self.model_badge = TerminalBadge("--")
        self.feed_badge = TerminalBadge("IDLE")
        badge_row.addWidget(self.symbol_badge)
        badge_row.addWidget(self.timeframe_badge)
        badge_row.addWidget(self.model_badge)
        badge_row.addWidget(self.feed_badge)
        left_panel.addLayout(badge_row)
        header_layout.addLayout(left_panel, 1)

        metrics_layout = QHBoxLayout()
        metrics_layout.setSpacing(8)
        metrics_layout.setContentsMargins(0, 0, 0, 0)

        self.latest_card = MetricCard("Last", "#38bdf8")
        self.forecast_card = MetricCard("Fcst End", "#f59e0b")
        self.move_card = MetricCard("Move", "#22c55e")
        self.mape_card = MetricCard("MAPE", "#a78bfa")

        metrics_layout.addWidget(self.latest_card)
        metrics_layout.addWidget(self.forecast_card)
        metrics_layout.addWidget(self.move_card)
        metrics_layout.addWidget(self.mape_card)
        header_layout.addLayout(metrics_layout, 2)
        layout.addWidget(header)

        self.main_tabs = QTabWidget()
        self.main_tabs.setStyleSheet("QTabWidget::pane { border: none; }")
        self.main_tabs.addTab(self.create_market_tab(), "Market")
        self.main_tabs.addTab(self.create_backtest_tab(), "Backtest")
        self.main_tabs.addTab(self.create_execution_tab(), "Execution")
        
        layout.addWidget(self.main_tabs)

        self.set_metric_defaults()
        return container

    def set_metric_defaults(self):
        self.latest_card.set_content("--", "Latest market close")
        self.forecast_card.set_content("--", "Forecast final close")
        self.move_card.set_content("--", "Predicted direction")
        self.mape_card.set_content("--", "Recent validation")

    def create_market_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.market_splitter = QSplitter(Qt.Vertical)

        forecast_group = QGroupBox("Forecast")
        forecast_layout = QVBoxLayout(forecast_group)
        forecast_layout.setContentsMargins(8, 8, 8, 8)
        forecast_layout.setSpacing(4)
        self.forecast_canvas = PriceChartCanvas(chart_type="price")
        self.forecast_canvas.setMinimumHeight(200)
        forecast_layout.addWidget(self.forecast_canvas)
        self.market_splitter.addWidget(forecast_group)

        validation_group = QGroupBox("Validation")
        validation_layout = QVBoxLayout(validation_group)
        validation_layout.setContentsMargins(8, 8, 8, 8)
        validation_layout.setSpacing(4)
        self.validation_canvas = PriceChartCanvas(chart_type="validation")
        self.validation_canvas.setMinimumHeight(160)
        validation_layout.addWidget(self.validation_canvas)
        self.market_splitter.addWidget(validation_group)

        self.market_splitter.setStretchFactor(0, 3)
        self.market_splitter.setStretchFactor(1, 2)
        layout.addWidget(self.market_splitter)

        return tab

    def create_execution_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        top_controls = QFrame()
        top_controls.setObjectName("metricCard")
        top_layout = QHBoxLayout(top_controls)
        top_layout.setContentsMargins(10, 6, 10, 6)
        top_layout.setSpacing(10)

        top_layout.addWidget(QLabel("Mode"))
        self.execution_mode_combo = QComboBox()
        self.execution_mode_combo.addItem("Paper", "paper")
        self.execution_mode_combo.addItem("Testnet (Soon)", "testnet")
        self.execution_mode_combo.addItem("Live (Locked)", "live")
        self.execution_mode_combo.setCurrentIndex(0)
        self.execution_mode_combo.currentIndexChanged.connect(self.on_execution_mode_changed)
        top_layout.addWidget(self.execution_mode_combo)

        self.paper_enable_checkbox = QCheckBox("Enable")
        self.paper_enable_checkbox.setChecked(self.execution.enabled)
        self.paper_enable_checkbox.toggled.connect(self.on_paper_trading_toggled)
        top_layout.addWidget(self.paper_enable_checkbox)

        top_layout.addWidget(QLabel("Equity"))
        self.paper_initial_equity_spin = QDoubleSpinBox()
        self.paper_initial_equity_spin.setRange(1000.0, 100000000.0)
        self.paper_initial_equity_spin.setDecimals(0)
        self.paper_initial_equity_spin.setSingleStep(1000.0)
        self.paper_initial_equity_spin.setValue(self.execution.initial_equity)
        self.paper_initial_equity_spin.valueChanged.connect(self.on_paper_initial_equity_changed)
        self.paper_initial_equity_spin.setMaximumWidth(110)
        top_layout.addWidget(self.paper_initial_equity_spin)

        self.paper_reset_btn = QPushButton("Reset")
        self.paper_reset_btn.clicked.connect(self.reset_paper_account)
        top_layout.addWidget(self.paper_reset_btn)

        top_layout.addSpacing(20)

        top_layout.addWidget(QLabel("Qty"))
        self.paper_order_qty_spin = QDoubleSpinBox()
        self.paper_order_qty_spin.setDecimals(4)
        self.paper_order_qty_spin.setRange(0.0001, 1000000.0)
        self.paper_order_qty_spin.setSingleStep(0.001)
        self.paper_order_qty_spin.setValue(self.execution.order_quantity)
        self.paper_order_qty_spin.valueChanged.connect(self.on_paper_order_value_changed)
        self.paper_order_qty_spin.setMaximumWidth(90)
        top_layout.addWidget(self.paper_order_qty_spin)

        top_layout.addWidget(QLabel("Lev"))
        self.paper_leverage_spin = QSpinBox()
        self.paper_leverage_spin.setRange(1, 125)
        self.paper_leverage_spin.setValue(self.execution.order_leverage)
        self.paper_leverage_spin.valueChanged.connect(self.on_paper_order_value_changed)
        self.paper_leverage_spin.setMaximumWidth(55)
        top_layout.addWidget(self.paper_leverage_spin)

        self.paper_risk_checkbox = QCheckBox("Risk%")
        self.paper_risk_checkbox.toggled.connect(self.on_paper_risk_toggled)
        self.paper_risk_checkbox.setChecked(self.execution.use_risk_fraction)
        top_layout.addWidget(self.paper_risk_checkbox)

        self.paper_risk_spin = QDoubleSpinBox()
        self.paper_risk_spin.setDecimals(2)
        self.paper_risk_spin.setRange(0.1, 50.0)
        self.paper_risk_spin.setSuffix("%")
        self.paper_risk_spin.setValue(self.execution.risk_fraction * 100)
        self.paper_risk_spin.valueChanged.connect(self.on_paper_risk_changed)
        self.paper_risk_spin.setMaximumWidth(70)
        top_layout.addWidget(self.paper_risk_spin)

        top_layout.addWidget(QLabel("Notional"))
        self.paper_notional_value = QLabel("--")
        self.paper_notional_value.setStyleSheet("color: #f5f5f5; font-weight: 600;")
        top_layout.addWidget(self.paper_notional_value)

        top_layout.addWidget(QLabel("Margin"))
        self.paper_margin_value = QLabel("--")
        self.paper_margin_value.setStyleSheet("color: #f5f5f5; font-weight: 600;")
        top_layout.addWidget(self.paper_margin_value)

        top_layout.addStretch()
        layout.addWidget(top_controls)

        metrics_layout = QHBoxLayout()
        metrics_layout.setSpacing(6)

        self.paper_signal_card = MetricCard("Signal", "#38bdf8")
        self.paper_position_card = MetricCard("Position", "#f59e0b")
        self.paper_pnl_card = MetricCard("PnL", "#22c55e")
        self.paper_trades_card = MetricCard("Trades", "#f59e0b")
        self.paper_equity_card = MetricCard("Equity", "#22c55e")

        for card in [self.paper_signal_card, self.paper_position_card, self.paper_pnl_card,
                     self.paper_trades_card, self.paper_equity_card]:
            card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            metrics_layout.addWidget(card)
        layout.addLayout(metrics_layout)

        position_group = QGroupBox("Position Watch")
        position_layout = QVBoxLayout(position_group)
        position_layout.setContentsMargins(6, 6, 6, 6)
        position_layout.setSpacing(4)
        self.paper_position_table = QTableWidget(1, 11)
        self.paper_position_table.setHorizontalHeaderLabels(
            ["Side", "Entry", "Qty", "Lev", "EntryPx", "Mark", "Notional", "Margin", "Stop", "TP", "Unreal"]
        )
        self.paper_position_table.verticalHeader().setVisible(False)
        self.paper_position_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.paper_position_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.paper_position_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.paper_position_table.setRowHeight(0, 28)
        position_layout.addWidget(self.paper_position_table)
        layout.addWidget(position_group)

        trade_group = QGroupBox("Trade Ledger")
        trade_layout = QVBoxLayout(trade_group)
        trade_layout.setContentsMargins(6, 6, 6, 6)
        trade_layout.setSpacing(4)
        self.paper_trade_table = QTableWidget(0, 11)
        self.paper_trade_table.setHorizontalHeaderLabels(
            ["Time", "Action", "Qty", "Lev", "Price", "Notional", "Stop", "TP", "P&L", "Fcst", "Reason"]
        )
        self.paper_trade_table.verticalHeader().setVisible(False)
        self.paper_trade_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.paper_trade_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.paper_trade_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        self.paper_trade_table.setMinimumHeight(100)
        self.paper_trade_table.verticalHeader().setDefaultSectionSize(24)
        trade_layout.addWidget(self.paper_trade_table)
        layout.addWidget(trade_group)

        self.paper_status_label = QLabel("")
        self.paper_status_label.setWordWrap(True)
        self.paper_status_label.setStyleSheet("color: #71717a; font-size: 10px;")
        layout.addWidget(self.paper_status_label)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMinimumHeight(120)
        self.results_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        self.results_text.setPlaceholderText("Log...")
        layout.addWidget(self.results_text)

        self.set_paper_defaults()
        return tab

    def create_backtest_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        controls = QFrame()
        controls.setObjectName("metricCard")
        ctrl_layout = QGridLayout(controls)
        ctrl_layout.setContentsMargins(10, 8, 10, 8)
        ctrl_layout.setHorizontalSpacing(12)
        ctrl_layout.setVerticalSpacing(6)

        ctrl_layout.addWidget(QLabel("Start"), 0, 0)
        self.bt_start_date = QDateEdit()
        self.bt_start_date.setCalendarPopup(True)
        self.bt_start_date.setDisplayFormat("yyyy-MM-dd")
        self.bt_start_date.setMinimumWidth(110)
        two_years_ago = QDate.currentDate().addYears(-2)
        self.bt_start_date.setDate(two_years_ago)
        ctrl_layout.addWidget(self.bt_start_date, 0, 1)

        ctrl_layout.addWidget(QLabel("Initial Equity"), 0, 2)
        self.bt_equity_spin = QDoubleSpinBox()
        self.bt_equity_spin.setRange(1000.0, 100000000.0)
        self.bt_equity_spin.setDecimals(0)
        self.bt_equity_spin.setSingleStep(1000.0)
        self.bt_equity_spin.setValue(1000.0)
        ctrl_layout.addWidget(self.bt_equity_spin, 0, 3)

        ctrl_layout.addWidget(QLabel("Slippage"), 0, 4)
        self.bt_slippage_spin = QDoubleSpinBox()
        self.bt_slippage_spin.setRange(0.0, 5.0)
        self.bt_slippage_spin.setDecimals(1)
        self.bt_slippage_spin.setSuffix("%")
        self.bt_slippage_spin.setValue(0.1)
        ctrl_layout.addWidget(self.bt_slippage_spin, 0, 5)

        ctrl_layout.addWidget(QLabel("Leverage"), 0, 6)
        self.bt_leverage_spin = QSpinBox()
        self.bt_leverage_spin.setRange(1, 125)
        self.bt_leverage_spin.setValue(5)
        ctrl_layout.addWidget(self.bt_leverage_spin, 0, 7)

        self.bt_run_btn = QPushButton("Run Backtest")
        self.bt_run_btn.setStyleSheet("""
            QPushButton {
                background: #16a34a;
                color: white;
                font-weight: 600;
                padding: 6px 16px;
            }
            QPushButton:hover { background: #15803d; }
        """)
        self.bt_run_btn.clicked.connect(self.run_backtest)
        ctrl_layout.addWidget(self.bt_run_btn, 0, 8)

        ctrl_layout.addWidget(QLabel("End"), 1, 0)
        self.bt_end_date = QDateEdit()
        self.bt_end_date.setCalendarPopup(True)
        self.bt_end_date.setDisplayFormat("yyyy-MM-dd")
        self.bt_end_date.setMinimumWidth(110)
        self.bt_end_date.setDate(QDate.currentDate())
        ctrl_layout.addWidget(self.bt_end_date, 1, 1)

        ctrl_layout.addWidget(QLabel("Fee Rate"), 1, 2)
        self.bt_fee_spin = QDoubleSpinBox()
        self.bt_fee_spin.setRange(0.0, 1.0)
        self.bt_fee_spin.setDecimals(2)
        self.bt_fee_spin.setSuffix("%")
        self.bt_fee_spin.setValue(0.05)
        ctrl_layout.addWidget(self.bt_fee_spin, 1, 3)

        ctrl_layout.addWidget(QLabel("Risk%"), 1, 4)
        self.bt_risk_spin = QDoubleSpinBox()
        self.bt_risk_spin.setRange(0.1, 50.0)
        self.bt_risk_spin.setDecimals(0)
        self.bt_risk_spin.setSuffix("%")
        self.bt_risk_spin.setValue(8.0)
        ctrl_layout.addWidget(self.bt_risk_spin, 1, 5)

        self.bt_cancel_btn = QPushButton("Cancel")
        self.bt_cancel_btn.setEnabled(False)
        self.bt_cancel_btn.setStyleSheet("""
            QPushButton {
                background: #dc2626;
                color: white;
                font-weight: 600;
                padding: 6px 16px;
            }
            QPushButton:hover { background: #b91c1c; }
            QPushButton:disabled { background: #525252; color: #a1a1a1; }
        """)
        self.bt_cancel_btn.clicked.connect(self.cancel_backtest)
        ctrl_layout.addWidget(self.bt_cancel_btn, 1, 8)

        self._bt_running = False
        self._bt_cancelled = False

        layout.addWidget(controls)

        self.bt_metrics_layout = QGridLayout()
        self.bt_metrics_layout.setSpacing(8)
        self.bt_metrics_layout.setContentsMargins(0, 4, 0, 4)

        self.bt_total_pnl_card = MetricCard("Total P&L", "#22c55e")
        self.bt_win_rate_card = MetricCard("Win Rate", "#38bdf8")
        self.bt_trades_card = MetricCard("Trades", "#f59e0b")
        self.bt_sharpe_card = MetricCard("Sharpe", "#a855f7")
        self.bt_max_dd_card = MetricCard("Max DD", "#ef4444")
        self.bt_return_card = MetricCard("Return", "#22c55e")

        self.bt_metrics_layout.addWidget(self.bt_total_pnl_card, 0, 0)
        self.bt_metrics_layout.addWidget(self.bt_win_rate_card, 0, 1)
        self.bt_metrics_layout.addWidget(self.bt_trades_card, 0, 2)
        self.bt_metrics_layout.addWidget(self.bt_sharpe_card, 0, 3)
        self.bt_metrics_layout.addWidget(self.bt_max_dd_card, 0, 4)
        self.bt_metrics_layout.addWidget(self.bt_return_card, 0, 5)

        self.bt_profit_factor_card = MetricCard("Profit Factor", "#22c55e")
        self.bt_avg_win_card = MetricCard("Avg Win", "#22c55e")
        self.bt_avg_loss_card = MetricCard("Avg Loss", "#ef4444")
        self.bt_sortino_card = MetricCard("Sortino", "#a855f7")
        self.bt_time_in_market_card = MetricCard("Time In Market", "#38bdf8")
        self.bt_avg_trade_card = MetricCard("Avg Trade", "#38bdf8")

        self.bt_metrics_layout.addWidget(self.bt_profit_factor_card, 1, 0)
        self.bt_metrics_layout.addWidget(self.bt_avg_win_card, 1, 1)
        self.bt_metrics_layout.addWidget(self.bt_avg_loss_card, 1, 2)
        self.bt_metrics_layout.addWidget(self.bt_sortino_card, 1, 3)
        self.bt_metrics_layout.addWidget(self.bt_time_in_market_card, 1, 4)
        self.bt_metrics_layout.addWidget(self.bt_avg_trade_card, 1, 5)

        for card in [self.bt_total_pnl_card, self.bt_win_rate_card, self.bt_trades_card,
                     self.bt_sharpe_card, self.bt_max_dd_card, self.bt_return_card,
                     self.bt_profit_factor_card, self.bt_avg_win_card, self.bt_avg_loss_card,
                     self.bt_sortino_card, self.bt_time_in_market_card, self.bt_avg_trade_card]:
            card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        layout.addLayout(self.bt_metrics_layout)

        trades_group = QGroupBox("Trade History")
        trades_layout = QVBoxLayout(trades_group)
        trades_layout.setContentsMargins(6, 6, 6, 6)

        self.bt_trade_table = QTableWidget(0, 10)
        self.bt_trade_table.setHorizontalHeaderLabels([
            "Entry Time", "Exit Time", "Side", "Entry", "Exit", "Qty", "Lev",
            "P&L", "Return %", "Reason"
        ])
        self.bt_trade_table.verticalHeader().setVisible(False)
        self.bt_trade_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.bt_trade_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
        self.bt_trade_table.setMinimumHeight(200)
        self.bt_trade_table.horizontalHeader().setStretchLastSection(False)
        for col in range(10):
            self.bt_trade_table.horizontalHeader().setSectionResizeMode(col, QHeaderView.Stretch)
        self.bt_trade_table.setColumnWidth(2, 50)
        self.bt_trade_table.setColumnWidth(6, 50)
        trades_layout.addWidget(self.bt_trade_table)
        layout.addWidget(trades_group)

        self.bt_results_text = QTextEdit()
        self.bt_results_text.setReadOnly(True)
        self.bt_results_text.setMinimumHeight(150)
        self.bt_results_text.setPlaceholderText("Backtest results will appear here...")
        layout.addWidget(self.bt_results_text)

        self._init_backtest_metrics()
        return tab

    def _init_backtest_metrics(self):
        self.bt_total_pnl_card.set_content("--", "Total profit/loss")
        self.bt_win_rate_card.set_content("--", "Win rate %")
        self.bt_trades_card.set_content("--", "Total trades")
        self.bt_sharpe_card.set_content("--", "Sharpe ratio")
        self.bt_max_dd_card.set_content("--", "Max drawdown")
        self.bt_return_card.set_content("--", "Total return %")
        self.bt_profit_factor_card.set_content("--", "Gross profit / loss")
        self.bt_avg_win_card.set_content("--", "Average win $")
        self.bt_avg_loss_card.set_content("--", "Average loss $")
        self.bt_sortino_card.set_content("--", "Sortino ratio")
        self.bt_time_in_market_card.set_content("--", "Time in market %")
        self.bt_avg_trade_card.set_content("--", "Average trade return")

    def cancel_backtest(self):
        self._bt_cancelled = True
        self.bt_results_text.append("[INFO] Cancelling backtest...")

    def run_backtest(self):
        if self._bt_running:
            return

        self._bt_running = True
        self._bt_cancelled = False
        self.bt_run_btn.setEnabled(False)
        self.bt_cancel_btn.setEnabled(True)
        self.bt_results_text.append("[INFO] Starting backtest...")

        try:
            initial_equity = self.bt_equity_spin.value()
            fee_rate = self.bt_fee_spin.value() / 100.0
            slippage_pct = self.bt_slippage_spin.value() / 100.0

            start_date = self.bt_start_date.date()
            end_date = self.bt_end_date.date()

            if end_date <= start_date:
                self.bt_results_text.append("[ERROR] End date must be after start date.")
                self._bt_running = False
                self.bt_run_btn.setEnabled(True)
                self.bt_cancel_btn.setEnabled(False)
                return

            if end_date > QDate.currentDate():
                self.bt_results_text.append("[ERROR] End date cannot be in the future.")
                return

            start_str = start_date.toString("yyyy-MM-dd")
            end_str = end_date.toString("yyyy-MM-dd")

            if initial_equity <= 0:
                self.bt_results_text.append("[ERROR] Initial equity must be greater than 0.")
                return

            self.bt_results_text.append(f"[INFO] Fetching {start_str} to {end_str} from Binance...")

            exchange = ccxt.binance({
                "enableRateLimit": True,
                "options": {"defaultType": "future"},
            })

            symbol = self.symbol_combo.currentText()
            timeframe = self.tf_combo.currentText()

            since = int(pd.Timestamp(start_str, tz="Asia/Shanghai").timestamp() * 1000)
            now = int(pd.Timestamp(end_str + " 23:59:59", tz="Asia/Shanghai").timestamp() * 1000)

            all_ohlcv = []
            current_since = since
            while current_since < now:
                try:
                    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=1500)
                    if not ohlcv:
                        break
                    all_ohlcv.extend(ohlcv)
                    last_ts = ohlcv[-1][0]
                    if last_ts >= now:
                        break
                    current_since = last_ts + 1
                except Exception as e:
                    self.bt_results_text.append(f"[WARN] Fetch error: {e}")
                    break

            if not all_ohlcv:
                self.bt_results_text.append("[ERROR] No data fetched from Binance.")
                return

            df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamps"] = to_display_timestamp_series(
                pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            )
            df = df.drop(columns=["timestamp"]).reset_index(drop=True)
            df["amount"] = df["volume"] * df["close"] * 0.0001

            self.bt_results_text.append(f"[INFO] Downloaded {len(df)} candles")

            lookback = self.lookback_spin.value()
            pred_len = self.predlen_spin.value()
            temperature = self.temp_spin.value()
            top_p = self.topp_spin.value()
            sample_count = self.sample_spin.value()

            context_df = df.copy()
            total_len = len(context_df)

            val_start_idx = max(lookback, int(total_len * 0.7))
            val_history_df = context_df.iloc[:val_start_idx].copy()
            val_actual_df = context_df.iloc[val_start_idx:].copy()

            self.bt_results_text.append(f"[INFO] Loading model for backtest...")
            predictor, backend_name = self.ensure_predictor_loaded()
            self.bt_results_text.append(f"[INFO] Model loaded: {backend_name}")

            def predictor_fn(context_df_for_pred, temperature=temperature, top_p=top_p, sample_count=sample_count):
                columns = ["open", "high", "low", "close", "volume"]
                x_df = context_df_for_pred[columns].copy()
                x_ts = ensure_timestamp_series(context_df_for_pred["timestamps"])

                step = infer_time_delta(context_df_for_pred["timestamps"])
                future_ts = build_future_timestamps(x_ts.iloc[-1], step, pred_len)

                future_pred_df, _ = self.generate_sampled_forecast(
                    predictor=predictor,
                    x_df=x_df,
                    x_ts=x_ts,
                    future_ts=future_ts,
                    pred_len=pred_len,
                    temperature=temperature,
                    top_p=top_p,
                    sample_count=sample_count,
                )
                return {"future_pred_df": future_pred_df}

            forecast_params = {
                "lookback": lookback,
                "pred_len": pred_len,
                "temperature": temperature,
                "top_p": top_p,
                "sample_count": sample_count,
            }

            backtester = Backtester(
                initial_equity=initial_equity,
                fee_rate=fee_rate,
                slippage_pct=slippage_pct,
            )

            total_validation_bars = len(context_df) - lookback
            self.bt_results_text.append(f"[INFO] Backtest: {lookback} lookback, {len(context_df)} total bars, {total_validation_bars} to predict")

            def progress_callback(current, total):
                if self._bt_cancelled:
                    return False
                if current % 50 == 0:
                    self.bt_results_text.append(f"[PROGRESS] {current}/{total} bars processed...")
                return True

            results = backtester.run(
                context_df=context_df,
                predictor_fn=predictor_fn,
                forecast_params=forecast_params,
                debug=True,
                progress_callback=progress_callback,
            )

            if self._bt_cancelled:
                self.bt_results_text.append("[INFO] Backtest cancelled by user.")
            else:
                self._display_backtest_results(results)

        except Exception as e:
            self.bt_results_text.append(f"[ERROR] Backtest failed: {str(e)}")
        finally:
            self._bt_running = False
            self.bt_run_btn.setEnabled(True)
            self.bt_cancel_btn.setEnabled(False)

    def _display_backtest_results(self, results):
        self.bt_total_pnl_card.set_content(
            f"${results.total_pnl:,.2f}",
            f"{results.total_return_pct:+.2f}% return"
        )

        self.bt_win_rate_card.set_content(
            f"{results.win_rate * 100:.1f}%",
            f"{results.winning_trades}W / {results.losing_trades}L"
        )

        self.bt_trades_card.set_content(
            str(results.total_trades),
            f"Closed trades"
        )

        self.bt_sharpe_card.set_content(
            f"{results.sharpe_ratio:.2f}",
            "Sharpe ratio"
        )

        self.bt_max_dd_card.set_content(
            f"${results.max_drawdown:,.2f}",
            f"{results.max_drawdown_pct * 100:.2f}% DD"
        )

        self.bt_return_card.set_content(
            f"{results.total_return_pct:+.2f}%",
            f"Total return"
        )

        self.bt_profit_factor_card.set_content(
            f"{results.profit_factor:.2f}",
            "Profit factor"
        )

        self.bt_avg_win_card.set_content(
            f"${results.avg_win:,.2f}",
            "Average win"
        )

        self.bt_avg_loss_card.set_content(
            f"${results.avg_loss:,.2f}",
            "Average loss"
        )

        self.bt_sortino_card.set_content(
            f"{results.sortino_ratio:.2f}",
            "Sortino ratio"
        )

        self.bt_time_in_market_card.set_content(
            f"{results.time_in_market_pct * 100:.1f}%",
            "Time in market"
        )

        self.bt_avg_trade_card.set_content(
            f"{results.avg_trade_return * 100:+.2f}%",
            "Avg trade return"
        )

        self.bt_trade_table.setRowCount(len(results.trades))
        for row, trade in enumerate(results.trades):
            self.bt_trade_table.setItem(row, 0, QTableWidgetItem(str(trade.entry_time.strftime("%Y-%m-%d %H:%M"))))
            self.bt_trade_table.setItem(row, 1, QTableWidgetItem(str(trade.exit_time.strftime("%Y-%m-%d %H:%M"))))
            self.bt_trade_table.setItem(row, 2, QTableWidgetItem(trade.side.upper()))
            self.bt_trade_table.setItem(row, 3, QTableWidgetItem(f"${trade.entry_price:,.2f}"))
            self.bt_trade_table.setItem(row, 4, QTableWidgetItem(f"${trade.exit_price:,.2f}"))
            self.bt_trade_table.setItem(row, 5, QTableWidgetItem(f"{trade.quantity:.4f}"))
            self.bt_trade_table.setItem(row, 6, QTableWidgetItem(f"{trade.leverage}x"))
            pnl_color = "#22c55e" if trade.pnl_amount >= 0 else "#ef4444"
            self.bt_trade_table.setItem(row, 7, QTableWidgetItem(f"${trade.pnl_amount:,.2f}"))
            self.bt_trade_table.item(row, 7).setForeground(QColor(pnl_color))
            self.bt_trade_table.setItem(row, 8, QTableWidgetItem(f"{trade.return_pct * 100:+.2f}%"))
            self.bt_trade_table.setItem(row, 9, QTableWidgetItem(trade.reason))

        summary = f"""
========== BACKTEST RESULTS ==========

Initial Equity: ${self.bt_equity_spin.value():,.0f}
Fee Rate: {self.bt_fee_spin.value():.2f}%
Slippage: {self.bt_slippage_spin.value():.3f}%

--- Performance ---
Total Trades: {results.total_trades}
Win Rate: {results.win_rate * 100:.2f}%
Total P&L: ${results.total_pnl:,.2f}
Total Return: {results.total_return_pct:.2f}%

--- Risk Metrics ---
Max Drawdown: ${results.max_drawdown:,.2f} ({results.max_drawdown_pct * 100:.2f}%)
Sharpe Ratio: {results.sharpe_ratio:.4f}
Sortino Ratio: {results.sortino_ratio:.4f}
Profit Factor: {results.profit_factor:.4f}
Time in Market: {results.time_in_market_pct * 100:.1f}%

--- Trade Stats ---
Avg Win: ${results.avg_win:,.2f}
Avg Loss: ${results.avg_loss:,.2f}
Avg Trade Return: {results.avg_trade_return * 100:.4f}%

--- Costs ---
Total Fees: ${results.total_fees:,.2f}
Total Slippage: ${results.total_slippage:,.2f}
=========================================
"""
        self.bt_results_text.setText(summary)

    def set_paper_defaults(self):
        if hasattr(self, "paper_signal_card"):
            self.paper_signal_card.set_content("WAIT", "Waiting for forecast")
        if hasattr(self, "paper_position_card"):
            self.paper_position_card.set_content("FLAT", "No position")
        if hasattr(self, "paper_pnl_card"):
            self.paper_pnl_card.set_content("$0", "+0.00% realized")
        if hasattr(self, "paper_trades_card"):
            self.paper_trades_card.set_content("0", "Closed trades")
        if hasattr(self, "paper_status_label"):
            self.paper_status_label.setText("Paper mode idle - run forecast to begin.")
        if hasattr(self, "paper_position_table"):
            for col in range(self.paper_position_table.columnCount()):
                self.paper_position_table.setItem(0, col, QTableWidgetItem("--"))
        if hasattr(self, "paper_trade_table"):
            self.paper_trade_table.setRowCount(0)
        if hasattr(self, "paper_equity_card"):
            self.paper_equity_card.set_content(format_price(self.execution.initial_equity), "+0.00%")
        self.update_execution_order_preview()

    def on_paper_trading_toggled(self, checked):
        self.execution.enabled = bool(checked)
        if self.paper_trading_active():
            self.paper_status_label.setText("Paper trading enabled. The next completed forecast candle can open or close local positions.")
        elif self.execution.enabled:
            self.paper_status_label.setText("Paper trading is armed, but only Local Paper mode can execute right now.")
        else:
            self.execution.last_decision = None
            self.paper_status_label.setText("Paper trading disabled. Signals are visible, but no local entries or exits will be simulated.")
        self.render_paper_mode()

    def on_execution_mode_changed(self, index):
        self.execution_mode = self.execution_mode_combo.itemData(index) or "paper"
        self.execution.mode = self.execution_mode
        self.update_execution_order_preview()
        self.render_paper_mode()

    def on_paper_order_value_changed(self, value):
        del value
        self.execution.order_quantity = float(self.paper_order_qty_spin.value())
        self.execution.order_leverage = int(self.paper_leverage_spin.value())
        self.update_execution_order_preview()

    def on_paper_risk_toggled(self, checked):
        self.execution.use_risk_fraction = checked
        self.paper_order_qty_spin.setVisible(not checked)
        self.paper_leverage_spin.setVisible(not checked)
        self.update_execution_order_preview()

    def on_paper_risk_changed(self, value):
        self.execution.risk_fraction = float(value) / 100.0
        self.update_execution_order_preview()
        if self.execution.last_snapshot is not None:
            self.render_paper_mode()

    def on_paper_initial_equity_changed(self, value):
        value = float(value)
        if value <= 0:
            return
        if abs(value - self.execution.initial_equity) < 1e-9:
            return
        self.execution.initial_equity = value
        self.reset_paper_account(reason=f"Paper account reset with initial equity {format_price(value)}.")

    def reset_paper_account(self, checked=False, *, reason=None):
        self.execution.reset()
        self.latest_market_df = None
        self.set_paper_defaults()
        if hasattr(self, "paper_status_label"):
            self.paper_status_label.setText(
                reason or f"Paper account reset. Starting equity {format_price(self.execution.initial_equity)}."
            )
        self.update_execution_order_preview()

    def paper_trading_active(self):
        return self.execution.trading_active()

    def current_execution_price(self):
        if self.execution.last_snapshot is not None:
            return float(self.execution.last_snapshot.current_price)
        if self.execution.position is not None:
            return float(self.execution.position.entry_price)
        if self.latest_market_df is not None and not self.latest_market_df.empty:
            return float(self.latest_market_df["close"].iloc[-1])
        return None

    def current_order_notional(self, price=None, quantity=None):
        mark_price = self.current_execution_price() if price is None else float(price)
        if mark_price is None:
            return None
        qty = self.execution.current_order_quantity() if quantity is None else float(quantity)
        return mark_price * qty

    def current_order_margin(self, price=None, quantity=None, leverage=None):
        notional = self.current_order_notional(price=price, quantity=quantity)
        if notional is None:
            return None
        lev = self.execution.current_order_leverage() if leverage is None else int(leverage)
        return notional / max(1, lev)

    def describe_paper_action(self, action):
        return self.execution.describe_action(action)

    def describe_paper_reason(self, reason):
        return self.execution.describe_reason(reason)

    def current_paper_equity(self, unrealized_amount=0.0):
        return self.execution.current_equity(unrealized_amount)

    def current_realized_return_pct(self):
        return self.execution.current_realized_return_pct()

    def set_table_item(self, table, row, column, value):
        existing = table.item(row, column)
        if existing is not None and existing.text() == value:
            return
        table.setItem(row, column, QTableWidgetItem(value))

    def compute_paper_summary(self):
        return self.execution.compute_summary()

    def render_paper_mode(self):
        snapshot = self.execution.last_snapshot
        decision = self.execution.last_decision
        position = self.execution.position

        if snapshot is None:
            self.set_paper_defaults()
            return

        disabled_detail = "Paper disabled - monitor mode."
        if not self.execution.enabled:
            signal_value = "DISABLED"
            signal_detail = disabled_detail
        elif self.execution_mode != "paper":
            signal_value = "MONITOR"
            signal_detail = "Mode reserved for future routing."
        else:
            signal_value = self.describe_paper_action(decision.action if decision else "no_action")
            signal_detail = self.describe_paper_reason(decision.reason if decision else "entry_conditions_not_met")
        self.paper_signal_card.set_content(signal_value, signal_detail)

        if position is None:
            self.paper_position_card.set_content("FLAT", "No position")
            unrealized_pct = 0.0
            unrealized_amount = 0.0
        else:
            mark_notional = self.execution.compute_position_notional(position, price=snapshot.current_price)
            self.paper_position_card.set_content(
                f"{position.side.upper()} {format_quantity(position.quantity)} @ {format_price(position.entry_price)}",
                f"{position.leverage}x | {format_price(mark_notional)}",
            )
            unrealized_pct = self.execution.compute_position_return_pct(position, snapshot.current_price)
            unrealized_amount = self.execution.compute_position_pnl_amount(position, snapshot.current_price)

        self.update_execution_order_preview()
        self.execution.append_equity_point(snapshot, unrealized_amount)
        self.paper_pnl_card.set_content(
            format_price(unrealized_amount),
            f"+{format_signed_pct(self.current_realized_return_pct())} realized | {format_price(self.current_paper_equity(unrealized_amount))} equity",
        )
        summary = self.compute_paper_summary()
        self.paper_trades_card.set_content(
            str(summary["trade_count"]),
            f"Win {summary['win_rate']*100:.0f}% | Avg {format_signed_pct(summary['avg_return'])}",
        )

        position_values = ["FLAT", "--", "--", "--", "--", format_price(snapshot.current_price), "--", "--", "--", "--", "+0.00%"]
        if position is not None:
            position_values = [
                position.side.upper(),
                position.entry_time.strftime("%Y-%m-%d %H:%M"),
                format_quantity(position.quantity),
                f"{position.leverage}x",
                format_price(position.entry_price),
                format_price(snapshot.current_price),
                format_price(self.execution.compute_position_notional(position, price=snapshot.current_price)),
                format_price(self.execution.compute_position_margin(position, price=position.entry_price)),
                format_price(position.stop_price),
                format_price(position.take_profit_price),
                f"{format_price(unrealized_amount)} | {format_signed_pct(unrealized_pct)}",
            ]
        for idx, value in enumerate(position_values):
            self.set_table_item(self.paper_position_table, 0, idx, value)

        self.paper_trade_table.setRowCount(len(self.execution.trade_history))
        for row, trade in enumerate(reversed(self.execution.trade_history)):
            values = [
                trade["time"].strftime("%m-%d %H:%M"),
                self.describe_paper_action(trade["action"]),
                format_quantity(trade["quantity"]),
                f"{trade['leverage']}x",
                format_price(trade["price"]),
                format_price(trade["notional"]) if trade["notional"] is not None else "--",
                format_price(trade["stop_price"]) if trade["stop_price"] is not None else "--",
                format_price(trade.get("take_profit_price")) if trade.get("take_profit_price") is not None else "--",
                format_price(trade["pnl_amount"]) if trade["pnl_amount"] is not None else "--",
                format_signed_pct(trade["forecast_return"]),
                self.describe_paper_reason(trade["reason"]),
            ]
            for col, value in enumerate(values):
                self.set_table_item(self.paper_trade_table, row, col, value)

        if hasattr(self, "paper_equity_card"):
            self.paper_equity_card.set_content(
                format_price(self.current_paper_equity()),
                format_signed_pct(self.current_realized_return_pct())
            )

        if not self.execution.enabled:
            position_text = "Disabled."
        elif self.execution_mode != "paper":
            position_text = "Reserved."
        else:
            position_text = "Active." if position is not None else "Flat."
        self.paper_status_label.setText(
            f"{position_text} Decision: {signal_value}. {signal_detail}"
        )

    def update_execution_order_preview(self):
        price = self.current_execution_price()
        notional = self.current_order_notional(price=price)
        margin = self.current_order_margin(price=price)
        if hasattr(self, "paper_notional_value"):
            self.paper_notional_value.setText(format_price(notional) if notional is not None else "--")
        if hasattr(self, "paper_margin_value"):
            self.paper_margin_value.setText(format_price(margin) if margin is not None else "--")
        if hasattr(self, "execution_mode_hint"):
            if self.execution_mode == "paper":
                mode_text = "Local Paper mode 會用目前數量與槓桿做本地模擬，尚未送出任何交易所委託。"
            elif self.execution_mode == "testnet":
                mode_text = "Binance testnet 入口已預留，現在仍只做本地模擬，不會對 testnet 下單。"
            else:
                mode_text = "Live mode 開關已預留，目前鎖定；執行層仍停留在本地 paper mode。"
            if price is not None:
                mode_text += f" 估算基準價 {format_price(price)}。"
            self.execution_mode_hint.setText(mode_text)

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
        if hasattr(self, "results_text") and self.results_text is not None:
            self.results_text.append(message)
        print(message)

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
        return self._PAPER_ACTION_LABELS.get(action, action.upper())

    def update_paper_mode(self, payload):
        self.latest_market_df = payload.get("df")
        snapshot, decision = self.execution.update(payload)
        if self.execution.last_snapshot is None:
            self.set_paper_defaults()
            self.paper_status_label.setText(self.describe_paper_reason("missing_validation_data"))
            return
        self.update_execution_order_preview()
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
        self.update_paper_mode(payload)
        self.forecast_canvas.plot_forecast(
            symbol=self.current_symbol,
            timeframe=self.current_timeframe,
            context_df=payload["context_df"],
            future_pred_df=payload["future_pred_df"],
            forecast_interval_df=payload["forecast_interval_df"],
            paper_trades=self.execution.trade_history,
        )
        self.validation_canvas.plot_validation(
            validation_history_df=payload["validation_history_df"],
            validation_pred_df=payload["validation_pred_df"],
            validation_actual_df=payload["validation_actual_df"],
        )
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
