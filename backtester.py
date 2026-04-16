"""Backtesting engine for Kronos paper strategy."""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from paper_strategy import (
    PaperStrategyConfig,
    PaperSignalSnapshot,
    PaperPosition,
    PaperDecision,
    build_signal_snapshot,
    build_entry_decision,
    build_exit_decision,
)


@dataclass
class BacktestTrade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    leverage: float
    pnl_amount: float
    pnl_pct: float
    return_pct: float
    fee: float
    slippage_cost: float
    reason: str
    bars_held: int = 0


@dataclass
class BacktestMetrics:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_return_pct: float
    avg_win: float
    avg_loss: float
    avg_trade_return: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    profit_factor: float
    total_fees: float
    total_slippage: float
    time_in_market_pct: float
    equity_curve: List[Dict]
    trades: List[BacktestTrade]

    def to_dict(self) -> Dict:
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "total_pnl": self.total_pnl,
            "total_return_pct": self.total_return_pct,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "avg_trade_return": self.avg_trade_return,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_pct": self.max_drawdown_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "profit_factor": self.profit_factor,
            "total_fees": self.total_fees,
            "total_slippage": self.total_slippage,
            "time_in_market_pct": self.time_in_market_pct,
        }


class Backtester:
    def __init__(
        self,
        initial_equity: float = 10000.0,
        fee_rate: float = 0.0005,
        slippage_pct: float = 0.001,
    ):
        self.initial_equity = initial_equity
        self.fee_rate = fee_rate
        self.slippage_pct = slippage_pct
        self.strategy_config = PaperStrategyConfig()

    def run(
        self,
        context_df: pd.DataFrame,
        predictor_fn,
        forecast_params: Dict,
        debug: bool = False,
        progress_callback=None,
    ) -> BacktestMetrics:
        position = None
        equity = self.initial_equity
        equity_curve = []
        trades = []
        total_fees = 0.0
        total_slippage = 0.0

        max_equity = equity
        max_drawdown = 0.0
        peak_equity = equity

        total_bars = len(context_df) - 1
        bars_with_position = 0

        debug_log = []
        lookback = forecast_params.get("lookback", 512)
        pred_len = forecast_params.get("pred_len", 18)
        temperature = forecast_params.get("temperature", 0.5)
        top_p = forecast_params.get("top_p", 0.8)
        sample_count = forecast_params.get("sample_count", 3)
        val_compute_interval = forecast_params.get("val_compute_interval", 1)

        total_iterations = len(context_df) - lookback - 1
        cancelled = False

        cached_val_pred_for_bar = None
        cached_val_context_df = None
        bar_interval_hours = self._infer_bar_interval_hours(context_df)

        for i in range(lookback, len(context_df) - 1):
            current_bar = context_df.iloc[i]
            signal_time = pd.Timestamp(current_bar["timestamps"]) if "timestamps" in context_df.columns else pd.Timestamp(current_bar.name)

            if progress_callback and not progress_callback(i - lookback, total_iterations):
                cancelled = True
                break

            context_for_pred = context_df.iloc[i - lookback + 1:i + 1].copy()

            try:
                pred_result = predictor_fn(context_for_pred, temperature=temperature, top_p=top_p, sample_count=sample_count)
                if pred_result is None:
                    continue

                future_pred_df = pred_result["future_pred_df"]
                if future_pred_df is None or len(future_pred_df) == 0:
                    continue

                forecast_price = float(future_pred_df["close"].iloc[-1])
                current_price = float(current_bar["close"])
            except Exception as e:
                if debug:
                    debug_log.append(f"[{signal_time}] Prediction error: {e}")
                continue

            val_context_start = max(0, i - lookback - pred_len + 1)
            val_context_end = i - pred_len + 1
            if val_context_end <= val_context_start:
                val_context_start = max(0, val_context_end - lookback)
            val_context_df = context_df.iloc[val_context_start:val_context_end].copy() if val_context_end > val_context_start else None

            val_pred_for_bar = None
            val_pred_delta_for_bar = None
            if val_context_df is not None and len(val_context_df) >= lookback // 2:
                try:
                    should_compute_val = (
                        i - lookback < val_compute_interval or
                        i % val_compute_interval == 0 or
                        cached_val_context_df is None or
                        len(val_context_df) != len(cached_val_context_df) or
                        not val_context_df[["open", "high", "low", "close", "volume"]].equals(cached_val_context_df[["open", "high", "low", "close", "volume"]])
                    )
                    
                    if should_compute_val:
                        val_pred_result = predictor_fn(val_context_df, temperature=temperature, top_p=top_p, sample_count=sample_count)
                        if val_pred_result is not None:
                            val_future_df = val_pred_result.get("future_pred_df")
                            if val_future_df is not None and len(val_future_df) > 0:
                                cached_val_pred_for_bar = {
                                    "price": float(val_future_df["close"].iloc[-1]),
                                    "delta": float(val_future_df["close"].iloc[-1] - val_future_df["close"].iloc[-2]) if len(val_future_df) > 1 else 0.0,
                                }
                                cached_val_context_df = val_context_df.copy()
                        val_pred_for_bar = cached_val_pred_for_bar
                    else:
                        val_pred_for_bar = cached_val_pred_for_bar
                except Exception:
                    val_pred_for_bar = cached_val_pred_for_bar

            if val_pred_for_bar is not None:
                val_pred_delta_for_bar = val_pred_for_bar["delta"]
                val_pred_for_bar = val_pred_for_bar["price"]

            val_start_price = float(context_df.iloc[max(0, i - pred_len)]["close"]) if i >= pred_len else float(context_df.iloc[0]["close"])
            val_hist_delta = 0.0
            if i > pred_len:
                val_hist_delta = float(context_df.iloc[i - pred_len]["close"]) - float(context_df.iloc[max(0, i - pred_len - 1)]["close"])

            snapshot = self._build_snapshot_at_bar(
                signal_time=signal_time,
                current_price=current_price,
                forecast_price=forecast_price,
                val_start_price=val_start_price,
                val_pred_price=val_pred_for_bar,
                val_pred_delta=val_pred_delta_for_bar,
                val_hist_delta=val_hist_delta,
            )
            if snapshot is None:
                continue

            if position is None:
                decision = build_entry_decision(snapshot, self.strategy_config)
                if debug and decision.action == "no_action":
                    debug_log.append(f"[{signal_time}] REJECTED: fcst={snapshot.forecast_return:.4f} val={snapshot.validation_pred_return:.4f}")

                if decision.action in ("enter_long", "enter_short"):
                    side = "long" if decision.action == "enter_long" else "short"
                    stop_distance_pct = decision.stop_distance_pct or (abs(snapshot.forecast_return) * 0.5)
                    tp_distance_pct = stop_distance_pct * 2

                    if side == "long":
                        stop_price = snapshot.current_price * (1 - stop_distance_pct)
                        tp_price = snapshot.current_price * (1 + tp_distance_pct)
                    else:
                        stop_price = snapshot.current_price * (1 + stop_distance_pct)
                        tp_price = snapshot.current_price * (1 - tp_distance_pct)

                    quantity = self._calculate_quantity(equity, snapshot.current_price, stop_distance_pct)
                    leverage = 2

                    entry_cost = snapshot.current_price * (1 + self.slippage_pct) if side == "long" else snapshot.current_price * (1 - self.slippage_pct)
                    fee = entry_cost * quantity * self.fee_rate
                    total_fees += fee
                    total_slippage += snapshot.current_price * quantity * self.slippage_pct

                    position = PaperPosition(
                        side=side,
                        entry_price=entry_cost,
                        stop_price=stop_price,
                        take_profit_price=tp_price,
                        entry_time=signal_time,
                        quantity=quantity,
                        leverage=leverage,
                    )
                    equity -= fee
                    if debug:
                        debug_log.append(f"[{signal_time}] ENTER {side.upper()} @ {entry_cost:.2f}")
                else:
                    pass

            else:
                bars_with_position += 1
                current_price = snapshot.current_price
                exit_price = None
                reason = None

                bars_held = 0
                if position.entry_time is not None:
                    time_diff = signal_time - position.entry_time
                    hours_held = time_diff.total_seconds() / 3600
                    if bar_interval_hours > 0:
                        bars_held = int(hours_held / bar_interval_hours)

                should_exit = False
                exit_decision = build_exit_decision(
                    position=position,
                    snapshot=snapshot,
                    config=self.strategy_config,
                    bar_interval_hours=bar_interval_hours,
                )

                if exit_decision.action in ("exit_long", "exit_short"):
                    should_exit = True
                    reason = exit_decision.reason

                    if reason == "stop_loss_hit":
                        exit_price = position.stop_price
                    elif reason == "take_profit_hit":
                        exit_price = position.take_profit_price
                    else:
                        exit_price = current_price

                if should_exit and exit_price is not None:
                    exit_cost = exit_price * (1 - self.slippage_pct) if position.side == "long" else exit_price * (1 + self.slippage_pct)
                    fee = exit_cost * position.quantity * self.fee_rate
                    total_fees += fee
                    total_slippage += position.quantity * abs(exit_price - position.entry_price) * self.slippage_pct

                    if position.side == "long":
                        pnl_amount = (exit_cost - position.entry_price) * position.quantity * position.leverage
                        return_pct = (exit_cost / position.entry_price - 1)
                    else:
                        pnl_amount = (position.entry_price - exit_cost) * position.quantity * position.leverage
                        return_pct = (position.entry_price / exit_cost - 1)

                    equity += pnl_amount - fee

                    trade = BacktestTrade(
                        entry_time=position.entry_time,
                        exit_time=signal_time,
                        side=position.side,
                        entry_price=position.entry_price,
                        exit_price=exit_cost,
                        quantity=position.quantity,
                        leverage=position.leverage,
                        pnl_amount=pnl_amount - fee,
                        pnl_pct=pnl_amount / (position.entry_price * position.quantity / position.leverage) if position.entry_price * position.quantity > 0 else 0,
                        return_pct=return_pct,
                        fee=fee,
                        slippage_cost=position.quantity * abs(exit_price - position.entry_price) * self.slippage_pct,
                        reason=reason,
                        bars_held=bars_held,
                    )
                    trades.append(trade)
                    position = None

            equity_curve.append({
                "time": signal_time,
                "equity": equity,
            })

            if equity > peak_equity:
                peak_equity = equity
            drawdown = peak_equity - equity
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        time_in_market_pct = bars_with_position / total_bars if total_bars > 0 else 0.0

        if cancelled:
            debug_log.append("[INFO] Backtest cancelled by user")

        if debug and debug_log:
            print("\n=== BACKTEST DEBUG LOG (first 50 entries) ===")
            for line in debug_log[:50]:
                print(line)
            if len(debug_log) > 50:
                print(f"... and {len(debug_log) - 50} more entries")
            print("=== END DEBUG LOG ===\n")

        return self._compute_metrics(
            equity, equity_curve, trades, total_fees, total_slippage, max_drawdown, time_in_market_pct
        )

    def _build_snapshot_at_idx(
        self,
        context_df: pd.DataFrame,
        future_pred_df: pd.DataFrame,
        validation_history_df: pd.DataFrame,
        validation_pred_df: pd.DataFrame,
        idx: int,
        signal_time: pd.Timestamp,
    ) -> Optional[PaperSignalSnapshot]:
        if idx < 2:
            return None

        context_len = len(context_df)
        val_hist_len = len(validation_history_df)
        val_pred_len = len(validation_pred_df)

        if idx >= context_len or idx >= val_hist_len or idx >= val_pred_len:
            return None

        current_price = float(context_df.iloc[idx]["close"])

        offset = context_len - val_hist_len
        val_hist_idx = idx - offset
        if val_hist_idx < 0:
            val_hist_idx = 0

        val_start_idx = min(val_hist_idx, val_hist_len - 1)
        val_start_price = float(validation_history_df.iloc[val_start_idx]["close"])

        offset_pred = context_len - val_pred_len
        val_pred_idx = idx - offset_pred
        if val_pred_idx < 0:
            val_pred_idx = 0

        val_pred_price = float(validation_pred_df.iloc[val_pred_idx]["close"])
        val_pred_delta = 0.0
        if val_pred_idx > 0:
            val_pred_delta = float(validation_pred_df.iloc[val_pred_idx]["close"]) - float(validation_pred_df.iloc[val_pred_idx - 1]["close"])

        val_hist_delta = 0.0
        if val_start_idx > 0:
            val_hist_delta = float(validation_history_df.iloc[val_start_idx]["close"]) - float(validation_history_df.iloc[val_start_idx - 1]["close"])

        forecast_idx = min(idx, len(future_pred_df) - 1) if len(future_pred_df) > 0 else 0
        forecast_price = float(future_pred_df.iloc[forecast_idx]["close"]) if len(future_pred_df) > 0 else current_price

        return PaperSignalSnapshot(
            signal_time=signal_time,
            current_price=current_price,
            forecast_price=forecast_price,
            forecast_return=(forecast_price / current_price - 1) if current_price > 0 else 0,
            validation_start_price=val_start_price,
            validation_pred_price=val_pred_price,
            validation_pred_return=(val_pred_price / val_start_price - 1) if val_start_price > 0 else 0,
            validation_history_delta=val_hist_delta,
            validation_pred_delta=val_pred_delta,
        )

    def _build_snapshot_at_bar(
        self,
        signal_time: pd.Timestamp,
        current_price: float,
        forecast_price: float,
        val_start_price: float,
        val_pred_price: Optional[float],
        val_pred_delta: Optional[float],
        val_hist_delta: float,
    ) -> Optional[PaperSignalSnapshot]:
        if val_pred_price is None:
            return None

        return PaperSignalSnapshot(
            signal_time=signal_time,
            current_price=current_price,
            forecast_price=forecast_price,
            forecast_return=(forecast_price / current_price - 1) if current_price > 0 else 0,
            validation_start_price=val_start_price,
            validation_pred_price=val_pred_price,
            validation_pred_return=(val_pred_price / val_start_price - 1) if val_start_price > 0 else 0,
            validation_history_delta=val_hist_delta,
            validation_pred_delta=val_pred_delta if val_pred_delta is not None else 0.0,
        )

    def _infer_bar_interval_hours(self, context_df: pd.DataFrame) -> float:
        if "timestamps" not in context_df.columns or len(context_df) < 2:
            return 1.0

        timestamps = pd.to_datetime(context_df["timestamps"])
        diffs = timestamps.diff().dropna()
        diffs = diffs[diffs > pd.Timedelta(0)]
        if diffs.empty:
            return 1.0

        return diffs.median().total_seconds() / 3600

    def _build_snapshot_with_forecast(
        self,
        signal_time: pd.Timestamp,
        current_price: float,
        forecast_price: float,
        validation_history_df: pd.DataFrame,
        validation_pred_df: pd.DataFrame,
        context_idx: int,
        total_context_len: int,
    ) -> Optional[PaperSignalSnapshot]:
        val_hist_len = len(validation_history_df)
        val_pred_len = len(validation_pred_df)

        if val_hist_len < 2 or val_pred_len < 2:
            return None

        val_start_idx = min(val_hist_len - 1, context_idx % val_hist_len)
        val_start_price = float(validation_history_df.iloc[val_start_idx]["close"])

        val_pred_idx = min(val_pred_len - 1, context_idx % val_pred_len)
        val_pred_price = float(validation_pred_df.iloc[val_pred_idx]["close"])

        val_pred_delta = 0.0
        if val_pred_idx > 0:
            val_pred_delta = float(validation_pred_df.iloc[val_pred_idx]["close"]) - float(validation_pred_df.iloc[val_pred_idx - 1]["close"])

        val_hist_delta = 0.0
        if val_start_idx > 0:
            val_hist_delta = float(validation_history_df.iloc[val_start_idx]["close"]) - float(validation_history_df.iloc[val_start_idx - 1]["close"])

        return PaperSignalSnapshot(
            signal_time=signal_time,
            current_price=current_price,
            forecast_price=forecast_price,
            forecast_return=(forecast_price / current_price - 1) if current_price > 0 else 0,
            validation_start_price=val_start_price,
            validation_pred_price=val_pred_price,
            validation_pred_return=(val_pred_price / val_start_price - 1) if val_start_price > 0 else 0,
            validation_history_delta=val_hist_delta,
            validation_pred_delta=val_pred_delta,
        )

    def _calculate_quantity(self, equity: float, price: float, stop_distance_pct: float) -> float:
        risk_amount = equity * 0.08
        stop_distance_price = price * stop_distance_pct
        if stop_distance_price <= 0:
            return 0.01
        qty = risk_amount / stop_distance_price
        return max(0.0001, qty)

    def _compute_metrics(
        self,
        final_equity: float,
        equity_curve: List[Dict],
        trades: List[BacktestTrade],
        total_fees: float,
        total_slippage: float,
        max_drawdown: float,
        time_in_market_pct: float,
    ) -> BacktestMetrics:
        closed_trades = [t for t in trades if t.pnl_amount is not None]
        total_trades = len(closed_trades)

        if total_trades == 0:
            return BacktestMetrics(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                total_pnl=0.0,
                total_return_pct=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                avg_trade_return=0.0,
                max_drawdown=max_drawdown,
                max_drawdown_pct=max_drawdown / self.initial_equity if self.initial_equity > 0 else 0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                profit_factor=0.0,
                total_fees=total_fees,
                total_slippage=total_slippage,
                time_in_market_pct=time_in_market_pct,
                equity_curve=equity_curve,
                trades=trades,
            )

        winning_trades = [t for t in closed_trades if t.pnl_amount > 0]
        losing_trades = [t for t in closed_trades if t.pnl_amount <= 0]
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0

        total_pnl = sum(t.pnl_amount for t in closed_trades)
        total_return_pct = ((final_equity / self.initial_equity) - 1) * 100 if self.initial_equity > 0 else 0.0

        avg_win = np.mean([t.pnl_amount for t in winning_trades]) if winning_trades else 0.0
        avg_loss = np.mean([t.pnl_amount for t in losing_trades]) if losing_trades else 0.0
        avg_trade_return = np.mean([t.return_pct for t in closed_trades]) if closed_trades else 0.0

        gross_profit = sum(t.pnl_amount for t in winning_trades)
        gross_loss = abs(sum(t.pnl_amount for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        returns_list = [t.return_pct for t in closed_trades]
        if len(returns_list) > 1:
            returns_std = np.std(returns_list)
            sharpe = (np.mean(returns_list) / returns_std * np.sqrt(252)) if returns_std > 0 else 0.0

            downside_returns = [r for r in returns_list if r < 0]
            if len(downside_returns) > 1:
                downside_std = np.std(downside_returns)
                sortino = (np.mean(returns_list) / downside_std * np.sqrt(252)) if downside_std > 0 else 0.0
            else:
                sortino = 0.0
        else:
            sharpe = 0.0
            sortino = 0.0

        max_dd_pct = max_drawdown / self.initial_equity if self.initial_equity > 0 else 0

        return BacktestMetrics(
            total_trades=total_trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            total_pnl=total_pnl,
            total_return_pct=total_return_pct,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_trade_return=avg_trade_return,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_dd_pct,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            profit_factor=profit_factor,
            total_fees=total_fees,
            total_slippage=total_slippage,
            time_in_market_pct=time_in_market_pct,
            equity_curve=equity_curve,
            trades=trades,
        )
