"""Paper trading execution engine for Kronos."""

from paper_strategy import (
    PaperDecision,
    PaperPosition,
    PaperSignalSnapshot,
    PaperStrategyConfig,
    build_entry_decision,
    build_exit_decision,
    build_signal_snapshot,
)

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Optional

import pandas as pd


PositionSide = Literal["long", "short"]

PAPER_ACTION_LABELS = {
    "enter_long": "ENTER LONG",
    "enter_short": "ENTER SHORT",
    "exit_long": "EXIT LONG",
    "exit_short": "EXIT SHORT",
    "hold": "HOLD",
    "no_action": "NO ACTION",
}

PAPER_REASON_LABELS = {
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


def format_price(value):
    return f"${value:,.0f}"


def format_signed_pct(value):
    return f"{value * 100:+.2f}%"


def format_quantity(value):
    return f"{value:,.4f}".rstrip("0").rstrip(".")


class ExecutionEngine:
    """Handles paper trading state and logic."""

    def __init__(self):
        self.enabled = True
        self.mode = "paper"
        self.order_quantity = 0.01
        self.order_leverage = 5
        self.use_risk_fraction = True
        self.risk_fraction = 0.08
        self.initial_equity = 1000.0
        self.realized_equity = self.initial_equity
        self.realized_pnl_pct = 0.0

        self.position: Optional[PaperPosition] = None
        self.trade_history: List[Dict] = []
        self.equity_history: List[Dict] = []
        self.last_snapshot: Optional[PaperSignalSnapshot] = None
        self.last_decision: Optional[PaperDecision] = None
        self.strategy_config = PaperStrategyConfig()

        self._on_trade_callback: Optional[Callable] = None
        self._on_equity_update_callback: Optional[Callable] = None
        self._log_callback: Optional[Callable] = None

    def set_log_callback(self, callback: Callable):
        self._log_callback = callback

    def set_on_trade_callback(self, callback: Callable):
        self._on_trade_callback = callback

    def set_on_equity_update_callback(self, callback: Callable):
        self._on_equity_update_callback = callback

    def trading_active(self) -> bool:
        return self.enabled and self.mode == "paper"

    def current_order_quantity(self) -> float:
        return max(0.0001, float(self.order_quantity))

    def current_order_leverage(self) -> int:
        return max(1, int(self.order_leverage))

    def current_order_quantity_by_risk(self, entry_price: float, stop_distance_pct: float) -> float:
        equity = self.current_equity()
        risk_amount = equity * self.risk_fraction
        stop_distance_price = entry_price * stop_distance_pct
        if stop_distance_price <= 0:
            return self.current_order_quantity()
        qty = risk_amount / stop_distance_price
        return max(0.0001, qty)

    def current_order_notional(self, price: float, quantity: float) -> float:
        return price * quantity

    def current_order_margin(self, price: float, quantity: float, leverage: int) -> float:
        return self.current_order_notional(price, quantity) / max(1, leverage)

    def current_equity(self, unrealized_amount: float = 0.0) -> float:
        return self.realized_equity + unrealized_amount

    def current_realized_return_pct(self) -> float:
        if self.initial_equity == 0:
            return 0.0
        return (self.realized_equity / self.initial_equity) - 1.0

    def compute_position_return_pct(self, position: PaperPosition, current_price: float) -> float:
        if position.side == "long":
            return (current_price / position.entry_price) - 1.0
        return (position.entry_price / current_price) - 1.0

    def compute_position_pnl_amount(self, position: PaperPosition, current_price: float) -> float:
        if position.side == "long":
            return (current_price - position.entry_price) * position.quantity
        return (position.entry_price - current_price) * position.quantity

    def compute_position_notional(self, position: PaperPosition, price: Optional[float] = None) -> float:
        mark_price = float(position.entry_price if price is None else price)
        return mark_price * position.quantity

    def compute_position_margin(self, position: PaperPosition, price: Optional[float] = None) -> float:
        notional = self.compute_position_notional(position, price=price)
        return notional / max(1, position.leverage)

    def append_equity_point(self, snapshot: PaperSignalSnapshot, unrealized_amount: float):
        point = {
            "time": pd.Timestamp(snapshot.signal_time),
            "equity": self.current_equity(unrealized_amount),
        }
        if self.equity_history and self.equity_history[-1]["time"] == point["time"]:
            self.equity_history[-1] = point
            return
        self.equity_history.append(point)
        self.equity_history = self.equity_history[-240:]
        if self._on_equity_update_callback:
            self._on_equity_update_callback()

    def compute_max_drawdown_pct(self) -> float:
        if not self.equity_history:
            return 0.0
        peak = float('-inf')
        max_dd = 0.0
        for point in self.equity_history:
            equity = point["equity"]
            if equity > peak:
                peak = equity
            dd = (equity / peak) - 1.0 if peak > 0 else 0.0
            if dd < max_dd:
                max_dd = dd
        return max_dd

    def compute_summary(self) -> Dict:
        closed_trades = [trade for trade in self.trade_history if trade.get("pnl_pct") is not None]
        trade_count = len(closed_trades)
        if trade_count == 0:
            return {
                "trade_count": 0,
                "win_rate": 0.0,
                "avg_return": 0.0,
                "max_drawdown": self.compute_max_drawdown_pct(),
            }

        wins = 0
        for trade in closed_trades:
            if trade["pnl_pct"] > 0:
                wins += 1
        return {
            "trade_count": trade_count,
            "win_rate": wins / trade_count if trade_count > 0 else 0.0,
            "avg_return": self.realized_pnl_pct / trade_count if trade_count > 0 else 0.0,
            "max_drawdown": self.compute_max_drawdown_pct(),
        }

    def record_trade(self, *, action: str, snapshot: PaperSignalSnapshot, reason: str,
                     position: Optional[PaperPosition] = None,
                     stop_price=None, take_profit_price=None,
                     pnl_pct=None, pnl_amount=None):
        quantity = position.quantity if position is not None else self.current_order_quantity()
        leverage = position.leverage if position is not None else self.current_order_leverage()
        notional = self.current_order_notional(price=snapshot.current_price, quantity=quantity)
        margin = self.current_order_margin(price=snapshot.current_price, quantity=quantity, leverage=leverage)
        trade = {
            "time": pd.Timestamp(snapshot.signal_time),
            "action": action,
            "price": float(snapshot.current_price),
            "reason": reason,
            "stop_price": stop_price,
            "take_profit_price": take_profit_price,
            "quantity": quantity,
            "leverage": leverage,
            "notional": notional,
            "margin": margin,
            "forecast_return": float(snapshot.forecast_return),
            "validation_pred_return": float(snapshot.validation_pred_return),
            "pnl_pct": pnl_pct,
            "pnl_amount": pnl_amount,
            "symbol": "BTC/USDT",
            "timeframe": "4h",
        }
        self.trade_history.append(trade)
        self.trade_history = self.trade_history[-40:]

        if self._log_callback:
            self._log_callback(
                f"[PAPER] {self.describe_action(action)} @ {format_price(trade['price'])} | "
                f"qty {format_quantity(trade['quantity'])} | lev {trade['leverage']}x | "
                f"forecast {format_signed_pct(trade['forecast_return'])} | "
                f"reason: {self.describe_reason(reason)}"
            )

        if self._on_trade_callback:
            self._on_trade_callback(trade)

    def describe_action(self, action: str) -> str:
        return PAPER_ACTION_LABELS.get(action, action.upper())

    def describe_reason(self, reason: str) -> str:
        return PAPER_REASON_LABELS.get(reason, reason.replace("_", " ").capitalize())

    def reset(self):
        self.position = None
        self.trade_history = []
        self.equity_history = []
        self.last_snapshot = None
        self.last_decision = None
        self.realized_equity = self.initial_equity
        self.realized_pnl_pct = 0.0

    def update(self, payload: dict) -> tuple:
        self.last_snapshot = None
        try:
            snapshot = build_signal_snapshot(payload)
        except ValueError:
            self.last_decision = None
            return None, None

        self.last_snapshot = snapshot

        if not self.trading_active():
            self.last_decision = None
            return snapshot, None

        if self.position is None:
            decision = build_entry_decision(snapshot, self.strategy_config)
            self.last_decision = decision
            if decision.action in ("enter_long", "enter_short"):
                side = "long" if decision.action == "enter_long" else "short"
                if self.use_risk_fraction and decision.stop_distance_pct:
                    qty = self.current_order_quantity_by_risk(snapshot.current_price, decision.stop_distance_pct)
                    lev = self.current_order_leverage()
                else:
                    qty = self.current_order_quantity()
                    lev = self.current_order_leverage()
                self.position = PaperPosition(
                    side=side,
                    entry_price=snapshot.current_price,
                    stop_price=decision.stop_price,
                    take_profit_price=decision.take_profit_price,
                    entry_time=snapshot.signal_time,
                    quantity=qty,
                    leverage=lev,
                )
                self.record_trade(
                    action=decision.action,
                    snapshot=snapshot,
                    reason=decision.reason,
                    position=self.position,
                    stop_price=decision.stop_price,
                    take_profit_price=decision.take_profit_price,
                )
                return snapshot, decision
        else:
            decision = build_exit_decision(self.position, snapshot, self.strategy_config)
            self.last_decision = decision
            if decision.action in ("exit_long", "exit_short"):
                pnl_pct = self.compute_position_return_pct(self.position, snapshot.current_price)
                pnl_amount = self.compute_position_pnl_amount(self.position, snapshot.current_price)
                self.realized_equity += pnl_amount
                self.realized_pnl_pct += pnl_pct
                self.record_trade(
                    action=decision.action,
                    snapshot=snapshot,
                    reason=decision.reason,
                    position=self.position,
                    stop_price=self.position.stop_price,
                    take_profit_price=self.position.take_profit_price,
                    pnl_pct=pnl_pct,
                    pnl_amount=pnl_amount,
                )
                self.position = None
                return snapshot, decision

        return snapshot, decision
