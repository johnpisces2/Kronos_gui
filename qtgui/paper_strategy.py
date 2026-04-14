from dataclasses import dataclass, field
from typing import Literal, Optional

import pandas as pd


PositionSide = Literal["long", "short"]
DecisionAction = Literal[
    "enter_long",
    "enter_short",
    "exit_long",
    "exit_short",
    "hold",
    "no_action",
]


@dataclass(frozen=True)
class PaperStrategyConfig:
    entry_threshold_pct: float = 0.02
    stop_loss_fraction: float = 0.5
    min_stop_loss_pct: float = 0.01
    max_stop_loss_pct: float = 0.04
    take_profit_multiplier: float = 2.0


@dataclass(frozen=True)
class PaperSignalSnapshot:
    signal_time: pd.Timestamp
    current_price: float
    forecast_price: float
    forecast_return: float
    validation_start_price: float
    validation_pred_price: float
    validation_pred_return: float
    validation_history_delta: float
    validation_pred_delta: float


@dataclass(frozen=True)
class PaperPosition:
    side: PositionSide
    entry_price: float
    stop_price: float
    take_profit_price: float
    entry_time: pd.Timestamp
    quantity: float = 1.0
    leverage: float = 1.0


@dataclass(frozen=True)
class PaperDecision:
    action: DecisionAction
    reason: str
    stop_distance_pct: Optional[float] = None
    stop_price: Optional[float] = None
    tp_distance_pct: Optional[float] = None
    take_profit_price: Optional[float] = None


def pct_change(current_value: float, reference_value: float) -> float:
    if reference_value == 0:
        return 0.0
    return (current_value / reference_value) - 1.0


def clamp_stop_distance(predicted_edge: float, config: PaperStrategyConfig) -> float:
    return min(
        config.max_stop_loss_pct,
        max(config.min_stop_loss_pct, predicted_edge * config.stop_loss_fraction),
    )


def build_signal_snapshot(payload: dict) -> PaperSignalSnapshot:
    context_df = payload["context_df"]
    future_pred_df = payload["future_pred_df"]
    validation_history_df = payload["validation_history_df"]
    validation_pred_df = payload["validation_pred_df"]

    if any(df is None or df.empty for df in (context_df, future_pred_df, validation_history_df, validation_pred_df)):
        raise ValueError("Paper strategy requires forecast data plus validation history/prediction data.")
    if len(validation_history_df) < 2 or len(validation_pred_df) < 2:
        raise ValueError("Paper strategy requires at least two validation history bars and two validation pred bars.")

    current_price = float(context_df["close"].iloc[-1])
    forecast_price = float(future_pred_df["close"].iloc[-1])
    validation_start_price = float(validation_history_df["close"].iloc[-1])
    validation_pred_price = float(validation_pred_df["close"].iloc[-1])
    validation_pred_prev = float(validation_pred_df["close"].iloc[-2])

    return PaperSignalSnapshot(
        signal_time=pd.Timestamp(future_pred_df["timestamps"].iloc[0]),
        current_price=current_price,
        forecast_price=forecast_price,
        forecast_return=pct_change(forecast_price, current_price),
        validation_start_price=validation_start_price,
        validation_pred_price=validation_pred_price,
        validation_pred_return=pct_change(validation_pred_price, validation_start_price),
        validation_history_delta=float(validation_history_df["close"].iloc[-1] - validation_history_df["close"].iloc[-2]),
        validation_pred_delta=float(validation_pred_df["close"].iloc[-1] - validation_pred_df["close"].iloc[-2]),
    )


def should_enter_long(snapshot: PaperSignalSnapshot, config: PaperStrategyConfig) -> bool:
    return (
        snapshot.forecast_return >= config.entry_threshold_pct
        and snapshot.validation_pred_return > 0
        and snapshot.validation_pred_delta > 0
        and snapshot.validation_history_delta > 0
    )


def should_enter_short(snapshot: PaperSignalSnapshot, config: PaperStrategyConfig) -> bool:
    return (
        snapshot.forecast_return <= -config.entry_threshold_pct
        and snapshot.validation_pred_return < 0
        and snapshot.validation_pred_delta < 0
        and snapshot.validation_history_delta < 0
    )


def build_entry_decision(snapshot: PaperSignalSnapshot, config: PaperStrategyConfig) -> PaperDecision:
    if should_enter_long(snapshot, config):
        stop_distance = clamp_stop_distance(abs(snapshot.forecast_return), config)
        tp_distance = stop_distance * config.take_profit_multiplier
        stop_price = snapshot.current_price * (1.0 - stop_distance)
        tp_price = snapshot.current_price * (1.0 + tp_distance)
        return PaperDecision(
            action="enter_long",
            reason="long_signal_confirmed",
            stop_distance_pct=stop_distance,
            stop_price=stop_price,
            tp_distance_pct=tp_distance,
            take_profit_price=tp_price,
        )

    if should_enter_short(snapshot, config):
        stop_distance = clamp_stop_distance(abs(snapshot.forecast_return), config)
        tp_distance = stop_distance * config.take_profit_multiplier
        stop_price = snapshot.current_price * (1.0 + stop_distance)
        tp_price = snapshot.current_price * (1.0 - tp_distance)
        return PaperDecision(
            action="enter_short",
            reason="short_signal_confirmed",
            stop_distance_pct=stop_distance,
            stop_price=stop_price,
            tp_distance_pct=tp_distance,
            take_profit_price=tp_price,
        )

    return PaperDecision(action="no_action", reason="entry_conditions_not_met")


def build_exit_decision(
    position: PaperPosition,
    snapshot: PaperSignalSnapshot,
    config: PaperStrategyConfig,
) -> PaperDecision:
    if position.side == "long":
        if snapshot.current_price <= position.stop_price:
            return PaperDecision(action="exit_long", reason="stop_loss_hit")
        if snapshot.current_price >= position.take_profit_price:
            return PaperDecision(action="exit_long", reason="take_profit_hit")
        if snapshot.validation_pred_return <= 0 or snapshot.validation_pred_delta <= 0:
            return PaperDecision(action="exit_long", reason="long_signal_invalidated")
        return PaperDecision(action="hold", reason="long_position_still_valid")

    if position.side == "short":
        if snapshot.current_price >= position.stop_price:
            return PaperDecision(action="exit_short", reason="stop_loss_hit")
        if snapshot.current_price <= position.take_profit_price:
            return PaperDecision(action="exit_short", reason="take_profit_hit")
        if snapshot.validation_pred_return >= 0 or snapshot.validation_pred_delta >= 0:
            return PaperDecision(action="exit_short", reason="short_signal_invalidated")
        return PaperDecision(action="hold", reason="short_position_still_valid")

    return PaperDecision(action="no_action", reason="unknown_position_side")
