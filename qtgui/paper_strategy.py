from dataclasses import dataclass
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
    entry_threshold_pct: float = 0.03
    stop_loss_fraction: float = 0.5
    min_stop_loss_pct: float = 0.01
    max_stop_loss_pct: float = 0.05


@dataclass(frozen=True)
class PaperSignalSnapshot:
    signal_time: pd.Timestamp
    current_price: float
    forecast_price: float
    forecast_return: float
    validation_start_price: float
    validation_pred_price: float
    validation_actual_price: float
    validation_pred_return: float
    validation_actual_return: float
    validation_history_delta: float
    validation_actual_delta: float


@dataclass(frozen=True)
class PaperPosition:
    side: PositionSide
    entry_price: float
    stop_price: float
    entry_time: pd.Timestamp


@dataclass(frozen=True)
class PaperDecision:
    action: DecisionAction
    reason: str
    stop_distance_pct: Optional[float] = None
    stop_price: Optional[float] = None


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
    validation_actual_df = payload["validation_actual_df"]

    if any(df is None or df.empty for df in (context_df, future_pred_df, validation_history_df, validation_pred_df, validation_actual_df)):
        raise ValueError("Paper strategy requires forecast data plus validation history/prediction/actual data.")
    if len(validation_history_df) < 2 or len(validation_actual_df) < 2:
        raise ValueError("Paper strategy requires at least two validation history bars and two validation actual bars.")

    current_price = float(context_df["close"].iloc[-1])
    forecast_price = float(future_pred_df["close"].iloc[-1])
    validation_start_price = float(validation_history_df["close"].iloc[-1])
    validation_pred_price = float(validation_pred_df["close"].iloc[-1])
    validation_actual_price = float(validation_actual_df["close"].iloc[-1])

    return PaperSignalSnapshot(
        signal_time=pd.Timestamp(future_pred_df["timestamps"].iloc[0]),
        current_price=current_price,
        forecast_price=forecast_price,
        forecast_return=pct_change(forecast_price, current_price),
        validation_start_price=validation_start_price,
        validation_pred_price=validation_pred_price,
        validation_actual_price=validation_actual_price,
        validation_pred_return=pct_change(validation_pred_price, validation_start_price),
        validation_actual_return=pct_change(validation_actual_price, validation_start_price),
        validation_history_delta=float(validation_history_df["close"].iloc[-1] - validation_history_df["close"].iloc[-2]),
        validation_actual_delta=float(validation_actual_df["close"].iloc[-1] - validation_actual_df["close"].iloc[-2]),
    )


def should_enter_long(snapshot: PaperSignalSnapshot, config: PaperStrategyConfig) -> bool:
    return (
        snapshot.forecast_return >= config.entry_threshold_pct
        and snapshot.validation_pred_return > 0
        and snapshot.validation_actual_return > 0
        and snapshot.validation_history_delta > 0
        and snapshot.validation_actual_delta > 0
    )


def should_enter_short(snapshot: PaperSignalSnapshot, config: PaperStrategyConfig) -> bool:
    return (
        snapshot.forecast_return <= -config.entry_threshold_pct
        and snapshot.validation_pred_return < 0
        and snapshot.validation_actual_return < 0
        and snapshot.validation_history_delta < 0
        and snapshot.validation_actual_delta < 0
    )


def build_entry_decision(snapshot: PaperSignalSnapshot, config: PaperStrategyConfig) -> PaperDecision:
    if should_enter_long(snapshot, config):
        stop_distance = clamp_stop_distance(abs(snapshot.forecast_return), config)
        stop_price = snapshot.current_price * (1.0 - stop_distance)
        return PaperDecision(
            action="enter_long",
            reason="forecast_upside_and_validation_trend_confirm_long",
            stop_distance_pct=stop_distance,
            stop_price=stop_price,
        )

    if should_enter_short(snapshot, config):
        stop_distance = clamp_stop_distance(abs(snapshot.forecast_return), config)
        stop_price = snapshot.current_price * (1.0 + stop_distance)
        return PaperDecision(
            action="enter_short",
            reason="forecast_downside_and_validation_trend_confirm_short",
            stop_distance_pct=stop_distance,
            stop_price=stop_price,
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
        if (
            snapshot.forecast_return < config.entry_threshold_pct
            or snapshot.validation_pred_return <= 0
            or snapshot.validation_actual_return <= 0
            or snapshot.validation_history_delta <= 0
            or snapshot.validation_actual_delta <= 0
        ):
            return PaperDecision(action="exit_long", reason="long_signal_invalidated")
        return PaperDecision(action="hold", reason="long_position_still_valid")

    if snapshot.current_price >= position.stop_price:
        return PaperDecision(action="exit_short", reason="stop_loss_hit")
    if (
        snapshot.forecast_return > -config.entry_threshold_pct
        or snapshot.validation_pred_return >= 0
        or snapshot.validation_actual_return >= 0
        or snapshot.validation_history_delta >= 0
        or snapshot.validation_actual_delta >= 0
    ):
        return PaperDecision(action="exit_short", reason="short_signal_invalidated")
    return PaperDecision(action="hold", reason="short_position_still_valid")
