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
    strong_forecast_threshold_pct: float = 0.02
    min_validation_conditions: int = 2
    stop_loss_fraction: float = 0.5
    min_stop_loss_pct: float = 0.02
    max_stop_loss_pct: float = 0.04
    take_profit_multiplier: float = 3.0
    max_bars_in_position: int = 16


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


MarketRegime = Literal["STRONG_BULL", "BULL", "NEUTRAL", "BEAR", "STRONG_BEAR"]


def get_market_regime(snapshot: PaperSignalSnapshot) -> MarketRegime:
    fcst = snapshot.forecast_return
    val_pred = snapshot.validation_pred_return
    val_delta = snapshot.validation_pred_delta
    hist_delta = snapshot.validation_history_delta

    fcst_bullish = fcst > 0.02
    fcst_bearish = fcst < -0.02

    val_pred_bullish = val_pred > 0
    val_pred_bearish = val_pred < 0

    val_delta_bullish = val_delta > 0
    val_delta_bearish = val_delta < 0

    hist_bullish = hist_delta > 0
    hist_bearish = hist_delta < 0

    bullish_count = sum([fcst_bullish, val_pred_bullish, val_delta_bullish, hist_bullish])
    bearish_count = sum([fcst_bearish, val_pred_bearish, val_delta_bearish, hist_bearish])

    trend_strength = abs(fcst) + abs(val_pred)

    if bullish_count >= 3 and trend_strength > 0.04:
        return "STRONG_BULL"
    elif bullish_count >= 2 and trend_strength > 0.02:
        return "BULL"
    elif bearish_count >= 3 and trend_strength > 0.04:
        return "STRONG_BEAR"
    elif bearish_count >= 2 and trend_strength > 0.02:
        return "BEAR"
    else:
        return "NEUTRAL"


def get_regime_adjusted_config(regime: MarketRegime, base_config: PaperStrategyConfig) -> PaperStrategyConfig:
    if regime == "STRONG_BULL":
        return PaperStrategyConfig(
            entry_threshold_pct=base_config.entry_threshold_pct,
            strong_forecast_threshold_pct=base_config.strong_forecast_threshold_pct,
            min_validation_conditions=1,
            stop_loss_fraction=base_config.stop_loss_fraction,
            min_stop_loss_pct=0.025,
            max_stop_loss_pct=0.05,
            take_profit_multiplier=4.0,
            max_bars_in_position=24,
        )
    elif regime == "BULL":
        return PaperStrategyConfig(
            entry_threshold_pct=base_config.entry_threshold_pct,
            strong_forecast_threshold_pct=base_config.strong_forecast_threshold_pct,
            min_validation_conditions=2,
            stop_loss_fraction=base_config.stop_loss_fraction,
            min_stop_loss_pct=0.02,
            max_stop_loss_pct=0.04,
            take_profit_multiplier=3.0,
            max_bars_in_position=16,
        )
    elif regime == "STRONG_BEAR":
        return PaperStrategyConfig(
            entry_threshold_pct=base_config.entry_threshold_pct,
            strong_forecast_threshold_pct=base_config.strong_forecast_threshold_pct,
            min_validation_conditions=1,
            stop_loss_fraction=base_config.stop_loss_fraction,
            min_stop_loss_pct=0.025,
            max_stop_loss_pct=0.05,
            take_profit_multiplier=4.0,
            max_bars_in_position=24,
        )
    elif regime == "BEAR":
        return PaperStrategyConfig(
            entry_threshold_pct=base_config.entry_threshold_pct,
            strong_forecast_threshold_pct=base_config.strong_forecast_threshold_pct,
            min_validation_conditions=2,
            stop_loss_fraction=base_config.stop_loss_fraction,
            min_stop_loss_pct=0.02,
            max_stop_loss_pct=0.04,
            take_profit_multiplier=3.0,
            max_bars_in_position=16,
        )
    else:
        return PaperStrategyConfig(
            entry_threshold_pct=base_config.entry_threshold_pct,
            strong_forecast_threshold_pct=base_config.strong_forecast_threshold_pct,
            min_validation_conditions=3,
            stop_loss_fraction=base_config.stop_loss_fraction,
            min_stop_loss_pct=0.015,
            max_stop_loss_pct=0.03,
            take_profit_multiplier=2.0,
            max_bars_in_position=8,
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
    regime = get_market_regime(snapshot)
    
    if regime not in ("STRONG_BULL", "BULL"):
        return False
    
    adjusted_config = get_regime_adjusted_config(regime, config)
    
    if snapshot.forecast_return < adjusted_config.strong_forecast_threshold_pct:
        return False

    val_count = sum([
        snapshot.validation_pred_return > 0,
        snapshot.validation_pred_delta > 0,
        snapshot.validation_history_delta > 0,
    ])

    return val_count >= adjusted_config.min_validation_conditions


def should_enter_short(snapshot: PaperSignalSnapshot, config: PaperStrategyConfig) -> bool:
    regime = get_market_regime(snapshot)

    if regime not in ("STRONG_BEAR", "BEAR"):
        return False

    adjusted_config = get_regime_adjusted_config(regime, config)

    if snapshot.forecast_return > -adjusted_config.strong_forecast_threshold_pct:
        return False

    val_count = sum([
        snapshot.validation_pred_return < 0,
        snapshot.validation_pred_delta < 0,
        snapshot.validation_history_delta < 0,
    ])

    return val_count >= adjusted_config.min_validation_conditions


def build_entry_decision(snapshot: PaperSignalSnapshot, config: PaperStrategyConfig) -> PaperDecision:
    regime = get_market_regime(snapshot)
    adjusted_config = get_regime_adjusted_config(regime, config)
    
    if should_enter_long(snapshot, config):
        stop_distance = clamp_stop_distance(abs(snapshot.forecast_return), adjusted_config)
        tp_distance = stop_distance * adjusted_config.take_profit_multiplier
        stop_price = snapshot.current_price * (1.0 - stop_distance)
        tp_price = snapshot.current_price * (1.0 + tp_distance)
        return PaperDecision(
            action="enter_long",
            reason=f"long_signal_confirmed ({regime})",
            stop_distance_pct=stop_distance,
            stop_price=stop_price,
            tp_distance_pct=tp_distance,
            take_profit_price=tp_price,
        )

    if should_enter_short(snapshot, config):
        stop_distance = clamp_stop_distance(abs(snapshot.forecast_return), adjusted_config)
        tp_distance = stop_distance * adjusted_config.take_profit_multiplier
        stop_price = snapshot.current_price * (1.0 + stop_distance)
        tp_price = snapshot.current_price * (1.0 - tp_distance)
        return PaperDecision(
            action="enter_short",
            reason=f"short_signal_confirmed ({regime})",
            stop_distance_pct=stop_distance,
            stop_price=stop_price,
            tp_distance_pct=tp_distance,
            take_profit_price=tp_price,
        )

    return PaperDecision(action="no_action", reason=f"entry_conditions_not_met ({regime})")


def build_exit_decision(
    position: PaperPosition,
    snapshot: PaperSignalSnapshot,
    config: PaperStrategyConfig,
    bar_interval_hours: float = 4.0,
) -> PaperDecision:
    bars_held = 0
    if hasattr(position, 'entry_time') and position.entry_time is not None:
        time_diff = snapshot.signal_time - position.entry_time
        hours_held = time_diff.total_seconds() / 3600
        if bar_interval_hours > 0:
            bars_held = int(hours_held / bar_interval_hours)

    regime = get_market_regime(snapshot)
    adjusted_config = get_regime_adjusted_config(regime, config)
    
    val_count_valid = sum([
        snapshot.validation_pred_return > 0,
        snapshot.validation_pred_delta > 0,
        snapshot.validation_history_delta > 0,
    ]) if position.side == "long" else sum([
        snapshot.validation_pred_return < 0,
        snapshot.validation_pred_delta < 0,
        snapshot.validation_history_delta < 0,
    ])

    if position.side == "long":
        if bars_held >= adjusted_config.max_bars_in_position:
            return PaperDecision(action="exit_long", reason=f"time_exit ({regime})")
        if snapshot.current_price <= position.stop_price:
            return PaperDecision(action="exit_long", reason="stop_loss_hit")
        if snapshot.current_price >= position.take_profit_price:
            return PaperDecision(action="exit_long", reason="take_profit_hit")
        if val_count_valid < adjusted_config.min_validation_conditions:
            return PaperDecision(action="exit_long", reason="long_signal_invalidated")
        return PaperDecision(action="hold", reason="long_position_still_valid")

    if position.side == "short":
        if bars_held >= adjusted_config.max_bars_in_position:
            return PaperDecision(action="exit_short", reason=f"time_exit ({regime})")
        if snapshot.current_price >= position.stop_price:
            return PaperDecision(action="exit_short", reason="stop_loss_hit")
        if snapshot.current_price <= position.take_profit_price:
            return PaperDecision(action="exit_short", reason="take_profit_hit")
        if val_count_valid < adjusted_config.min_validation_conditions:
            return PaperDecision(action="exit_short", reason="short_signal_invalidated")
        return PaperDecision(action="hold", reason="short_position_still_valid")

    return PaperDecision(action="no_action", reason="unknown_position_side")
