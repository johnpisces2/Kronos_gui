#!/usr/bin/env python3
"""Comprehensive analysis of model predictions vs actual market movements."""

import argparse
import os
import sys
from typing import Iterable, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

DISPLAY_TIMEZONE = "Asia/Taipei"
LOOKBACK = 440
PRED_LEN = 72
TEMPERATURE = 0.5
TOP_P = 0.8
DEFAULT_FEE_RATE = 0.0005
DEFAULT_SLIPPAGE_PCT = 0.001


def to_display_timestamp_series(values, assume_utc=False):
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


def infer_time_delta(timestamps):
    ts = ensure_timestamp_series(timestamps)
    diffs = ts.diff().dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]
    if diffs.empty:
        return pd.Timedelta(hours=1)
    return diffs.median()


def build_future_timestamps(last_timestamp, step, pred_len):
    future = [last_timestamp + step * idx for idx in range(1, pred_len + 1)]
    return pd.Series(pd.to_datetime(future))


def fetch_binance_data(symbol, timeframe, start_str, end_str):
    import ccxt

    exchange = ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })

    since = int(pd.Timestamp(start_str, tz="Asia/Shanghai").timestamp() * 1000)
    now = int(pd.Timestamp(end_str + " 23:59:59", tz="Asia/Shanghai").timestamp() * 1000)

    all_ohlcv = []
    current_since = since

    print(f"Fetching {symbol} {timeframe} from {start_str} to {end_str}...")

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
        except Exception as exc:
            print(f"Fetch error: {exc}")
            break

    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamps"] = to_display_timestamp_series(
        pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    )
    df = df.drop(columns=["timestamp"]).reset_index(drop=True)

    print(f"Downloaded {len(df)} candles")
    return df


def direction_label(value: float):
    if pd.isna(value):
        return pd.NA
    if value > 0:
        return "UP"
    if value < 0:
        return "DOWN"
    return "FLAT"


def sign_match(predicted: pd.Series, actual: pd.Series):
    predicted_sign = np.sign(predicted)
    actual_sign = np.sign(actual)
    matches = predicted_sign == actual_sign
    matches = pd.Series(matches, index=predicted.index, dtype="boolean")
    matches[predicted.isna() | actual.isna()] = pd.NA
    return matches


def calculate_trade_return_columns(df_analysis, fee_rate, slippage_pct):
    price_ratio = 1 + df_analysis["actual_future_return"]

    long_after_cost = (
        price_ratio * (1 - slippage_pct) * (1 - fee_rate)
        / ((1 + slippage_pct) * (1 + fee_rate))
    ) - 1
    short_after_cost = (
        ((1 - slippage_pct) * (1 - fee_rate))
        / (price_ratio * (1 + slippage_pct) * (1 + fee_rate))
    ) - 1

    df_analysis["trade_return_long_before_cost"] = df_analysis["actual_future_return"]
    df_analysis["trade_return_short_before_cost"] = -df_analysis["actual_future_return"]
    df_analysis["trade_return_long_after_cost"] = long_after_cost
    df_analysis["trade_return_short_after_cost"] = short_after_cost

    return df_analysis


def enrich_prediction_analysis(df_analysis, pred_len, fee_rate, slippage_pct):
    df_analysis = df_analysis.copy()
    df_analysis["timestamp"] = pd.to_datetime(df_analysis["timestamp"])
    sort_keys = [col for col in ("bar_index", "timestamp") if col in df_analysis.columns]
    if sort_keys:
        df_analysis = df_analysis.sort_values(sort_keys).reset_index(drop=True)

    df_analysis["forecast_direction"] = df_analysis["forecast_return"].map(direction_label)
    df_analysis["actual_future_direction"] = df_analysis["actual_future_return"].map(direction_label)
    df_analysis["prediction_correct"] = sign_match(
        df_analysis["forecast_return"],
        df_analysis["actual_future_return"],
    )

    validation_reference = df_analysis["current_price"].shift(pred_len)
    actual_validation_return = pd.Series(np.nan, index=df_analysis.index, dtype="float64")
    valid_mask = validation_reference.notna() & (validation_reference != 0)
    actual_validation_return.loc[valid_mask] = (
        df_analysis.loc[valid_mask, "current_price"] / validation_reference.loc[valid_mask] - 1
    )
    df_analysis["actual_validation_return"] = actual_validation_return
    df_analysis["actual_validation_direction"] = actual_validation_return.map(direction_label)
    df_analysis["val_pred_direction"] = df_analysis["val_pred_return"].map(direction_label)
    df_analysis["val_prediction_correct"] = sign_match(
        df_analysis["val_pred_return"],
        df_analysis["actual_validation_return"],
    )

    df_analysis["val_bullish_count"] = (
        (df_analysis["val_pred_return"] > 0).astype(int)
        + (df_analysis["val_pred_delta"] > 0).astype(int)
        + (df_analysis["val_hist_delta"] > 0).astype(int)
    )
    df_analysis["val_bearish_count"] = (
        (df_analysis["val_pred_return"] < 0).astype(int)
        + (df_analysis["val_pred_delta"] < 0).astype(int)
        + (df_analysis["val_hist_delta"] < 0).astype(int)
    )

    return calculate_trade_return_columns(df_analysis, fee_rate, slippage_pct)


def analyze_predictions(df, predictor, lookback, pred_len, fee_rate, slippage_pct):
    """Analyze all predictions vs actual outcomes."""

    columns = ["open", "high", "low", "close", "volume"]
    records = []

    step = infer_time_delta(df["timestamps"])

    print(f"\nAnalyzing {len(df) - lookback} bars...")

    for i in range(lookback, len(df) - pred_len):
        if i % 50 == 0:
            print(f"  Progress: {i - lookback}/{len(df) - lookback - pred_len}")

        context_df = df.iloc[i - lookback + 1:i + 1][columns].copy()
        x_ts = ensure_timestamp_series(df["timestamps"].iloc[i - lookback + 1:i + 1])
        future_ts = build_future_timestamps(df["timestamps"].iloc[i], step, pred_len)

        try:
            pred_result = predictor.predict(
                df=context_df,
                x_timestamp=x_ts,
                y_timestamp=future_ts,
                pred_len=pred_len,
                T=TEMPERATURE,
                top_p=TOP_P,
                sample_count=1,
                verbose=False,
            )
            future_pred_df = pred_result.reset_index().rename(columns={"index": "timestamps"})
        except Exception as exc:
            print(f"Prediction error at bar {i}: {exc}")
            continue

        current_price = float(df["close"].iloc[i])
        forecast_price = float(future_pred_df["close"].iloc[-1])
        forecast_return = (forecast_price / current_price - 1)

        val_context_start = max(0, i - lookback - pred_len + 1)
        val_context_end = i - pred_len + 1
        if val_context_end <= val_context_start:
            val_context_start = max(0, val_context_end - lookback)

        if val_context_end > val_context_start:
            val_context_df = df.iloc[val_context_start:val_context_end][columns].copy()
            val_x_ts = ensure_timestamp_series(df["timestamps"].iloc[val_context_start:val_context_end])
            val_future_ts = build_future_timestamps(df["timestamps"].iloc[val_context_end - 1], step, pred_len)

            try:
                val_pred_result = predictor.predict(
                    df=val_context_df,
                    x_timestamp=val_x_ts,
                    y_timestamp=val_future_ts,
                    pred_len=pred_len,
                    T=TEMPERATURE,
                    top_p=TOP_P,
                    sample_count=1,
                    verbose=False,
                )
                val_pred_df = val_pred_result.reset_index().rename(columns={"index": "timestamps"})
            except Exception:
                val_pred_df = None
        else:
            val_pred_df = None

        val_start_price = float(df["close"].iloc[max(0, i - pred_len)])
        val_hist_delta = (
            float(df["close"].iloc[i - pred_len]) - float(df["close"].iloc[max(0, i - pred_len - 1)])
            if i >= pred_len else 0
        )

        val_pred_return = 0.0
        val_pred_delta = 0.0
        if val_pred_df is not None and len(val_pred_df) > 0:
            val_pred_return = (float(val_pred_df["close"].iloc[-1]) / val_start_price - 1)
            val_pred_delta = (
                float(val_pred_df["close"].iloc[-1]) - float(val_pred_df["close"].iloc[-2])
                if len(val_pred_df) > 1 else 0.0
            )

        actual_future_return = (
            float(df["close"].iloc[i + pred_len]) / current_price - 1
            if i + pred_len < len(df) else np.nan
        )
        actual_validation_return = (
            current_price / float(df["close"].iloc[i - pred_len]) - 1
            if i >= pred_len else np.nan
        )

        records.append({
            "bar_index": i,
            "timestamp": df["timestamps"].iloc[i],
            "current_price": current_price,
            "forecast_price": forecast_price,
            "forecast_return": forecast_return,
            "forecast_direction": direction_label(forecast_return),
            "val_pred_return": val_pred_return,
            "val_pred_direction": direction_label(val_pred_return),
            "val_pred_delta": val_pred_delta,
            "val_hist_delta": val_hist_delta,
            "actual_future_return": actual_future_return,
            "actual_future_direction": direction_label(actual_future_return),
            "actual_validation_return": actual_validation_return,
            "actual_validation_direction": direction_label(actual_validation_return),
            "prediction_correct": sign_match(pd.Series([forecast_return]), pd.Series([actual_future_return])).iloc[0],
            "val_prediction_correct": sign_match(pd.Series([val_pred_return]), pd.Series([actual_validation_return])).iloc[0],
        })

    df_analysis = pd.DataFrame(records)
    return enrich_prediction_analysis(df_analysis, pred_len, fee_rate, slippage_pct)


def select_non_overlapping_signals(signals: pd.DataFrame, pred_len: int):
    selected_indices = []
    last_bar_index = None

    for idx, row in signals.sort_values("bar_index").iterrows():
        bar_index = int(row["bar_index"])
        if last_bar_index is None or bar_index - last_bar_index >= pred_len:
            selected_indices.append(idx)
            last_bar_index = bar_index

    return signals.loc[selected_indices].copy()


def summarize_signal_subset(signals: pd.DataFrame, before_cost_col: str, after_cost_col: str):
    if signals.empty:
        return {
            "signal_count": 0,
            "win_rate_before_cost": np.nan,
            "win_rate_after_cost": np.nan,
            "avg_return_before_cost": np.nan,
            "avg_return_after_cost": np.nan,
            "median_return_before_cost": np.nan,
            "median_return_after_cost": np.nan,
            "total_return_before_cost": np.nan,
            "total_return_after_cost": np.nan,
        }

    return {
        "signal_count": int(len(signals)),
        "win_rate_before_cost": float((signals[before_cost_col] > 0).mean()),
        "win_rate_after_cost": float((signals[after_cost_col] > 0).mean()),
        "avg_return_before_cost": float(signals[before_cost_col].mean()),
        "avg_return_after_cost": float(signals[after_cost_col].mean()),
        "median_return_before_cost": float(signals[before_cost_col].median()),
        "median_return_after_cost": float(signals[after_cost_col].median()),
        "total_return_before_cost": float((1 + signals[before_cost_col]).prod() - 1),
        "total_return_after_cost": float((1 + signals[after_cost_col]).prod() - 1),
    }


def analyze_threshold_effectiveness(df_analysis, long_thresholds, short_thresholds, pred_len):
    """Analyze how different thresholds affect raw and tradable outcomes."""

    results = []

    for direction, thresholds in (("LONG", long_thresholds), ("SHORT", short_thresholds)):
        threshold_sign = 1 if direction == "LONG" else -1
        validation_col = "val_bullish_count" if direction == "LONG" else "val_bearish_count"
        before_cost_col = "trade_return_long_before_cost" if direction == "LONG" else "trade_return_short_before_cost"
        after_cost_col = "trade_return_long_after_cost" if direction == "LONG" else "trade_return_short_after_cost"

        for th in thresholds:
            for val_th in (1, 2, 3):
                if direction == "LONG":
                    mask = df_analysis["forecast_return"] >= th / 100
                else:
                    mask = df_analysis["forecast_return"] <= -th / 100

                signals = df_analysis[mask & (df_analysis[validation_col] >= val_th)].copy()
                non_overlap = select_non_overlapping_signals(signals, pred_len)

                raw_stats = summarize_signal_subset(signals, before_cost_col, after_cost_col)
                non_overlap_stats = summarize_signal_subset(non_overlap, before_cost_col, after_cost_col)

                results.append({
                    "direction": direction,
                    "forecast_th_pct": th,
                    "validation_min_count": val_th,
                    "signal_count_raw": raw_stats["signal_count"],
                    "win_rate_before_cost_raw": raw_stats["win_rate_before_cost"],
                    "win_rate_after_cost_raw": raw_stats["win_rate_after_cost"],
                    "avg_return_before_cost_raw": raw_stats["avg_return_before_cost"],
                    "avg_return_after_cost_raw": raw_stats["avg_return_after_cost"],
                    "median_return_before_cost_raw": raw_stats["median_return_before_cost"],
                    "median_return_after_cost_raw": raw_stats["median_return_after_cost"],
                    "total_return_before_cost_raw": raw_stats["total_return_before_cost"],
                    "total_return_after_cost_raw": raw_stats["total_return_after_cost"],
                    "signal_count_nonoverlap": non_overlap_stats["signal_count"],
                    "win_rate_before_cost_nonoverlap": non_overlap_stats["win_rate_before_cost"],
                    "win_rate_after_cost_nonoverlap": non_overlap_stats["win_rate_after_cost"],
                    "avg_return_before_cost_nonoverlap": non_overlap_stats["avg_return_before_cost"],
                    "avg_return_after_cost_nonoverlap": non_overlap_stats["avg_return_after_cost"],
                    "median_return_before_cost_nonoverlap": non_overlap_stats["median_return_before_cost"],
                    "median_return_after_cost_nonoverlap": non_overlap_stats["median_return_after_cost"],
                    "total_return_before_cost_nonoverlap": non_overlap_stats["total_return_before_cost"],
                    "total_return_after_cost_nonoverlap": non_overlap_stats["total_return_after_cost"],
                })

    return pd.DataFrame(results)


def print_top_thresholds(threshold_results: pd.DataFrame, direction: str):
    subset = threshold_results[threshold_results["direction"] == direction].copy()
    subset = subset[subset["signal_count_nonoverlap"] > 0]
    if subset.empty:
        print(f"\n--- {direction} Tradable Threshold Analysis ---")
        print("No qualifying signals.")
        return

    ordered = subset.sort_values(
        ["total_return_after_cost_nonoverlap", "avg_return_after_cost_nonoverlap"],
        ascending=False,
    )
    print(f"\n--- {direction} Tradable Threshold Analysis (sorted by non-overlap after-cost return) ---")
    print(ordered.head(10).to_string(index=False))


def print_basic_statistics(df_analysis, pred_len):
    print("\n" + "=" * 80)
    print("PHASE 2: Basic Statistics")
    print("=" * 80)

    print(f"\nTotal bars analyzed: {len(df_analysis)}")
    print(f"Date range: {df_analysis['timestamp'].iloc[0]} to {df_analysis['timestamp'].iloc[-1]}")

    print("\n--- Forecast Statistics ---")
    print(f"Mean forecast return: {df_analysis['forecast_return'].mean() * 100:+.2f}%")
    print(f"Std forecast return: {df_analysis['forecast_return'].std() * 100:.2f}%")
    print(f"Min forecast return: {df_analysis['forecast_return'].min() * 100:+.2f}%")
    print(f"Max forecast return: {df_analysis['forecast_return'].max() * 100:+.2f}%")
    print(f"Positive forecasts: {(df_analysis['forecast_return'] > 0).sum()} / {len(df_analysis)} ({(df_analysis['forecast_return'] > 0).mean() * 100:.1f}%)")
    print(f"Negative forecasts: {(df_analysis['forecast_return'] < 0).sum()} / {len(df_analysis)} ({(df_analysis['forecast_return'] < 0).mean() * 100:.1f}%)")

    print("\n--- Prediction Accuracy ---")
    print(f"Forecast direction accuracy: {df_analysis['prediction_correct'].mean() * 100:.1f}%")
    print(f"Validation direction accuracy: {df_analysis['val_prediction_correct'].dropna().mean() * 100:.1f}%")

    print(f"\n--- Actual Future Returns (next {pred_len} bars) ---")
    print(f"Mean actual return: {df_analysis['actual_future_return'].mean() * 100:+.2f}%")
    print(f"Std: {df_analysis['actual_future_return'].std() * 100:.2f}%")
    print(f"Actual UP: {(df_analysis['actual_future_return'] > 0).sum()}")
    print(f"Actual DOWN: {(df_analysis['actual_future_return'] < 0).sum()}")

    print("\n--- Tradable Returns After Costs ---")
    print(f"Long avg after-cost return: {df_analysis['trade_return_long_after_cost'].mean() * 100:+.2f}%")
    print(f"Short avg after-cost return: {df_analysis['trade_return_short_after_cost'].mean() * 100:+.2f}%")


def print_regime_analysis(df_analysis):
    print("\n" + "=" * 80)
    print("PHASE 4: Regime Analysis")
    print("=" * 80)

    df_analysis = df_analysis.copy()
    df_analysis["regime"] = "NEUTRAL"
    df_analysis.loc[
        (df_analysis["forecast_return"] > 0.02)
        & (df_analysis["val_pred_return"] > 0)
        & (df_analysis["val_pred_delta"] > 0)
        & (df_analysis["val_hist_delta"] > 0),
        "regime"
    ] = "STRONG_BULL"
    df_analysis.loc[
        (df_analysis["forecast_return"] > 0.02)
        & (df_analysis["val_pred_return"] > 0)
        & ((df_analysis["val_pred_delta"] > 0) | (df_analysis["val_hist_delta"] > 0)),
        "regime"
    ] = "BULL"
    df_analysis.loc[
        (df_analysis["forecast_return"] < -0.02)
        & (df_analysis["val_pred_return"] < 0)
        & (df_analysis["val_pred_delta"] < 0)
        & (df_analysis["val_hist_delta"] < 0),
        "regime"
    ] = "STRONG_BEAR"
    df_analysis.loc[
        (df_analysis["forecast_return"] < -0.02)
        & (df_analysis["val_pred_return"] < 0)
        & ((df_analysis["val_pred_delta"] < 0) | (df_analysis["val_hist_delta"] < 0)),
        "regime"
    ] = "BEAR"

    print("\n--- Regime Distribution ---")
    regime_counts = df_analysis["regime"].value_counts()
    for regime, count in regime_counts.items():
        pct = count / len(df_analysis) * 100
        print(f"  {regime}: {count} ({pct:.1f}%)")

    print("\n--- Regime Actual Performance ---")
    for regime in ("STRONG_BULL", "BULL", "NEUTRAL", "BEAR", "STRONG_BEAR"):
        regime_df = df_analysis[df_analysis["regime"] == regime]
        if len(regime_df) == 0:
            continue
        actual_up_pct = (regime_df["actual_future_return"] > 0).mean() * 100
        avg_return = regime_df["actual_future_return"].mean() * 100
        print(f"\n{regime}:")
        print(f"  Count: {len(regime_df)}")
        print(f"  Actual UP %: {actual_up_pct:.1f}%")
        print(f"  Avg actual return: {avg_return:+.2f}%")
        print(f"  Forecast avg: {regime_df['forecast_return'].mean() * 100:+.2f}%")
        print(f"  Val_pred avg: {regime_df['val_pred_return'].mean() * 100:+.2f}%")


def print_validation_analysis(df_analysis):
    print("\n" + "=" * 80)
    print("PHASE 5: Validation Conditions Analysis")
    print("=" * 80)

    print("\n--- Validation Condition Count vs Actual Direction ---")
    for val_count in (1, 2, 3):
        bullish_subset = df_analysis[df_analysis["val_bullish_count"] >= val_count]
        bearish_subset = df_analysis[df_analysis["val_bearish_count"] >= val_count]

        if not bullish_subset.empty:
            accuracy = (bullish_subset["actual_future_return"] > 0).mean() * 100
            avg_return = bullish_subset["actual_future_return"].mean() * 100
            print(f"  BULLISH {val_count}/3: count={len(bullish_subset)}, accuracy={accuracy:.1f}%, avg_return={avg_return:+.2f}%")

        if not bearish_subset.empty:
            accuracy = (bearish_subset["actual_future_return"] < 0).mean() * 100
            avg_return = bearish_subset["actual_future_return"].mean() * 100
            print(f"  BEARISH {val_count}/3: count={len(bearish_subset)}, accuracy={accuracy:.1f}%, avg_return={avg_return:+.2f}%")


def parse_thresholds(values: Iterable[int]):
    return [int(value) for value in values]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", help="Reuse an existing prediction_analysis.csv instead of rerunning the model.")
    parser.add_argument("--output-prediction", default="prediction_analysis.csv")
    parser.add_argument("--output-threshold", default="threshold_analysis.csv")
    parser.add_argument("--lookback", type=int, default=LOOKBACK)
    parser.add_argument("--pred-len", type=int, default=PRED_LEN)
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE)
    parser.add_argument("--slippage-pct", type=float, default=DEFAULT_SLIPPAGE_PCT)
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--long-thresholds", nargs="*", type=int, default=[1, 2, 3, 4, 5, 6, 8, 10])
    parser.add_argument("--short-thresholds", nargs="*", type=int, default=[1, 2, 3, 4, 5, 6, 8, 10])
    return parser.parse_args()


def load_predictor():
    from kronos_mlx import Kronos as MLXKronos
    from kronos_mlx import KronosTokenizer as MLXTokenizer
    from kronos_mlx import KronosPredictor as MLXPredictor

    tokenizer = MLXTokenizer.from_pretrained(os.path.join(PROJECT_ROOT, "model", "kronos-mlx-tokenizer-base"))
    model = MLXKronos.from_pretrained(os.path.join(PROJECT_ROOT, "model", "kronos-mlx-base"), bits=8)
    return MLXPredictor(model, tokenizer, max_context=512)


def main():
    args = parse_args()

    print("=" * 80)
    print("COMPREHENSIVE PREDICTION ANALYSIS")
    print("=" * 80)
    print(f"LOOKBACK: {args.lookback}, PRED_LEN: {args.pred_len}")
    print(f"Fee rate: {args.fee_rate:.4f}, Slippage: {args.slippage_pct:.4f}")
    print("=" * 80)

    if args.input_csv:
        print(f"\nLoading existing prediction data from {args.input_csv}...")
        df_analysis = pd.read_csv(args.input_csv)
        df_analysis = enrich_prediction_analysis(
            df_analysis,
            pred_len=args.pred_len,
            fee_rate=args.fee_rate,
            slippage_pct=args.slippage_pct,
        )
    else:
        end_date = pd.Timestamp.now(tz=DISPLAY_TIMEZONE)
        end_str = end_date.strftime("%Y-%m-%d")
        start_date = end_date - pd.Timedelta(days=args.days)
        start_str = start_date.strftime("%Y-%m-%d")

        print(f"\nPeriod: {start_str} to {end_str} ({args.days} days)")
        df = fetch_binance_data(args.symbol, args.timeframe, start_str, end_str)

        print("\nLoading MLX model...")
        predictor = load_predictor()
        print("Model loaded.")

        print("\n" + "=" * 80)
        print("PHASE 1: Collecting prediction data...")
        print("=" * 80)

        df_analysis = analyze_predictions(
            df=df,
            predictor=predictor,
            lookback=args.lookback,
            pred_len=args.pred_len,
            fee_rate=args.fee_rate,
            slippage_pct=args.slippage_pct,
        )

    df_analysis.to_csv(args.output_prediction, index=False)
    print(f"\nSaved {len(df_analysis)} records to {args.output_prediction}")

    print_basic_statistics(df_analysis, args.pred_len)

    print("\n" + "=" * 80)
    print("PHASE 3: Threshold Effectiveness Analysis")
    print("=" * 80)

    threshold_results = analyze_threshold_effectiveness(
        df_analysis=df_analysis,
        long_thresholds=parse_thresholds(args.long_thresholds),
        short_thresholds=parse_thresholds(args.short_thresholds),
        pred_len=args.pred_len,
    )
    threshold_results.to_csv(args.output_threshold, index=False)
    print(f"Saved {len(threshold_results)} rows to {args.output_threshold}")

    print_top_thresholds(threshold_results, "LONG")
    print_top_thresholds(threshold_results, "SHORT")

    print_regime_analysis(df_analysis)
    print_validation_analysis(df_analysis)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Forecast direction accuracy: {df_analysis['prediction_correct'].mean() * 100:.1f}%")
    print(f"Validation direction accuracy: {df_analysis['val_prediction_correct'].dropna().mean() * 100:.1f}%")
    print(f"Long average after-cost return: {df_analysis['trade_return_long_after_cost'].mean() * 100:+.2f}%")
    print(f"Short average after-cost return: {df_analysis['trade_return_short_after_cost'].mean() * 100:+.2f}%")
    print("Files generated:")
    print(f"  - {args.output_prediction}")
    print(f"  - {args.output_threshold}")


if __name__ == "__main__":
    main()
