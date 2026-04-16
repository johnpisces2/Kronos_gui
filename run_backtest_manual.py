#!/usr/bin/env python3
"""Manual backtest script for Kronos BTC prediction."""

import os
import sys
import ccxt
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from kronos_mlx import Kronos as MLXKronos, KronosTokenizer as MLXTokenizer, KronosPredictor as MLXPredictor
from backtester import Backtester

LOOKBACK = 480
PRED_LEN = 18
TEMPERATURE = 0.5
TOP_P = 0.8
SAMPLE_COUNT = 3

DISPLAY_TIMEZONE = "Asia/Taipei"

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

def load_mlx_model():
    model_path = os.path.join(PROJECT_ROOT, "model", "kronos-mlx-base")
    tokenizer_path = os.path.join(PROJECT_ROOT, "model", "kronos-mlx-tokenizer-base")
    
    print(f"Loading MLX model from {model_path}...")
    tokenizer = MLXTokenizer.from_pretrained(tokenizer_path)
    model = MLXKronos.from_pretrained(model_path, bits=8)
    predictor = MLXPredictor(model, tokenizer, max_context=512)
    return predictor

def fetch_binance_data(symbol, timeframe, start_str, end_str):
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
        except Exception as e:
            print(f"Fetch error: {e}")
            break
    
    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamps"] = to_display_timestamp_series(
        pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    )
    df = df.drop(columns=["timestamp"]).reset_index(drop=True)
    
    print(f"Downloaded {len(df)} candles")
    return df

def main():
    if len(sys.argv) > 1:
        end_str = sys.argv[1]
    else:
        end_date = pd.Timestamp.now(tz=DISPLAY_TIMEZONE)
        end_str = end_date.strftime("%Y-%m-%d")
    
    if len(sys.argv) > 2:
        start_str = sys.argv[2]
    else:
        start_date = pd.Timestamp(end_str, tz=DISPLAY_TIMEZONE) - pd.Timedelta(days=180)
        start_str = start_date.strftime("%Y-%m-%d")
    
    print(f"\n=== Backtest Period: {start_str} to {end_str} ===\n")
    
    symbol = "BTC/USDT"
    timeframe = "4h"
    
    df = fetch_binance_data(symbol, timeframe, start_str, end_str)
    
    context_df = df.copy()
    total_len = len(context_df)
    print(f"Total bars: {total_len}")
    
    val_start_idx = max(LOOKBACK, int(total_len * 0.7))
    val_history_df = context_df.iloc[:val_start_idx].copy()
    val_actual_df = context_df.iloc[val_start_idx:].copy()
    print(f"Validation split: {val_start_idx} history, {len(val_actual_df)} actual bars")
    
    predictor = load_mlx_model()
    print("Model loaded successfully\n")
    
    def predictor_fn(context_df_for_pred, temperature=TEMPERATURE, top_p=TOP_P, sample_count=SAMPLE_COUNT):
        columns = ["open", "high", "low", "close", "volume"]
        x_df = context_df_for_pred[columns].copy()
        x_ts = ensure_timestamp_series(context_df_for_pred["timestamps"])
        
        step = infer_time_delta(context_df_for_pred["timestamps"])
        future_ts = build_future_timestamps(x_ts.iloc[-1], step, PRED_LEN)
        
        try:
            future_pred_df = predictor.predict(
                df=x_df,
                x_timestamp=x_ts,
                y_timestamp=future_ts,
                pred_len=PRED_LEN,
                T=temperature,
                top_p=top_p,
                sample_count=1,
                verbose=False,
            ).reset_index().rename(columns={"index": "timestamps"})
            return {"future_pred_df": future_pred_df}
        except Exception as e:
            print(f"Prediction error: {e}")
            return {"future_pred_df": None}
    
    forecast_params = {
        "lookback": LOOKBACK,
        "pred_len": PRED_LEN,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "sample_count": SAMPLE_COUNT,
        "val_compute_interval": 1,
    }
    
    backtester = Backtester(
        initial_equity=10000.0,
        fee_rate=0.0005,
        slippage_pct=0.001,
    )
    
    total_validation_bars = len(context_df) - LOOKBACK
    print(f"Backtest: {LOOKBACK} lookback, {total_validation_bars} bars to predict\n")
    print("Running backtest (progress every 50 bars)...\n")
    
    progress_count = [0]
    
    def progress_callback(current, total):
        progress_count[0] = current
        if current % 50 == 0:
            print(f"[PROGRESS] {current}/{total} bars processed...")
        return True
    
    results = backtester.run(
        context_df=context_df,
        predictor_fn=predictor_fn,
        forecast_params=forecast_params,
        debug=True,
        progress_callback=progress_callback,
    )
    
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    print(f"Period:           {start_str} to {end_str}")
    print(f"Lookback:         {LOOKBACK}")
    print(f"Pred Len:         {PRED_LEN}")
    print(f"Temperature:      {TEMPERATURE}")
    print(f"Initial Equity:   ${backtester.initial_equity:,.2f}")
    print("-"*60)
    print(f"Total Trades:     {results.total_trades}")
    print(f"Winning Trades:   {results.winning_trades}")
    print(f"Losing Trades:    {results.losing_trades}")
    print(f"Win Rate:         {results.win_rate*100:.2f}%")
    print("-"*60)
    print(f"Total P&L:        ${results.total_pnl:,.2f}")
    print(f"Total Return:     {results.total_return_pct:.2f}%")
    print(f"Avg Win:          ${results.avg_win:,.2f}")
    print(f"Avg Loss:         ${results.avg_loss:,.2f}")
    print(f"Profit Factor:    {results.profit_factor:.3f}")
    print("-"*60)
    print(f"Max Drawdown:     ${results.max_drawdown:,.2f} ({results.max_drawdown_pct*100:.2f}%)")
    print(f"Sharpe Ratio:     {results.sharpe_ratio:.3f}")
    print(f"Sortino Ratio:    {results.sortino_ratio:.3f}")
    print(f"Time in Market:   {results.time_in_market_pct*100:.2f}%")
    print("-"*60)
    print(f"Total Fees:       ${results.total_fees:,.2f}")
    print(f"Total Slippage:   ${results.total_slippage:,.2f}")
    print("="*60)

if __name__ == "__main__":
    main()