#!/usr/bin/env python3
"""Manual backtest script with detailed trade analysis."""

import os
import sys
import ccxt
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from kronos_mlx import Kronos as MLXKronos, KronosTokenizer as MLXTokenizer, KronosPredictor as MLXPredictor
from backtester import Backtester

LOOKBACK = 256
PRED_LEN = 18
TEMPERATURE = 0.5
TOP_P = 0.8
SAMPLE_COUNT = 1

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
        start_date = pd.Timestamp(end_str, tz=DISPLAY_TIMEZONE) - pd.Timedelta(days=90)
        start_str = start_date.strftime("%Y-%m-%d")
    
    print(f"\n=== Backtest Period: {start_str} to {end_str} ===\n")
    
    symbol = "BTC/USDT"
    timeframe = "4h"
    
    df = fetch_binance_data(symbol, timeframe, start_str, end_str)
    
    context_df = df.copy()
    total_len = len(context_df)
    print(f"Total bars: {total_len}")
    
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
    
    print(f"Backtest: {LOOKBACK} lookback, {len(context_df) - LOOKBACK} bars to predict\n")
    print("Running backtest...\n")
    
    results = backtester.run(
        context_df=context_df,
        predictor_fn=predictor_fn,
        forecast_params=forecast_params,
        debug=False,
        progress_callback=None,
    )
    
    print("\n" + "="*120)
    print("DETAILED TRADE LOG")
    print("="*120)
    print(f"{'#':<3} {'Entry Time':<20} {'Exit Time':<20} {'Side':<6} {'Entry Price':>12} {'Exit Price':>12} {'Bars':>5} {'Reason':<25} {'P&L ($)':>12} {'Return %':>10}")
    print("-"*120)
    
    for i, trade in enumerate(results.trades, 1):
        entry_time = str(trade.entry_time)[:19]
        exit_time = str(trade.exit_time)[:19] if trade.exit_time else "OPEN"
        side = trade.side.upper()
        entry_price = f"{trade.entry_price:,.2f}"
        exit_price = f"{trade.exit_price:,.2f}" if trade.exit_price else "N/A"
        bars = trade.bars_held if hasattr(trade, 'bars_held') else "N/A"
        reason = trade.reason[:25] if trade.reason else ""
        pnl = f"{trade.pnl_amount:,.2f}" if trade.pnl_amount else "0.00"
        ret = f"{trade.pnl_pct*100:+.2f}%" if trade.pnl_pct else "0.00%"
        
        print(f"{i:<3} {entry_time:<20} {exit_time:<20} {side:<6} {entry_price:>12} {exit_price:>12} {bars:>5} {reason:<25} {pnl:>12} {ret:>10}")
    
    print("="*120)
    print("\nSUMMARY BY SIDE")
    print("-"*60)
    
    longs = [t for t in results.trades if t.side == "long"]
    shorts = [t for t in results.trades if t.side == "short"]
    
    if longs:
        long_wins = [t for t in longs if t.pnl_amount and t.pnl_amount > 0]
        long_loss = [t for t in longs if t.pnl_amount and t.pnl_amount <= 0]
        long_pnl = sum(t.pnl_amount for t in longs if t.pnl_amount)
        print(f"LONG:  {len(longs)} trades ({len(long_wins)} win / {len(long_loss)} loss)")
        print(f"  P&L: ${long_pnl:,.2f}")
        print(f"  Win Rate: {len(long_wins)/len(longs)*100:.1f}%")
        if long_pnl != 0:
            print(f"  Avg Win: ${sum(t.pnl_amount for t in long_wins)/len(long_wins):,.2f}" if long_wins else "  Avg Win: N/A")
            print(f"  Avg Loss: ${sum(t.pnl_amount for t in long_loss)/len(long_loss):,.2f}" if long_loss else "  Avg Loss: N/A")
    
    if shorts:
        short_wins = [t for t in shorts if t.pnl_amount and t.pnl_amount > 0]
        short_loss = [t for t in shorts if t.pnl_amount and t.pnl_amount <= 0]
        short_pnl = sum(t.pnl_amount for t in shorts if t.pnl_amount)
        print(f"SHORT: {len(shorts)} trades ({len(short_wins)} win / {len(short_loss)} loss)")
        print(f"  P&L: ${short_pnl:,.2f}")
        print(f"  Win Rate: {len(short_wins)/len(shorts)*100:.1f}%")
        if short_pnl != 0:
            print(f"  Avg Win: ${sum(t.pnl_amount for t in short_wins)/len(short_wins):,.2f}" if short_wins else "  Avg Win: N/A")
            print(f"  Avg Loss: ${sum(t.pnl_amount for t in short_loss)/len(short_loss):,.2f}" if short_loss else "  Avg Loss: N/A")
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Period:           {start_str} to {end_str}")
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
    print("="*60)

if __name__ == "__main__":
    main()