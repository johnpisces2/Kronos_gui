import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ccxt
import sys
import warnings
import datetime
warnings.filterwarnings('ignore')

sys.path.append("../")
from kronos_mlx import Kronos, KronosTokenizer, KronosPredictor


def get_btc_1h_data_binance():
    """Fetch BTC USDT perpetuals 1h data from Binance using CCXT"""
    print("Fetching BTC-USDT perpetuals 1h data from Binance...")
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })
    
    symbol = 'BTC/USDT'
    timeframe = '1h'
    limit = 600
    
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamps'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.drop('timestamp', axis=1, inplace=True)
    df['amount'] = df['volume'] * df['close'] * 0.0001
    
    print(f"Downloaded {len(df)} rows of BTC 1h data")
    print(f"Date range: {df['timestamps'].min()} to {df['timestamps'].max()}")
    return df


def plot_prediction(kline_df, pred_df, title="BTC Price Prediction"):
    """Plot prediction results"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    hist_len = len(kline_df) - len(pred_df)
    hist_df = kline_df.iloc[:hist_len]
    
    ax1.plot(hist_df['timestamps'], hist_df['close'], 
             label='Historical Close', color='blue', linewidth=1.5)
    ax1.plot(pred_df['timestamps'], pred_df['close'], 
             label='Predicted Close', color='red', linewidth=1.5, linestyle='--')
    ax1.set_ylabel('Price (USD)', fontsize=12)
    ax1.legend(loc='lower left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(title)
    
    if 'volume' in kline_df.columns and 'volume' in pred_df.columns:
        ax2.plot(hist_df['timestamps'], hist_df['volume'], 
                 label='Historical Volume', color='blue', linewidth=1, alpha=0.7)
        ax2.plot(pred_df['timestamps'], pred_df['volume'], 
                 label='Predicted Volume', color='red', linewidth=1, linestyle='--', alpha=0.7)
        ax2.set_ylabel('Volume', fontsize=12)
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)
    
    plt.xlabel('Time', fontsize=12)
    plt.tight_layout()
    plt.savefig('btc_prediction_result.png', dpi=150, bbox_inches='tight')
    print("Chart saved to btc_prediction_result.png")
    plt.show()


def main():
    print("=" * 60)
    print("Kronos BTC Price Prediction (1h timeframe, 3-day forecast)")
    print("=" * 60)
    
    lookback = 440
    pred_len = 72
    
    print(f"\nConfiguration:")
    print(f"  - Lookback (history): {lookback} hours")
    print(f"  - Prediction length:  {pred_len} hours (3 days)")
    print(f"  - Context used:       {lookback + pred_len} / 512")
    
    print("\n[1/4] Loading Kronos model and tokenizer...")
    tokenizer = KronosTokenizer.from_pretrained("gxcsoccer/kronos-mlx-tokenizer-base")
    model = Kronos.from_pretrained("/Users/ppppp/Desktop/workspace/Kronos_gui/model/kronos-mlx-base", bits=8)
    predictor = KronosPredictor(model, tokenizer, max_context=512)
    print("Model loaded successfully! (MLX Kronos-base with 8-bit quantization)")
    
    print("\n[2/4] Fetching BTC 1h data from Binance...")
    df = get_btc_1h_data_binance()
    
    if len(df) < lookback + pred_len:
        print(f"ERROR: Insufficient data. Need at least {lookback + pred_len} rows, got {len(df)}")
        return
    
    print("\n[3/4] Preparing data for prediction...")
    x_df = df.iloc[:lookback][['open', 'high', 'low', 'close', 'volume']]
    if 'amount' not in x_df.columns:
        x_df['amount'] = x_df['volume'] * x_df['close'] * 0.0001
    x_timestamp = df.iloc[:lookback]['timestamps']
    y_timestamp = df.iloc[lookback:lookback+pred_len]['timestamps']
    
    print(f"  - Using data from {x_timestamp.iloc[0]} to {x_timestamp.iloc[-1]}")
    print(f"  - Predicting from {y_timestamp.iloc[0]} to {y_timestamp.iloc[-1]}")
    
    print("\n[4/4] Running prediction...")
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T=1.0,
        top_p=0.9,
        sample_count=1,
        verbose=True
    )
    
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    print(f"\nLast 5 historical candles (close prices):")
    print(x_df['close'].tail().to_string())
    print(f"\nPredicted next 5 candles (close prices):")
    print(pred_df['close'].head().to_string())
    print(f"\nPredicted last 5 candles (close prices):")
    print(pred_df['close'].tail().to_string())
    
    last_hist_close = x_df['close'].iloc[-1]
    last_pred_close = pred_df['close'].iloc[-1]
    change_pct = (last_pred_close - last_hist_close) / last_hist_close * 100
    print(f"\nPrice change prediction: ${last_hist_close:.2f} -> ${last_pred_close:.2f} ({change_pct:+.2f}%)")
    
    kline_df = pd.concat([x_df, pred_df], axis=0)
    kline_df['timestamps'] = pd.concat([x_timestamp, y_timestamp], axis=0).values
    kline_df = kline_df.reset_index(drop=True)
    
    plot_prediction(
        kline_df.reset_index(), 
        pred_df.reset_index(),
        title=f"BTC Price Prediction - 3 Day Forecast\n(Lookback: {lookback}h, Predicted: {pred_len}h)"
    )
    
    pred_df.to_csv('btc_prediction_data.csv', index_label='index')
    print("\nPrediction data saved to btc_prediction_data.csv")


if __name__ == "__main__":
    main()
