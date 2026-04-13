# Kronos BTC Prediction GUI

這個 GUI 會直接從 Binance 期貨拉 OHLCV，先用最新 `lookback` K 線做未來預測，再用最近一段已收盤資料做一輪 validation 指標。

## How to Run

建議先安裝核心依賴，再安裝 GUI 依賴：

```bash
cd /Users/ppppp/Desktop/workspace/Kronos_gui
python3 -m pip install -r requirements.txt
python3 -m pip install -r qtgui/requirements.txt
```

接著啟動 GUI：

```bash
cd /Users/ppppp/Desktop/workspace/Kronos_gui/qtgui
python3 kronos_gui.py
```


## Backend Behavior

- GUI 下拉選單可直接切換本地模型
- 目前支援本地 `model/kronos-mlx-base`、`model/kronos-small`、`model/kronos-mini`
- `kronos-mlx-base` 需要環境裡有 `kronos_mlx`
- `small` / `mini` 使用本地 PyTorch 權重與本地 tokenizer
- 真實下單沒有實作，現在只做資料抓取與行情預測

## Features

- 修正左側控制面板顯示問題
- 使用背景執行緒載入模型 / 抓資料 / 跑預測，避免直接從 worker thread 更新 Qt widget
- 支援 Binance futures: `BTC/USDT`, `ETH/USDT`, `BNB/USDT`, `SOL/USDT`
- 支援 `5m`, `15m`, `1h`, `4h`, `1d`
- 顯示 forward forecast 與最近一段 validation 的 MAE / RMSE / MAPE

## Recommended Starting Point

- Symbol: `BTC/USDT`
- Timeframe: `4h`
- Lookback: `280`
- Prediction Length: `18`
- Temperature: `0.5`
- Top P: `0.8`
- Sample Count: `3`
