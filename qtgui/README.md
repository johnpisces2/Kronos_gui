# Kronos BTC Forecast Terminal

Kronos BTC Forecast Terminal 是一個基於 Qt 的圖形化桌面應用，直接從 Binance 期貨取得 OHLCV 資料，利用 Kronos 模型進行價格預測，並透過最近已收盤的 K 線進行模型驗證（Backtesting）。

## 系統需求

- Python 3.10+
- PyQt5 或 PyQt6
- torch (PyTorch)
- 相關依賴套件（見 requirements.txt）

## 安裝與執行

### 1. 安裝依賴

```bash
cd /path/to/Kronos_gui
pip install -r requirements.txt
```

### 2. 啟動 GUI

```bash
cd /path/to/Kronos_gui/qtgui
python kronos_gui.py
```

## 支援的模型

| 模型 | Backend | 說明 |
|------|---------|------|
| Kronos-base (MLX 8-bit) | MLX | Apple Silicon 優化版本 |
| Kronos-small (PyTorch) | PyTorch | 平衡品質與速度 |
| Kronos-mini (PyTorch) | PyTorch | 輕量快速，Context length 最長達 2048 |

## 支援的運算設備

自動偵測順序：**MPS → XPU → CUDA → CPU**

- **MPS**: Apple Silicon GPU
- **XPU**: Intel GPU (需安裝 Intel Extension for PyTorch)
- **CUDA**: NVIDIA GPU
- **CPU**: 備援方案

## 功能特色

### 資料來源
- 直接從 Binance Futures 取得即時 OHLCV 資料
- 支援 Symbol: `BTC/USDT`, `ETH/USDT`, `BNB/USDT`, `SOL/USDT`
- 支援 Timeframe: `5m`, `15m`, `1h`, `4h`, `1d`

### 預測參數
- **Limit**: 歷史 K 線數量上限（120 - 2048）
- **Lookback**: 用於預測的歷史資料筆數
- **Forecast**: 未來預測點數
- **Temp (Temperature)**: 控制預測隨機性（0.1 - 2.0）
- **Top P (Nucleus Sampling)**: 控制預測多樣性（0.1 - 1.0）
- **Samples**: 產生多條預測路徑並取平均（1 - 100）

### 圖表顯示
- **Chart 頁**: 雙圖顯示
  - 上圖：Historical close + Forward forecast（藍色歷史、紫色預測、黃色置信區間）
  - 下圖：Validation（回測驗證 - 灰色歷史、紫色預測、綠色實際值）
- **Log 頁**: 執行日誌與詳細資訊
- 滑鼠懸停顯示十字交叉線與座標價格

### 驗證指標
- **MAE** (Mean Absolute Error): 平均絕對誤差
- **RMSE** (Root Mean Square Error): 均方根誤差
- **MAPE** (Mean Absolute Percentage Error): 平均絕對百分比誤差

## 建議起始參數

| 參數 | 數值 |
|------|------|
| Symbol | BTC/USDT |
| Timeframe | 4h |
| Lookback | 280 |
| Prediction Length | 18 |
| Temperature | 0.5 |
| Top P | 0.8 |
| Sample Count | 3 |

## 資料夾結構

```
Kronos_gui/
├── model/                    # 模型權重與 Tokenizer
│   ├── kronos-mini/         # Mini 模型
│   ├── kronos-small/        # Small 模型
│   └── kronos-mlx-base/     # MLX 模型 (Apple Silicon)
├── qtgui/                   # Qt 圖形化介面
│   └── kronos_gui.py       # 主程式
├── webui/                   # Web 介面
├── examples/                # 範例腳本
├── finetune/               # 微調相關
└── requirements.txt        # 依賴列表
```

## 常見問題

**Q: 為什麼 Validation 圖沒有顯示綠色實際價格線？**  
A: 驗證需要足夠的歷史資料（Lookback + Forecast 長度），請確認 Limit 設定足夠大。

**Q: 支援 Intel GPU 嗎？**  
A: 是的，請安裝 [Intel Extension for PyTorch (IPEX)](https://intel.github.io/intel-extension-for-pytorch/)。

**Q: 如何提高預測品質？**  
A: 可以嘗試：1) 增加 Sample Count 取平均、2) 調整 Temperature、3) 使用更大的模型。
