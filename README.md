# Kronos BTC Forecast Terminal

Kronos BTC Forecast Terminal 是一個**加密貨幣期貨價格預測系統**，從 Binance Futures 取得 OHLCV 資料，利用 Kronos Foundation Model 進行價格預測，並透過 Backtesting 驗證模型表現。

---

## 功能特色

### 資料來源
- 直接從 Binance Futures 取得**即時 OHLCV 資料**
- 支援 Symbol：`BTC/USDT`、`ETH/USDT`、`BNB/USDT`、`SOL/USDT`
- 支援 Timeframe：`5m`、`15m`、`1h`、`4h`、`1d`

### 預測功能
- **Single Forecast**：單次預測
- **Auto Forecast**：自動預測，根據 Timeframe 自動排程
  - 每次 K 棒收盤時自動執行預測
  - 預測結果自動儲存 Log 與圖表

### 圖表顯示
- **Forecast 頁**：歷史價格 + 預測結果
  - 灰色線：Historical close
  - 黃色區間：MC interval (Monte Carlo Confidence Band)
  - 橙色線：Forecast close
  - 藍色十字線：最新真實價格定位
- **Validation 頁**：回測驗證
  - 灰色線：Validation history
  - 紫色線：Predicted close
  - 綠色線：Actual close
  - 藍色十字線：最新真實價格定位
- **Log 頁**：執行日誌

### 驗證指標
| 指標 | 全名 | 說明 |
|------|------|------|
| MAE | Mean Absolute Error | 平均絕對誤差 |
| RMSE | Root Mean Square Error | 均方根誤差 |
| MAPE | Mean Absolute Percentage Error | 平均絕對百分比誤差 |

---

## 安裝與執行

### 安裝依賴

```bash
cd /path/to/Kronos_gui
pip install -r requirements.txt
```

### Qt GUI 執行

```bash
cd /path/to/Kronos_gui/qtgui
python kronos_gui.py
```

### Web UI 執行

```bash
cd /path/to/Kronos_gui/webui
python run.py
```

啟動後訪問 http://localhost:7070

---

## 支援的模型

| 模型 | Backend | 參數量 | Context Length |
|------|---------|--------|----------------|
| Kronos-mini | PyTorch | 4.1M | 2048 |
| Kronos-small | PyTorch | 24.7M | 512 |
| Kronos-base (MLX) | MLX (Apple Silicon) | 102.3M | 512 |

---

## 支援的運算設備

自動偵測順序：**MPS → XPU → CUDA → CPU**

| 設備 | 說明 |
|------|------|
| MPS | Apple Silicon GPU |
| XPU | Intel GPU (需安裝 IPEX) |
| CUDA | NVIDIA GPU |
| CPU | 備援方案 |

---

## 預測參數說明

| 參數 | 範圍 | 說明 |
|------|------|------|
| Limit | 120 - 2048 | 歷史 K 線下載數量 |
| Lookback | 50 - 512 | 用於預測的歷史資料筆數 |
| Forecast | 1 - 200 | 未來預測點數 |
| Temperature | 0.1 - 2.0 | 控制預測隨機性，越高越隨機 |
| Top P | 0.1 - 1.0 | Nucleus Sampling，控制在多少高機率範圍內取樣 |
| Samples | 1 - 100 | 產生多條預測路徑並取平均 |

### 建議起始參數

| 參數 | 數值 |
|------|------|
| Symbol | BTC/USDT |
| Timeframe | 4h |
| Lookback | 280 |
| Forecast | 18 |
| Temperature | 0.5 |
| Top P | 0.8 |
| Sample Count | 3 |

---

## Auto Forecast 功能

當按下 **Run Auto Forecast** 時：

1. 所有控制項（Symbol、Timeframe、參數等）會變成**反白不可修改**
2. 系統會根據當前 Timeframe，在每根 K 棒**收盤時**自動執行預測
3. 預測結果會自動儲存到 `prediction_results/` 目錄

### 儲存結構

```
prediction_results/
└── BTC_USDT/
    └── 4h/
        ├── log_20260414_143022.txt      # 預測日誌
        ├── forecast_20260414_143022.png  # 預測圖表
        └── validation_20260414_143022.png # 驗證圖表
```

### Log 範例內容

```
=== Kronos Futures Forecast ===

Model: Kronos-small (PyTorch, local)
Symbol: BTC/USDT
Timeframe: 4h
Latest candle: 2026-04-14 12:00:00
Context candles: 280
Forecast candles: 18
Sampling: T=0.5, top_p=0.8, sample_count=3

Forward forecast:
  Future window: 2026-04-14 16:00:00 -> 2026-04-16 04:00:00
  Last close: $73,450
  Next close forecast: $73,820
  Final close forecast: $74,150
  Forecast move: +0.95%

Recent validation on latest closed candles:
  Window: 2026-04-10 20:00:00 -> 2026-04-13 16:00:00
  MAE: $125
  RMSE: $168
  MAPE: 1.82%
```

---

## 資料夾結構

```
Kronos_gui/
├── model/                       # 模型權重與 Tokenizer
│   ├── kronos-mini/            # Mini 模型
│   ├── kronos-small/           # Small 模型
│   └── kronos-mlx-base/       # MLX 模型 (Apple Silicon)
├── qtgui/                      # Qt 圖形化介面
│   └── kronos_gui.py          # 主程式
├── webui/                      # Web 介面
├── tests/                      # 單元測試
├── examples/                   # 範例腳本
├── finetune/                   # 微調相關
├── prediction_results/          # Auto Forecast 輸出目錄
├── requirements.txt            # 依賴列表
└── README.md                   # 本說明文件
```

---

## 常見問題

**Q: 為什麼 Validation 圖沒有顯示綠色 Actual close 線？**  
A: 驗證需要足夠的歷史資料（Lookback + Forecast 長度），請確認 Limit 設定足夠大。

**Q: 支援 Intel GPU 嗎？**  
A: 是的，請安裝 [Intel Extension for PyTorch (IPEX)](https://intel.github.io/intel-extension-for-pytorch/)。

**Q: 如何提高預測品質？**  
A: 可以嘗試：
1. 增加 Sample Count 取平均
2. 調整 Temperature（較高 = 較多隨機性）
3. 使用更大的模型

**Q: Auto Forecast 按下 Stop 後為什麼還在更新？**  
A: 這是 Qt dispatch 事件的問題，已在單元測試中驗證 cancellation 邏輯。

---

## 單元測試

```bash
cd /path/to/Kronos_gui
python -m unittest tests.test_kronos_gui -v
```

目前共有 78 個測試項目，涵蓋：
- Auto Forecast 啟動/停止邏輯
- Timer 取消機制
- Worker Cancellation 多檢查點
- K棒時間計算
- 價格格式化
- 預測指標計算
- Validation Metrics
- 資料切片
- 檔案命名與路徑

---

## 技術架構

| 層面 | 技術 |
|------|------|
| GUI Framework | PyQt5 / PyQt6 |
| 資料處理 | Pandas, NumPy |
| 深度學習 | PyTorch, Transformers |
| 資料取得 | ccxt (Binance Futures) |
| 圖表 | Matplotlib |
| Web Backend | Flask |
| Web Frontend | HTML, CSS, JavaScript, Plotly.js |
