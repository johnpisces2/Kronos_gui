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
- **Local Paper Mode**：在 Qt GUI 的 `Paper Mode` 分頁中模擬本地持倉、停損、平倉與交易紀錄

### 圖表顯示
- **Market 頁**：Forecast 與 Validation 同頁顯示
  - 上半部：歷史價格 + 預測結果
  - 灰色線：Historical close
  - 黃色區間：MC interval (Monte Carlo Confidence Band)
  - 橙色線：Forecast close
  - 主圖同步疊加 local paper trades 進出場標記
  - 藍色十字線：最新真實價格定位
  - 下半部：回測驗證
  - 灰色線：Validation history
  - 紫色線：Predicted close
  - 綠色線：Actual close
  - 藍色十字線：最新真實價格定位
- **Execution 頁**：本地紙上交易與後續實單擴充區
  - `Execution Mode` 面板已預留 `Local Paper / Binance Testnet / Live Trading`
  - 目前**只有 Local Paper 會生效**
  - `Enable Paper Trading` 開關
  - `Initial Equity` 初始資金控制
  - `Reset Paper Account` 帳本重設
  - `Simulated Order Panel` 可設定下單數量、槓桿，並即時估算名目價值與保證金
  - 訊號判斷
  - 本地持倉表格
  - 停損價格
  - 已實現 / 未實現 PnL
  - 每筆交易明細表
  - 累積績效曲線
  - 進出場標記圖（將本地交易標回價格走勢）
  - 最近交易紀錄
- **Log 頁**：執行日誌

### 驗證指標
| 指標 | 全名 | 說明 |
|------|------|------|
| MAE | Mean Absolute Error | 平均絕對誤差 |
| RMSE | Root Mean Square Error | 均方根誤差 |
| MAPE | Mean Absolute Percentage Error | 平均絕對百分比誤差 |

### Local Paper Mode 規劃
- 採用**本地 paper mode**，只在本機模擬持倉、停損與平倉
- **不送出 Binance 真實委託**
- 下單面板中的「數量 / 槓桿 / 名目價值」已直接影響本地持倉、交易明細與績效曲線
- 訊號評估頻率與目前 Forecast 相同：**每根 timeframe K 棒收盤後更新一次**
- 後續若要接交易所 API，可沿用相同訊號規則，再把本地模擬倉位替換成交易所倉位同步

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

## Local Paper Mode 策略

本策略先以 **local paper mode** 為主，所有下單、停損、平倉只記錄在本機，不對 Binance 發送真實委託。

### 一、訊號更新時機

- 只在目前選定 timeframe 的 **新 K 棒收盤後** 重新計算一次訊號
- 不在 K 棒尚未收盤時反覆變更方向
- 因此 paper mode 的訊號節奏會與 Forecast 頁面的更新節奏一致

### 二、核心欄位定義

假設：

- `current_price`：目前最新已收盤 K 棒的 close
- `forecast_price`：本次 forecast 視窗最後一根預測 close
- `validation_start_price`：validation history 最後一根 close
- `validation_pred_price`：validation predicted close 最後一根
- `validation_actual_price`：validation actual close 最後一根

則策略使用以下欄位：

```text
forecast_return = forecast_price / current_price - 1
validation_pred_return = validation_pred_price / validation_start_price - 1
validation_actual_return = validation_actual_price / validation_start_price - 1
validation_history_delta = validation_history[-1] - validation_history[-2]
validation_actual_delta = validation_actual[-1] - validation_actual[-2]
```

### 三、作多條件

當以下條件全部成立時，產生 `enter_long`：

```text
forecast_return >= 0.02
validation_pred_return > 0
validation_actual_return > 0
validation_history_delta > 0
validation_actual_delta > 0
```

意思是：

- forecast 最終預估至少比當前價格高 2%
- validation 的模型預測方向仍然向上
- validation 的實際價格方向也向上
- validation history 與 validation actual 的最新動能都沒有轉弱

### 四、作空條件

當以下條件全部成立時，產生 `enter_short`：

```text
forecast_return <= -0.02
validation_pred_return < 0
validation_actual_return < 0
validation_history_delta < 0
validation_actual_delta < 0
```

### 五、停損距離

進場時的預測價差定義為：

```text
predicted_edge = abs(forecast_return)
```

停損距離使用「預測價差的一半」，但加上上下限，避免太窄或太寬：

```text
stop_distance_pct = clamp(predicted_edge * 0.5, min_stop=0.01, max_stop=0.04)
```

預設解讀：

- 最小停損距離：1%
- 最大停損距離：4%

停損價：

```text
long_stop_price = entry_price * (1 - stop_distance_pct)
short_stop_price = entry_price * (1 + stop_distance_pct)
```

例如：

- `forecast_return = +4%`
- 停損距離 = `4% * 0.5 = 2%`
- 若作多進場價為 `100`，停損價為 `98`

### 六、平倉條件

#### 多單平倉

當以下任一條件成立時，產生 `exit_long`：

```text
current_price <= stop_price
forecast_return < 0.02
validation_pred_return <= 0
validation_actual_return <= 0
validation_history_delta <= 0
validation_actual_delta <= 0
```

#### 空單平倉

當以下任一條件成立時，產生 `exit_short`：

```text
current_price >= stop_price
forecast_return > -0.02
validation_pred_return >= 0
validation_actual_return >= 0
validation_history_delta >= 0
validation_actual_delta >= 0
```

### 七、目前策略模組位置

目前已整理成可直接接 GUI 新分頁的本地策略模組：

- `qtgui/paper_strategy.py`

目前提供的資料結構與方法包含：

- `PaperStrategyConfig`
- `PaperSignalSnapshot`
- `PaperPosition`
- `PaperDecision`
- `build_signal_snapshot(payload)`
- `build_entry_decision(snapshot, config)`
- `build_exit_decision(position, snapshot, config)`

這樣 GUI 新分頁只要在每次 forecast 完成後，將 payload 餵進 `build_signal_snapshot()`，再根據目前是否持倉決定呼叫 entry 或 exit 判斷即可。

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
python3 -m pytest tests/test_kronos_gui.py tests/test_paper_strategy.py -q
```

目前測試重點涵蓋：
- GUI helper 與時間轉換
- Auto Forecast 啟動/停止與 stale callback 防護
- Qt GUI integration
- Validation 視窗切片
- Local paper mode 進出場與停損規則
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
