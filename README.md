# Kronos BTC Forecast Terminal

Kronos BTC Forecast Terminal 是一個**加密貨幣期貨價格預測系統**，從 Binance Futures 取得 OHLCV 資料，利用 Kronos Foundation Model 進行價格預測，並透過 Backtesting 驗證模型表現。

---

## 📁 程式碼架構

```
Kronos_gui/
├── main.py                 # 應用程式入口點
├── kronos_gui.py           # Qt GUI 主程式 (KronosGUI 類別)
├── execution.py            # 紙帶交易執行引擎 (ExecutionEngine 類別)
├── paper_strategy.py       # 交易策略演算法 (決策邏輯)
├── model/                  # 模型權重與Tokenizer
│   ├── kronos-mini/
│   ├── kronos-small/
│   ├── kronos-mlx-base/
│   └── ...
├── tests/                 # 單元測試
└── README.md
```

### 核心類別說明

| 檔案 | 類別/模組 | 職責 |
|------|-----------|------|
| `main.py` | - | 應用程式 entry point，建立 QApplication 並啟動 GUI |
| `kronos_gui.py` | `KronosGUI` | GUI 主視窗，持有 `ExecutionEngine` 實例，委託處理交易邏輯 |
| `execution.py` | `ExecutionEngine` | 封裝所有紙帶交易狀態和方法（持倉、 equity、交易歷史） |
| `paper_strategy.py` | 策略函式群 | 進場/出場決策演算法、停損/止盈計算 |

### 設計模式

**GUI 與交易邏輯分離**：
- `KronosGUI` 持有 `self.execution = ExecutionEngine()`
- GUI 委託 `execution.update(payload)` 處理交易邏輯
- `execution.py` 專注於狀態管理，不涉及 UI

---

## 📊 策略演算法詳解

### 資料結構

```python
# paper_strategy.py

@dataclass(frozen=True)
class PaperSignalSnapshot:
    signal_time: pd.Timestamp      # 訊號時間
    current_price: float          # 目前價格（最新已收盤K棒的close）
    forecast_price: float         # 本次預測最終價格
    forecast_return: float        # 預測報酬率 = forecast_price / current_price - 1
    validation_start_price: float # Validation history 最後一根收盤價
    validation_pred_price: float   # Validation 預測最後一根價格
    validation_pred_return: float  # Validation 預測報酬率
    validation_history_delta: float # Validation history 最近兩根差值
    validation_pred_delta: float   # Validation 預測最近兩根差值

@dataclass(frozen=True)
class PaperPosition:
    side: PositionSide            # "long" 或 "short"
    entry_price: float            # 進場價格
    stop_price: float             # 停損價格
    take_profit_price: float      # 止盈價格
    entry_time: pd.Timestamp      # 進場時間
    quantity: float = 1.0         # 數量
    leverage: float = 1.0         # 槓桿倍數

@dataclass(frozen=True)
class PaperDecision:
    action: DecisionAction        # "enter_long", "enter_short", "exit_long", "exit_short", "hold", "no_action"
    reason: str                  # 決策原因
    stop_distance_pct: Optional[float]  # 停損距離百分比
    stop_price: Optional[float]   # 停損價格
    tp_distance_pct: Optional[float]    # 止盈距離百分比
    take_profit_price: Optional[float]  # 止盈價格
```

### 策略參數設定

```python
@dataclass(frozen=True)
class PaperStrategyConfig:
    entry_threshold_pct: float = 0.02      # 進場門檻：預測報酬率需 >= 2%
    stop_loss_fraction: float = 0.5        # 停損距離 = 預測價差 × 0.5
    min_stop_loss_pct: float = 0.01        # 最小停損：1%
    max_stop_loss_pct: float = 0.04        # 最大停損：4%
    take_profit_multiplier: float = 2.0    # 止盈距離 = 停損距離 × 2
```

### 進場條件邏輯

#### 作多條件 (`enter_long`)

必須**全部滿足**：

```
forecast_return >= 2%          # 預測最終價格比現在高至少2%
validation_pred_return > 0     # Validation 模型的預測方向向上
validation_pred_delta > 0       # Validation 預測的短期動能向上
validation_history_delta > 0    # Validation 歷史價格的動能向上
```

#### 作空條件 (`enter_short`)

必須**全部滿足**：

```
forecast_return <= -2%          # 預測最終價格比現在低至少2%
validation_pred_return < 0     # Validation 模型的預測方向向下
validation_pred_delta < 0       # Validation 預測的短期動能向下
validation_history_delta < 0    # Validation 歷史價格的動能向下
```

### 停損/止盈計算

```python
# 預測價差
predicted_edge = abs(forecast_return)

# 停損距離（限制在 1% ~ 4% 之間）
stop_distance_pct = clamp(predicted_edge * 0.5, min=1%, max=4%)

# 多單停損/止盈價
long_stop_price = current_price * (1 - stop_distance_pct)
long_tp_price = current_price * (1 + stop_distance_pct * 2)

# 空單停損/止盈價
short_stop_price = current_price * (1 + stop_distance_pct)
short_tp_price = current_price * (1 - stop_distance_pct * 2)
```

### 出場條件邏輯

#### 多單平倉 (`exit_long`)

當以下**任一條件**成立時觸發：

| 條件 | 原因 |
|------|------|
| `current_price <= stop_price` | 停損觸發 |
| `current_price >= take_profit_price` | 止盈觸發 |
| `validation_pred_return <= 0` | Validation 預測方向轉空 |
| `validation_pred_delta <= 0` | Validation 短期動能轉空 |

#### 空單平倉 (`exit_short`)

當以下**任一條件**成立時觸發：

| 條件 | 原因 |
|------|------|
| `current_price >= stop_price` | 停損觸發 |
| `current_price <= take_profit_price` | 止盈觸發 |
| `validation_pred_return >= 0` | Validation 預測方向轉多 |
| `validation_pred_delta >= 0` | Validation 短期動能轉多 |

---

## ⚙️ ExecutionEngine 實作細節

位置：`execution.py`

### 状态属性

| 屬性 | 類型 | 說明 |
|------|------|------|
| `enabled` | bool | 是否啟用紙帶交易 |
| `mode` | str | 執行模式："paper" / "testnet" / "live" |
| `position` | Optional[PaperPosition] | 目前持倉，無持倉時為 None |
| `trade_history` | List[Dict] | 已平倉交易記錄（最多保留40筆） |
| `equity_history` | List[Dict] | Equity 歷史點（最多保留240筆） |
| `initial_equity` | float | 初始資金 |
| `realized_equity` | float | 已實現 Equity（含已平倉損益） |
| `last_snapshot` | Optional[PaperSignalSnapshot] | 最新的訊號快照 |
| `last_decision` | Optional[PaperDecision] | 最新的進場/出場決策 |

### 核心方法

#### `update(payload: dict) -> tuple[PaperSignalSnapshot, Optional[PaperDecision]]`

處理一個預測 payload，更新持倉狀態，返回 (snapshot, decision)。

流程：
1. 解析 payload 建立 `PaperSignalSnapshot`
2. 若無持倉 → 呼叫 `build_entry_decision()` 判斷是否進場
3. 若有持倉 → 呼叫 `build_exit_decision()` 判斷是否出場
4. 進場/出場時記錄交易、更新 equity

#### `current_order_quantity_by_risk(entry_price, stop_distance_pct) -> float`

根據風險比例計算訂單數量：

```
risk_amount = equity * risk_fraction (預設 8%)
stop_distance_price = entry_price * stop_distance_pct
quantity = risk_amount / stop_distance_price
```

#### `compute_position_return_pct(position, current_price) -> float`

計算持倉回報率：
- Long：`current_price / entry_price - 1`
- Short：`entry_price / current_price - 1`

#### `compute_position_pnl_amount(position, current_price) -> float`

計算持倉未實現損益：
- Long：`(current_price - entry_price) * quantity`
- Short：`(entry_price - current_price) * quantity`

---

## 🔄 交易流程圖

```
每一次 Forecast 完成
        │
        ▼
┌───────────────────┐
│  execution.update │
│   (payload)      │
└────────┬──────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  build_signal_snapshot(payload)     │
│  建立 PaperSignalSnapshot           │
└────────┬────────────────────────────┘
         │
         ▼
    ┌─────────┐
    │ 有持倉？ │
    └────┬────┘
       │        │
      Yes      No
       │        │
       ▼        ▼
┌─────────────┐  ┌────────────────────────┐
│ build_exit  │  │ build_entry_decision   │
│ _decision() │  │ 檢查進場條件           │
└──────┬──────┘  └───────────┬────────────┘
       │                      │
       ▼                      ▼
  ┌─────────┐         ┌──────────────┐
  │ 平倉？  │         │ 滿足進場條件？│
  └────┬────┘         └──────┬───────┘
       │                     │
      Yes                   No
       │                     │
       ▼                     ▼
  ┌────────────┐      ┌──────────┐
  │ record_trade│      │ no_action│
  │ 更新 equity│      │ 不進場   │
  │ 清空倉位  │      └──────────┘
  └────────────┘
```

---

## 🧪 單元測試

```bash
cd /path/to/Kronos_gui
python -m unittest tests.test_kronos_gui tests.test_paper_trading_regression tests.test_paper_strategy
```

### 測試覆蓋項目

| 測試檔案 | 覆蓋範圍 |
|----------|----------|
| `test_kronos_gui.py` | GUI 元件、Auto Forecast、Helper 函式 |
| `test_paper_trading_regression.py` | ExecutionEngine 所有方法、策略邏輯 |
| `test_paper_strategy.py` | 進場/出场決策函式、停損/止盈計算 |

---

## 📦 依賴

| 套件 | 用途 |
|------|------|
| PyQt5 / PyQt6 | GUI 框架 |
| pandas, numpy | 資料處理 |
| torch | 模型推理 (PyTorch backend) |
| mlx | Apple Silicon 模型推理 |
| ccxt | Binance 期貨資料取得 |
| matplotlib | 圖表繪製 |

---

## ⚡ 自動設備偵測

執行時自動偵測可用設備（優先順序）：

```
MPS (Apple Silicon) > XPU (Intel GPU) > CUDA (NVIDIA) > CPU
```