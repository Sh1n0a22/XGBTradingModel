# Crypto Trading Bot — Multi-Symbol Futures ML Model

Інтелектуальна система для автоматизованої торгівлі криптоактивами на ринку perpetual futures. Використовує XGBoost для ідентифікації торгових паттернів на кількох парах одночасно, підтримує лонги та шорти.

---

## Особливості

- **Multi-Symbol Training** — модель навчається на спільних паттернах кількох символів (BTC, ETH, SOL тощо), що покращує узагальнення ринкової поведінки
- **Walk-Forward Validation** — тестування на 5 послідовних часових вікнах замість класичного train/test split для коректної оцінки стабільності
- **Futures-Ready** — враховує специфіку ф'ючерсів: taker-комісія 0.05%, funding rates кожні 8 годин, підтримка шортів
- **Modular Pipeline** — чіткий поділ на збір даних → feature engineering → навчання → backtest → inference

---

## Встановлення

```bash
git clone https://github.com/your-repo/ml-v2.git
cd ml-v2

python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## Pipeline

### 1. Збір даних — `1_backfill.py`

Завантажує історичні 1m свічки з Binance для вказаних символів.

Налаштуйте список символів і тип ринку у файлі перед запуском:

```python
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
MARKET = "futures"   # або "spot"
DAYS_BACK = 365
```

```bash
python 1_backfill.py
```

Результат: `data/{symbol}_1m.csv` — по одному файлу на кожен символ.

---

### 2. Feature Engineering — `2_features.py`

Автоматично знаходить усі `*_1m.csv` в `data/`, агрегує 1m → 5m → 1h, розраховує індикатори та будує цільову змінну.

```bash
python 2_features.py
```

Результат: `data/dataset.csv` з колонкою `symbol` для розрізнення пар.

Розраховані індикатори на трьох таймфреймах (1m / 5m / 1h):

| Група | Індикатори |
|---|---|
| Trend | EMA 12, EMA 26, MACD (12/26/9) |
| Momentum | RSI 14 |
| Volatility | Bollinger Bands 20, ATR 14 |
| Volume | OBV, VWAP, Relative Volume |

Зверни увагу на розподіл класів у виводі. Якщо `hold` > 85% — зменш `TARGET_THRESHOLD_PCT` у файлі.

---

### 3. Навчання — `3_train.py`

Тренує XGBoost з walk-forward валідацією на 5 послідовних вікнах.

```bash
python 3_train.py
```

Результат:
- `models/xgb_model.json` — фінальна модель для inference
- `models/feature_order.json` — порядок фічей (критично для узгодження з inference-сервісом)

На що звертати увагу у виводі:

```
Accuracy:    mean=0.923, std=0.008   # низький std = стабільна модель
Buy recall:  mean=0.091, std=0.012
Sell recall: mean=0.043, std=0.009
```

Низький `recall` для buy/sell — норма для цієї моделі: вона консервативна і рідко сигналить, але з відносно високою точністю (precision).

---

### 4. Backtest — `4_backtest.py`

Симулює торгівлю на test-периоді з урахуванням реальних умов ф'ючерсного ринку.

```bash
python 4_backtest.py
```

Враховується:
- Taker-комісія: **0.05%** за угоду (реальна ставка Binance Futures)
- Funding rate: **0.01%** кожні 8 годин для відкритих позицій
- Long і Short угоди окремо

Звіт виводиться по кожному символу:

```
--- BTCUSDT ---
Угод: 41
Стратегія: +6.81%  |  Buy & Hold: -20.01%  |  Різниця: +26.81%
Long: 21, Short: 20 | Win rate: 68.3% | Середній P&L: +0.392%
```

Поріг впевненості (`CONFIDENCE_THRESHOLD`) регулює кількість угод. Рекомендовані значення: `0.35–0.45`.

---

### 5. Inference Service — `5_serve.py`

FastAPI-сервіс для отримання прогнозів у реальному часі.

```bash
python 5_serve.py
```

Сервіс піднімається на `http://localhost:8000`.

**Перевірка:**
```bash
curl http://localhost:8000/health
# {"status": "ok", "model_loaded": true}
```

**Запит:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "ETHUSDT",
    "features": {
      "m1_ema_fast": 3201.5,
      "m1_ema_slow": 3198.2,
      "m1_rsi": 48.2,
      "m1_macd": 1.3,
      ...
    }
  }'
```

**Відповідь:**
```json
{
  "symbol": "ETHUSDT",
  "action": "buy",
  "confidence": 0.62,
  "probabilities": {
    "hold": 0.21,
    "buy": 0.62,
    "sell": 0.17
  }
}
```

Поле `confidence` — ймовірність передбаченого класу. Значення нижче `CONFIDENCE_THRESHOLD` слід ігнорувати (повертати `hold`).

---

### 6. Діагностика — `diagnose.py`

Перевіряє, чи модель взагалі здатна видавати buy/sell сигнали на історичних даних. Корисно після перетренування або при підозрі на проблеми з якістю сигналу.

```bash
python diagnose.py
```

---

## Налаштування стратегії

### Ключові параметри

| Параметр | Файл | За замовчуванням | Опис |
|---|---|---|---|
| `TARGET_THRESHOLD_PCT` | `2_features.py` | `0.0015` | Мінімальний рух ціни для класифікації як buy/sell |
| `CONFIDENCE_THRESHOLD` | `4_backtest.py` | `0.40` | Мінімальна впевненість для відкриття угоди |
| `TAKER_FEE_PCT` | `4_backtest.py` | `0.0005` | Комісія за угоду |
| `FUNDING_RATE_PER_8H` | `4_backtest.py` | `0.0001` | Funding rate за 8-годинний період |

### Додавання нового символу

1. Додай символ у `1_backfill.py`:
```python
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "LINKUSDT"]
```
2. Перезапусти весь pipeline (кроки 1–4).
3. Перевір метрики walk-forward окремо для нового символу перед використанням у продакшні.

### Symbol-специфічне навчання

Якщо різні монети потребують індивідуального підходу, можна додати символ як one-hot ознаку в `2_features.py`:

```python
dataset = pd.get_dummies(dataset, columns=["symbol"], prefix="sym")
```

Це додасть колонки `sym_BTCUSDT`, `sym_ETHUSDT` тощо. При такій зміні необхідно оновити `FeatureBuilderService` на стороні NestJS-бекенду.

---

## Структура проєкту

```
ml-v2/
├── 1_backfill.py          # збір даних з Binance
├── 2_features.py          # feature engineering
├── 3_train.py             # навчання моделі
├── 4_backtest.py          # backtesting стратегії
├── 5_serve.py             # inference API
├── diagnose.py            # діагностика моделі
├── requirements.txt       # залежності
├── models/                # згенерується після 3_train.py (не в git)
│   ├── xgb_model.json
│   └── feature_order.json
└── data/                  # згенерується після 1_backfill.py (не в git)
    ├── btcusdt_1m.csv
    └── dataset.csv
```

---

## Важливі застереження

**Статистична значущість** — для довіри до результатів backtest потрібно щонайменше 30–50 угод на тестовому вікні. Менша кількість може бути випадковістю.

**Overfitting** — порівнюй результати walk-forward валідації з backtest. Великий розрив між тренувальною і тестовою метриками сигналізує про переnavчання.

**Ринковий режим** — модель тренується на конкретному історичному відрізку. Зміна ринкового режиму (наприклад, з трендового на бічний) може погіршити якість сигналів.

**Не фінансова порада** — цей проєкт є дослідницьким інструментом. Торгівля ф'ючерсами пов'язана з суттєвим ризиком втрати капіталу.