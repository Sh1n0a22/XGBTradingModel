import os
import pandas as pd
import numpy as np
from xgboost import XGBClassifier

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DATASET_FILE = os.path.join(DATA_DIR, "dataset.csv")

TARGET_COL = "target"
EXCLUDE_FROM_FEATURES = ["close", "symbol", TARGET_COL]

TEST_SIZE_RATIO = 0.2  # останні 20% часу -- чесний test, модель тренується тільки на тому, що ДО нього

# Futures taker fee ~0.05% за угоду (нижче за spot ~0.1%, перевірено на актуальних даних Binance 2026)
TAKER_FEE_PCT = 0.0005
CONFIDENCE_THRESHOLD = 0.4  # підібрано емпірично -- даватиме баланс кількості угод і якості сигналу

# Funding rate на perpetual futures -- періодична виплата між лонгами і
# шортами, зазвичай кожні 8 годин, типово ~0.01% за період в обидва боки.
# Це НЕ комісія біржі, а P&L ефект самого продукту. Враховуємо як невелику
# постійну "вартість тримання" позиції в часі.
FUNDING_RATE_PER_8H = 0.0001
CANDLES_PER_FUNDING_PERIOD = 8 * 60  # 8 годин в 1m свічках


def load_test_split():
    df = pd.read_csv(DATASET_FILE, index_col=0, parse_dates=True)
    df = df.sort_index()

    feature_columns = [c for c in df.columns if c not in EXCLUDE_FROM_FEATURES]
    split_idx = int(len(df) * (1 - TEST_SIZE_RATIO))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    return train_df, test_df, feature_columns


def train_test_only_model(train_df: pd.DataFrame, feature_columns: list) -> XGBClassifier:
    """Окрема модель ЛИШЕ для чесного backtest -- не бачила test-період взагалі."""
    X_train = train_df[feature_columns]
    y_train = train_df[TARGET_COL].astype(int)

    model = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        objective="multi:softprob", num_class=3, eval_metric="mlogloss", random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def simulate(test_df: pd.DataFrame, predictions: np.ndarray, probabilities: np.ndarray) -> dict:
    """
    position: 0 = немає позиції, 1 = long, -1 = short.
    Funding застосовується щоразу, як минає CANDLES_PER_FUNDING_PERIOD свічок
    при відкритій позиції (спрощено, в реальності funding -- це точний час доби).
    """
    closes = test_df["close"].values
    position = 0
    entry_price = None
    entry_idx = None
    capital = 1.0
    trades = []

    def close_position(exit_idx: int, exit_price: float):
        nonlocal capital
        if position == 1:
            raw_return = (exit_price - entry_price) / entry_price
        else:  # short
            raw_return = (entry_price - exit_price) / entry_price

        periods_held = (exit_idx - entry_idx) // CANDLES_PER_FUNDING_PERIOD
        funding_cost = periods_held * FUNDING_RATE_PER_8H

        capital *= 1 + raw_return - funding_cost
        capital *= 1 - TAKER_FEE_PCT
        trades.append({
            "direction": "long" if position == 1 else "short",
            "raw_return_pct": raw_return * 100,
            "funding_cost_pct": funding_cost * 100,
        })

    for i in range(len(test_df)):
        action = predictions[i]
        confidence = probabilities[i][action]
        price = closes[i]

        if confidence < CONFIDENCE_THRESHOLD:
            continue

        if action == 1:  # buy / long сигнал
            if position == -1:
                close_position(i, price)
                position = 0
            if position == 0:
                capital *= 1 - TAKER_FEE_PCT
                position = 1
                entry_price = price
                entry_idx = i

        elif action == 2:  # sell / short сигнал
            if position == 1:
                close_position(i, price)
                position = 0
            if position == 0:
                capital *= 1 - TAKER_FEE_PCT
                position = -1
                entry_price = price
                entry_idx = i

    if position != 0:
        close_position(len(test_df) - 1, closes[-1])

    return {
        "final_capital": capital,
        "total_return_pct": (capital - 1.0) * 100,
        "num_trades": len(trades),
        "trades": trades,
    }


def buy_and_hold_baseline(test_df: pd.DataFrame) -> float:
    start_price = test_df["close"].iloc[0]
    end_price = test_df["close"].iloc[-1]
    return (end_price - start_price) / start_price * 100


def run_backtest_for_symbol(symbol_label: str, test_df: pd.DataFrame, model, feature_columns: list):
    X_test = test_df[feature_columns]
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    result = simulate(test_df, predictions, probabilities)
    baseline_pct = buy_and_hold_baseline(test_df)

    print(f"\n--- {symbol_label} ---")
    print(f"Період: {test_df.index.min()} -> {test_df.index.max()} ({len(test_df)} рядків)")
    print(f"Угод: {result['num_trades']}")
    print(f"Стратегія: {result['total_return_pct']:+.2f}%  |  Buy & Hold: {baseline_pct:+.2f}%  |  "
          f"Різниця: {result['total_return_pct'] - baseline_pct:+.2f}%")

    if result["num_trades"] > 0:
        returns = [t["raw_return_pct"] for t in result["trades"]]
        longs = sum(1 for t in result["trades"] if t["direction"] == "long")
        shorts = sum(1 for t in result["trades"] if t["direction"] == "short")
        win_rate = sum(1 for r in returns if r > 0) / len(returns) * 100
        print(f"Long: {longs}, Short: {shorts} | Win rate: {win_rate:.1f}% | "
              f"Середній P&L: {np.mean(returns):+.3f}%")


def main():
    train_df, test_df, feature_columns = load_test_split()

    print(f"Train: {len(train_df)} рядків (до {train_df.index.max()})")
    print(f"Test:  {len(test_df)} рядків (з {test_df.index.min()})")
    print(f"Комісія: {TAKER_FEE_PCT * 100}% за угоду | Confidence threshold: {CONFIDENCE_THRESHOLD}")

    print("\nТренування backtest-моделі (НЕ та сама, що йде у продакшн -- ця не бачила test-період)...")
    model = train_test_only_model(train_df, feature_columns)

    if "symbol" in test_df.columns:
        for symbol in sorted(test_df["symbol"].unique()):
            symbol_test_df = test_df[test_df["symbol"] == symbol]
            run_backtest_for_symbol(symbol, symbol_test_df, model, feature_columns)
    else:
        run_backtest_for_symbol("ALL", test_df, model, feature_columns)


if __name__ == "__main__":
    main()
