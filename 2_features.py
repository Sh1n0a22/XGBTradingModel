import os
import glob
import re
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_FILE = os.path.join(DATA_DIR, "dataset.csv")

TARGET_HORIZON_MIN = 10

# ФʼЮЧЕРСИ мають значно нижчу комісію за спот (taker ~0.05% проти ~0.1%).
# Порог тут покриває komісію відкриття+закриття (0.05% x 2 = 0.1%) + запас
# на spread/slippage. Якщо більшість угод будуть maker (лімітні ордери,
# ~0.02%) -- можна знизити цей порог ще більше пізніше.
TARGET_THRESHOLD_PCT = 0.0015  # 0.15%

RELATIVE_VOLUME_LOOKBACK = 20


def discover_symbols() -> list:
    """Знаходить усі {symbol}_1m.csv файли в data/ автоматично."""
    pattern = os.path.join(DATA_DIR, "*_1m.csv")
    files = glob.glob(pattern)
    symbols = []
    for f in files:
        match = re.match(r"(.+)_1m\.csv$", os.path.basename(f))
        if match:
            symbols.append(match.group(1).upper())
    return sorted(symbols)


def load_raw_candles(symbol: str) -> pd.DataFrame:
    file_path = os.path.join(DATA_DIR, f"{symbol.lower()}_1m.csv")
    df = pd.read_csv(file_path)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("open_time").sort_index()
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def aggregate(df_1m: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = df_1m.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return agg.dropna()


def calculate_indicators(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    ema_fast = EMAIndicator(close=df["close"], window=12).ema_indicator()
    ema_slow = EMAIndicator(close=df["close"], window=26).ema_indicator()
    macd_calc = MACD(close=df["close"], window_fast=12, window_slow=26, window_sign=9)
    rsi = RSIIndicator(close=df["close"], window=14).rsi()
    bb = BollingerBands(close=df["close"], window=20, window_dev=2)
    atr = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14).average_true_range()
    obv = OnBalanceVolumeIndicator(close=df["close"], volume=df["volume"]).on_balance_volume()
    vwap = VolumeWeightedAveragePrice(
        high=df["high"], low=df["low"], close=df["close"], volume=df["volume"], window=14
    ).volume_weighted_average_price()

    rolling_avg_volume = df["volume"].rolling(RELATIVE_VOLUME_LOOKBACK).mean()
    relative_volume = df["volume"] / rolling_avg_volume.replace(0, np.nan)

    out[f"{prefix}_ema_fast"] = ema_fast
    out[f"{prefix}_ema_slow"] = ema_slow
    out[f"{prefix}_macd"] = macd_calc.macd()
    out[f"{prefix}_macd_signal"] = macd_calc.macd_signal()
    out[f"{prefix}_rsi"] = rsi
    out[f"{prefix}_bb_upper"] = bb.bollinger_hband()
    out[f"{prefix}_bb_lower"] = bb.bollinger_lband()
    out[f"{prefix}_atr"] = atr
    out[f"{prefix}_obv"] = obv
    out[f"{prefix}_vwap"] = vwap
    out[f"{prefix}_relative_volume"] = relative_volume

    return out


def build_target(close: pd.Series) -> pd.Series:
    future_close = close.shift(-TARGET_HORIZON_MIN)
    future_return = (future_close - close) / close

    target = pd.Series(0, index=close.index)
    target[future_return > TARGET_THRESHOLD_PCT] = 1   # buy / long
    target[future_return < -TARGET_THRESHOLD_PCT] = 2  # sell / short

    target.iloc[-TARGET_HORIZON_MIN:] = np.nan
    return target


def build_dataset_for_symbol(symbol: str) -> pd.DataFrame:
    print(f"\n[{symbol}] Завантаження сирих 1m свічок...")
    df_1m = load_raw_candles(symbol)
    print(f"  {len(df_1m)} рядків, з {df_1m.index.min()} по {df_1m.index.max()}")

    df_5m = aggregate(df_1m, "5min")
    df_1h = aggregate(df_1m, "1h")
    print(f"  5m: {len(df_5m)} рядків, 1h: {len(df_1h)} рядків")

    feat_1m = calculate_indicators(df_1m, "m1")
    feat_5m = calculate_indicators(df_5m, "m5")
    feat_1h = calculate_indicators(df_1h, "h1")

    feat_5m_aligned = feat_5m.reindex(feat_1m.index, method="ffill")
    feat_1h_aligned = feat_1h.reindex(feat_1m.index, method="ffill")

    dataset = pd.concat([feat_1m, feat_5m_aligned, feat_1h_aligned], axis=1)
    dataset["close"] = df_1m["close"]
    dataset["symbol"] = symbol
    dataset["target"] = build_target(df_1m["close"])

    before = len(dataset)
    dataset = dataset.dropna()
    print(f"  Видалено {before - len(dataset)} рядків з NaN, лишилось {len(dataset)}")

    return dataset


def main():
    symbols = discover_symbols()
    if not symbols:
        raise RuntimeError(
            f"Не знайдено жодного файлу *_1m.csv у {DATA_DIR}. "
            f"Спочатку запусти 1_backfill.py."
        )
    print(f"Знайдено символів: {symbols}")

    all_datasets = []
    for symbol in symbols:
        dataset = build_dataset_for_symbol(symbol)
        all_datasets.append(dataset)

    combined = pd.concat(all_datasets, axis=0).sort_index()
    combined.to_csv(OUTPUT_FILE)

    print(f"\nГотово. {len(combined)} рядків (всі символи) збережено у {OUTPUT_FILE}")
    print("\nРозподіл класів target (по всіх символах разом):")
    print(combined["target"].value_counts().rename({0: "hold", 1: "buy", 2: "sell"}))

    if len(symbols) > 1:
        print("\nРозподіл по символах:")
        print(combined.groupby("symbol")["target"].value_counts())


if __name__ == "__main__":
    main()
