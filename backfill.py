import time
import csv
import os
from datetime import datetime, timezone, timedelta
import urllib.request
import json

# Додавай сюди нові символи -- решта пайплайну (2_features.py, 3_train.py,
# 4_backtest.py) автоматично підхоплять усі файли в data/, що відповідають
# паттерну {symbol}_1m.csv.
SYMBOLS = ["ETHUSDT","SOLUSDT","BCHUSDT","LTCUSDT","ETCUSDT","LINKUSDT"]

INTERVAL = "1m"
DAYS_BACK = 1085

# MARKET = "futures" -- тягне дані з Binance USDT-M Futures klines endpoint.
# MARKET = "spot"    -- тягне дані зі звичайного spot endpoint.
# Структура відповіді ідентична для обох, відрізняється лише базовий URL.
# Якщо плануєш торгувати фʼючерсами -- важливо тренувати модель саме на
# futures-даних, бо ціна/волатильність на futures і spot можуть незначно
# відрізнятись (basis), і це теоретично впливає на якість сигналу.
MARKET = "futures"

BASE_URLS = {
    "spot": "https://api.binance.com/api/v3/klines",
    "futures": "https://fapi.binance.com/fapi/v1/klines",
}

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
LIMIT = 1000


def fetch_klines(base_url: str, symbol: str, interval: str, start_time_ms: int, end_time_ms: int) -> list:
    params = (
        f"symbol={symbol}&interval={interval}"
        f"&startTime={start_time_ms}&endTime={end_time_ms}&limit={LIMIT}"
    )
    url = f"{base_url}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as response:
        return json.loads(response.read().decode())


def interval_to_ms(interval: str) -> int:
    unit = interval[-1]
    value = int(interval[:-1])
    multipliers = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}
    return value * multipliers[unit]


def backfill_symbol(symbol: str, base_url: str) -> None:
    output_file = os.path.join(OUTPUT_DIR, f"{symbol.lower()}_{INTERVAL}.csv")

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=DAYS_BACK)
    current_ms = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)
    step_ms = interval_to_ms(INTERVAL) * LIMIT

    print(f"\n[{symbol}] Завантаження ({MARKET}) з {start_time.date()} по {end_time.date()}")

    total_rows = 0
    request_count = 0

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["open_time", "open", "high", "low", "close", "volume", "close_time"])

        while current_ms < end_ms:
            batch_end = min(current_ms + step_ms, end_ms)
            try:
                klines = fetch_klines(base_url, symbol, INTERVAL, current_ms, batch_end)
            except Exception as e:
                print(f"  Помилка запиту, повтор через 5с: {e}")
                time.sleep(5)
                continue

            if not klines:
                current_ms = batch_end
                continue

            for k in klines:
                writer.writerow([k[0], k[1], k[2], k[3], k[4], k[5], k[6]])
                total_rows += 1

            current_ms = klines[-1][0] + interval_to_ms(INTERVAL)
            request_count += 1

            if request_count % 20 == 0:
                progress_date = datetime.fromtimestamp(current_ms / 1000, tz=timezone.utc).date()
                print(f"  ...{total_rows} свічок, дійшли до {progress_date}")

            time.sleep(0.25)

    print(f"[{symbol}] Готово. {total_rows} рядків -> {output_file}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base_url = BASE_URLS[MARKET]

    for symbol in SYMBOLS:
        backfill_symbol(symbol, base_url)

    print(f"\nУсі символи завантажено: {SYMBOLS}")


if __name__ == "__main__":
    main()
