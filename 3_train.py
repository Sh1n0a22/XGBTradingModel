import os
import json
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
DATASET_FILE = os.path.join(DATA_DIR, "dataset.csv")

TARGET_COL = "target"
EXCLUDE_FROM_FEATURES = ["close", "symbol", TARGET_COL]

# Walk-forward: ділимо часовий діапазон на N_WINDOWS рівних вікон.
# Вікно 1: train на даних до t1, test на [t1, t2]
# Вікно 2: train на даних до t2 (тобто і test-дані вікна 1 теж йдуть у train), test на [t2, t3]
# ... і так далі. Це симулює реальне використання: модель постійно
# перетреновується на нових даних і тестується на майбутньому, якого не бачила.
N_WALK_FORWARD_WINDOWS = 5
MIN_TRAIN_RATIO = 0.4  # перше тренування -- мінімум на 40% історії, щоб не тренуватись на замалому шматку


def build_model() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
    )


def walk_forward_validate(df: pd.DataFrame, feature_columns: list) -> None:
    """Тренує і тестує модель на кількох послідовних часових вікнах підряд."""
    n = len(df)
    min_train_size = int(n * MIN_TRAIN_RATIO)
    remaining = n - min_train_size
    window_size = remaining // N_WALK_FORWARD_WINDOWS

    print(f"\n{'=' * 60}")
    print(f"WALK-FORWARD ВАЛІДАЦІЯ ({N_WALK_FORWARD_WINDOWS} вікон)")
    print(f"{'=' * 60}")

    all_reports = []

    for window_idx in range(N_WALK_FORWARD_WINDOWS):
        train_end = min_train_size + window_idx * window_size
        test_end = min(train_end + window_size, n)

        if train_end >= test_end:
            continue

        train_df = df.iloc[:train_end]
        test_df = df.iloc[train_end:test_end]

        X_train = train_df[feature_columns]
        y_train = train_df[TARGET_COL].astype(int)
        X_test = test_df[feature_columns]
        y_test = test_df[TARGET_COL].astype(int)

        model = build_model()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        report = classification_report(
            y_test, y_pred, target_names=["hold", "buy", "sell"], output_dict=True, zero_division=0
        )
        all_reports.append(report)

        print(f"\n--- Вікно {window_idx + 1}/{N_WALK_FORWARD_WINDOWS} ---")
        print(f"Train: {train_df.index.min()} -> {train_df.index.max()} ({len(train_df)} рядків)")
        print(f"Test:  {test_df.index.min()} -> {test_df.index.max()} ({len(test_df)} рядків)")
        print(
            f"Accuracy: {report['accuracy']:.3f} | "
            f"Buy recall: {report['buy']['recall']:.3f} | "
            f"Sell recall: {report['sell']['recall']:.3f}"
        )

    if not all_reports:
        print("Недостатньо даних для жодного walk-forward вікна.")
        return

    print(f"\n{'=' * 60}")
    print("ПІДСУМОК ПО ВСІХ ВІКНАХ (стабільність важливіша за пікові значення)")
    print(f"{'=' * 60}")
    accuracies = [r["accuracy"] for r in all_reports]
    buy_recalls = [r["buy"]["recall"] for r in all_reports]
    sell_recalls = [r["sell"]["recall"] for r in all_reports]
    print(f"Accuracy:    mean={np.mean(accuracies):.3f}, std={np.std(accuracies):.3f}")
    print(f"Buy recall:  mean={np.mean(buy_recalls):.3f}, std={np.std(buy_recalls):.3f}")
    print(f"Sell recall: mean={np.mean(sell_recalls):.3f}, std={np.std(sell_recalls):.3f}")
    print(
        "\nВисокий std відносно mean -- результат нестабільний між періодами,"
        " не довіряй одному найкращому вікну."
    )


def train_final_model(df: pd.DataFrame, feature_columns: list) -> XGBClassifier:
    """Фінальна модель для продакшну -- тренується на ВСІХ доступних даних."""
    print(f"\n{'=' * 60}")
    print("ФІНАЛЬНЕ ТРЕНУВАННЯ (на всіх даних, для inference-сервісу)")
    print(f"{'=' * 60}")

    X = df[feature_columns]
    y = df[TARGET_COL].astype(int)

    model = build_model()
    model.fit(X, y)

    importance = pd.Series(model.feature_importances_, index=feature_columns)
    print("\nFeature importance (топ-10):")
    print(importance.sort_values(ascending=False).head(10))

    return model


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Завантаження датасету...")
    df = pd.read_csv(DATASET_FILE, index_col=0, parse_dates=True)
    df = df.sort_index()  # критично: walk-forward вимагає строгого хронологічного порядку

    if "symbol" in df.columns:
        symbols_present = df["symbol"].unique().tolist()
        print(f"Символи в датасеті: {symbols_present}")

    feature_columns = [c for c in df.columns if c not in EXCLUDE_FROM_FEATURES]
    print(f"{len(feature_columns)} фічей, {len(df)} рядків загалом")

    walk_forward_validate(df, feature_columns)

    model = train_final_model(df, feature_columns)

    model_path = os.path.join(MODELS_DIR, "xgb_model.json")
    model.save_model(model_path)
    print(f"\nМодель збережено: {model_path}")

    feature_order_path = os.path.join(MODELS_DIR, "feature_order.json")
    with open(feature_order_path, "w") as f:
        json.dump(feature_columns, f, indent=2)
    print(f"Порядок фічей збережено: {feature_order_path}")

    print(
        "\nУВАГА: фінальна модель натренована на ВСІХ даних включно з найновішими."
        " Метрики walk-forward вище -- ваш РЕАЛЬНИЙ орієнтир якості,"
        " а не повторний прогон на тих самих даних, що вже бачила фінальна модель."
    )


if __name__ == "__main__":
    main()
