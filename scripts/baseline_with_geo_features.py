"""
地理空間特徴量を追加したベースライン

新規追加特徴量:
1. K-meansクラスタリング（緯度経度）+ 集約特徴量
2. Target Encoding（地域別平均価格など）
3. 距離特徴量（主要都市までの距離）
4. 派生特徴量（築年数、単価など）
"""

import gc
import pickle
import sys
from datetime import datetime
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.preprocess import preprocess_for_catboost  # noqa: E402
from src.features.geo_features import (  # noqa: E402
    create_cluster_aggregation_features,
    create_derived_features,
    create_distance_features,
    create_kmeans_clusters,
    create_target_encoding_features,
)
from src.models.train_catboost import predict_with_models, train_catboost_cv  # noqa: E402

warnings.filterwarnings("ignore")

# パス設定
DATA_DIR = project_root / "data"
TRAIN_PATH = DATA_DIR / "raw" / "train.csv"
TEST_PATH = DATA_DIR / "raw" / "test.csv"
SAMPLE_PATH = DATA_DIR / "raw" / "sample_submit.csv"
OUTPUT_DIR = project_root / "submissions" / "exp003_geo_features"
PROCESSED_DIR = DATA_DIR / "processed"

# 前処理済みデータのパス
PROCESSED_TRAIN = PROCESSED_DIR / "train_processed.parquet"
PROCESSED_TEST = PROCESSED_DIR / "test_processed.parquet"
PROCESSED_TARGET = PROCESSED_DIR / "target.parquet"
PROCESSED_CAT_FEATURES = PROCESSED_DIR / "cat_features.pkl"

print("=" * 60)
print("地理空間特徴量を追加したベースライン")
print("=" * 60)

# 前処理済みデータの確認とロード
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
use_cache = all(
    [
        PROCESSED_TRAIN.exists(),
        PROCESSED_TEST.exists(),
        PROCESSED_TARGET.exists(),
        PROCESSED_CAT_FEATURES.exists(),
    ]
)

if use_cache:
    print("\n✅ 前処理済みデータを読み込みます...")
    train_features = pd.read_parquet(PROCESSED_TRAIN)
    test_features = pd.read_parquet(PROCESSED_TEST)
    target = pd.read_parquet(PROCESSED_TARGET).squeeze()
    with open(PROCESSED_CAT_FEATURES, "rb") as f:
        cat_features = pickle.load(f)

    print(f"Train shape: {train_features.shape}")
    print(f"Test shape: {test_features.shape}")
    print(f"カテゴリカル特徴量数: {len(cat_features)}")

else:
    print("\n🔄 前処理を実行します...")

    # データ読み込み
    print("\n[1] データ読み込み...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")

    # 基本前処理
    print("\n[2] 基本前処理...")
    train_features, test_features, target, cat_features = preprocess_for_catboost(
        train, test, target_col="money_room", apply_log=True
    )

    # 元のデータフレームをメモリから削除
    del train, test
    gc.collect()

    # 地理空間特徴量の追加
    print("\n" + "=" * 60)
    print("地理空間特徴量の作成")
    print("=" * 60)

    # 目的変数を一時的に結合（Target Encoding用）
    train_with_target = train_features.copy()
    train_with_target["money_room"] = target

    # train_featuresは一旦不要
    del train_features
    gc.collect()

    # 1. K-meansクラスタリング
    train_with_target, test_features, kmeans = create_kmeans_clusters(
        train_with_target,
        test_features,
        lat_col="lat",
        lon_col="lon",
        n_clusters=50,
        random_state=42,
    )

    # kmeansモデルは不要
    del kmeans
    gc.collect()

    # 2. クラスターごとの集約特徴量
    train_with_target, test_features = create_cluster_aggregation_features(
        train_with_target,
        test_features,
        target_col="money_room",
        cluster_col="geo_cluster",
        agg_cols=["house_area", "year_built", "walk_distance1", "money_kyoueki"],
    )

    # 3. Target Encoding
    train_with_target, test_features = create_target_encoding_features(
        train_with_target,
        test_features,
        target_col="money_room",
        categorical_cols=["city", "prefecture", "eki_name1"],
        smoothing=10.0,
    )

    # 4. 距離特徴量
    train_with_target = create_distance_features(
        train_with_target, lat_col="lat", lon_col="lon"
    )
    test_features = create_distance_features(
        test_features, lat_col="lat", lon_col="lon"
    )

    # 5. 派生特徴量
    train_with_target = create_derived_features(train_with_target)
    test_features = create_derived_features(test_features)

    # 目的変数を分離
    target = train_with_target["money_room"]
    train_features = train_with_target.drop(columns=["money_room"])

    # train_with_targetは不要
    del train_with_target
    gc.collect()

    print(f"\n最終的な特徴量数: {len(train_features.columns)}")

    # カテゴリカル特徴量の更新（新しく追加された特徴量は数値型）
    # geo_clusterはカテゴリカルとして扱う
    if "geo_cluster" in train_features.columns:
        train_features["geo_cluster"] = train_features["geo_cluster"].astype(str)
        test_features["geo_cluster"] = test_features["geo_cluster"].astype(str)
        if "geo_cluster" not in cat_features:
            cat_features.append("geo_cluster")

    print(f"カテゴリカル特徴量数: {len(cat_features)}")

    # 前処理済みデータを保存
    print("\n💾 前処理済みデータを保存...")
    train_features.to_parquet(PROCESSED_TRAIN, index=False)
    test_features.to_parquet(PROCESSED_TEST, index=False)
    pd.DataFrame({"target": target}).to_parquet(PROCESSED_TARGET, index=False)
    with open(PROCESSED_CAT_FEATURES, "wb") as f:
        pickle.dump(cat_features, f)
    print(f"保存先: {PROCESSED_DIR}/")

# sample_submitは常に読み込む（軽いので）
sample_sub = pd.read_csv(SAMPLE_PATH, header=None, names=["id", "money_room"])

# モデルパラメータ
params = {
    "iterations": 500,  # メモリ削減
    "learning_rate": 0.05,
    "depth": 5,  # メモリ削減
    "loss_function": "MAE",
    "eval_metric": "MAE",
    "random_seed": 42,
    "verbose": 100,
    "early_stopping_rounds": 50,
}

# Cross Validation
print("\n" + "=" * 60)
print("Cross Validation")
print("=" * 60)
models, cv_scores = train_catboost_cv(
    train_features,
    target,
    cat_features,
    n_splits=3,
    params=params,
    verbose=100,  # メモリ削減
)

# targetはもう不要
del target
gc.collect()

# テストデータで予測
print("\n[3] テストデータで予測...")
predictions = predict_with_models(models, test_features, cat_features, apply_expm1=True)

# test_featuresはもう不要
del test_features
gc.collect()

# Submission作成
print("\n[4] Submission作成...")
submission = sample_sub.copy()
submission["money_room"] = predictions.astype(int)

# predictionsは不要
del predictions, sample_sub
gc.collect()

# 保存
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = OUTPUT_DIR / f"submission_{timestamp}.csv"
submission.to_csv(output_path, index=False, header=False)

print(f"\n{'='*60}")
print("完了!")
print(f"{'='*60}")
print(f"Submission saved to: {output_path}")
print("\n予測値の統計:")
print(submission["money_room"].describe())
print(f"\nCV MAPE: {np.mean(cv_scores):.4f}% (+/- {np.std(cv_scores):.4f}%)")
print(f"{'='*60}")
print("Ready to submit! 🚀")
print(f"{'='*60}")

# 特徴量重要度の保存
print("\n[5] 特徴量重要度の保存...")
feature_importance = pd.DataFrame(
    {"feature": train_features.columns, "importance": models[0].feature_importances_}
).sort_values("importance", ascending=False)

importance_path = OUTPUT_DIR / f"feature_importance_{timestamp}.csv"
feature_importance.to_csv(importance_path, index=False)
print(f"Feature importance saved to: {importance_path}")

print("\nTop 30 重要な特徴量:")
print(feature_importance.head(30).to_string(index=False))

# 最終的なクリーンアップ
del train_features, models, feature_importance, submission
gc.collect()

print("\n✅ メモリクリーンアップ完了")
