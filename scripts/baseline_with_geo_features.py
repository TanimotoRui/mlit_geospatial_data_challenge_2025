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
import time
from datetime import datetime
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.data.preprocess import preprocess_for_catboost  # noqa: E402
from src.features.geo_features import (  # noqa: E402
    create_cluster_aggregation_features, create_derived_features,
    create_distance_features, create_kmeans_clusters,
    create_target_encoding_features)
from src.models.train_catboost import (predict_with_models,  # noqa: E402
                                       train_catboost_cv)

# warnings.filterwarnings("ignore")  # Warning表示を有効化

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

print("=" * 80)
print("🚀 地理空間特徴量を追加したベースライン")
print("=" * 80)
print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# 全体の処理時間を計測
overall_start_time = time.time()

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
    load_start = time.time()

    train_features = pd.read_parquet(PROCESSED_TRAIN)
    print(f"  📁 train_processed.parquet 読み込み完了")

    test_features = pd.read_parquet(PROCESSED_TEST)
    print(f"  📁 test_processed.parquet 読み込み完了")

    target = pd.read_parquet(PROCESSED_TARGET).squeeze()
    print(f"  📁 target.parquet 読み込み完了")

    with open(PROCESSED_CAT_FEATURES, "rb") as f:
        cat_features = pickle.load(f)
    print(f"  📁 cat_features.pkl 読み込み完了")

    load_time = time.time() - load_start
    print(f"\n  ⏱️  データ読み込み時間: {load_time:.2f}秒")
    print(f"\n  📊 Train shape: {train_features.shape}")
    print(f"  📊 Test shape: {test_features.shape}")
    print(f"  📊 カテゴリカル特徴量数: {len(cat_features)}")

else:
    print("\n🔄 前処理を実行します...")
    preprocess_start = time.time()

    # データ読み込み
    print("\n" + "=" * 80)
    print("[STEP 1/7] 📂 データ読み込み")
    print("=" * 80)
    step_start = time.time()

    train = pd.read_csv(TRAIN_PATH, low_memory=False)
    print(f"  ✓ Train data loaded: {train.shape}")

    test = pd.read_csv(TEST_PATH, low_memory=False)
    print(f"  ✓ Test data loaded: {test.shape}")

    print(f"  ⏱️  読み込み時間: {time.time() - step_start:.2f}秒")

    # 基本前処理
    print("\n" + "=" * 80)
    print("[STEP 2/7] 🔧 基本前処理")
    print("=" * 80)
    step_start = time.time()

    train_features, test_features, target, cat_features = preprocess_for_catboost(
        train, test, target_col="money_room", apply_log=True
    )

    print(f"  ⏱️  前処理時間: {time.time() - step_start:.2f}秒")

    # 元のデータフレームをメモリから削除
    del train, test
    gc.collect()

    # 地理空間特徴量の追加
    print("\n" + "=" * 80)
    print("[STEP 3/7] 🌍 地理空間特徴量の作成")
    print("=" * 80)
    geo_start = time.time()

    # 目的変数を一時的に結合（Target Encoding用）
    train_with_target = train_features.copy()
    train_with_target["money_room"] = target

    # train_featuresは一旦不要
    del train_features
    gc.collect()

    # 1. K-meansクラスタリング
    print("\n  [3-1] K-meansクラスタリング...")
    substep_start = time.time()
    train_with_target, test_features, kmeans = create_kmeans_clusters(
        train_with_target,
        test_features,
        lat_col="lat",
        lon_col="lon",
        n_clusters=50,
        random_state=42,
    )
    print(f"        ⏱️  {time.time() - substep_start:.2f}秒")

    # kmeansモデルは不要
    del kmeans
    gc.collect()

    # 2. クラスターごとの集約特徴量
    print("\n  [3-2] クラスター集約特徴量...")
    substep_start = time.time()
    train_with_target, test_features = create_cluster_aggregation_features(
        train_with_target,
        test_features,
        target_col="money_room",
        cluster_col="geo_cluster",
        agg_cols=["house_area", "year_built", "walk_distance1", "money_kyoueki"],
    )
    print(f"        ⏱️  {time.time() - substep_start:.2f}秒")

    # 3. Target Encoding
    print("\n  [3-3] Target Encoding...")
    substep_start = time.time()
    train_with_target, test_features = create_target_encoding_features(
        train_with_target,
        test_features,
        target_col="money_room",
        categorical_cols=["city", "prefecture", "eki_name1"],
        smoothing=10.0,
    )
    print(f"        ⏱️  {time.time() - substep_start:.2f}秒")

    # 4. 距離特徴量
    print("\n  [3-4] 距離特徴量...")
    substep_start = time.time()
    train_with_target = create_distance_features(
        train_with_target, lat_col="lat", lon_col="lon"
    )
    test_features = create_distance_features(
        test_features, lat_col="lat", lon_col="lon"
    )
    print(f"        ⏱️  {time.time() - substep_start:.2f}秒")

    # 5. 派生特徴量
    print("\n  [3-5] 派生特徴量...")
    substep_start = time.time()
    train_with_target = create_derived_features(train_with_target)
    test_features = create_derived_features(test_features)
    print(f"        ⏱️  {time.time() - substep_start:.2f}秒")

    print(f"\n  🌍 地理空間特徴量作成 完了: {time.time() - geo_start:.2f}秒")

    # 目的変数を分離
    target = train_with_target["money_room"]
    train_features = train_with_target.drop(columns=["money_room"])

    # train_with_targetは不要
    del train_with_target
    gc.collect()

    print(f"\n  📊 最終的な特徴量数: {len(train_features.columns)}")

    # カテゴリカル特徴量の更新（新しく追加された特徴量は数値型）
    # geo_clusterはカテゴリカルとして扱う
    if "geo_cluster" in train_features.columns:
        train_features["geo_cluster"] = train_features["geo_cluster"].astype(str)
        test_features["geo_cluster"] = test_features["geo_cluster"].astype(str)
        if "geo_cluster" not in cat_features:
            cat_features.append("geo_cluster")

    print(f"  📊 カテゴリカル特徴量数: {len(cat_features)}")

    # 前処理済みデータを保存
    print("\n" + "=" * 80)
    print("[STEP 4/7] 💾 前処理済みデータを保存")
    print("=" * 80)
    save_start = time.time()

    train_features.to_parquet(PROCESSED_TRAIN, index=False)
    print(f"  ✓ train_processed.parquet 保存完了")

    test_features.to_parquet(PROCESSED_TEST, index=False)
    print(f"  ✓ test_processed.parquet 保存完了")

    pd.DataFrame({"target": target}).to_parquet(PROCESSED_TARGET, index=False)
    print(f"  ✓ target.parquet 保存完了")

    with open(PROCESSED_CAT_FEATURES, "wb") as f:
        pickle.dump(cat_features, f)
    print(f"  ✓ cat_features.pkl 保存完了")

    print(f"  ⏱️  保存時間: {time.time() - save_start:.2f}秒")
    print(f"  📁 保存先: {PROCESSED_DIR}/")

    preprocess_time = time.time() - preprocess_start
    print(f"\n  ✅ 前処理 完了: {preprocess_time:.2f}秒 ({preprocess_time/60:.1f}分)")

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
print("\n" + "=" * 80)
print("[STEP 5/7] 🤖 Cross Validation (3-Fold)")
print("=" * 80)
print(f"  📊 Train samples: {len(train_features):,}")
print(f"  📊 Features: {len(train_features.columns)}")
print(f"  📊 Categorical features: {len(cat_features)}")
print(f"  🎯 Model: CatBoost Regressor")
print(
    f"  🔧 Iterations: {params['iterations']}, Depth: {params['depth']}, LR: {params['learning_rate']}"
)
print("=" * 80)

cv_start = time.time()
models, cv_scores = train_catboost_cv(
    train_features,
    target,
    cat_features,
    n_splits=3,
    params=params,
    verbose=100,  # メモリ削減
)
cv_time = time.time() - cv_start
print(f"\n  ⏱️  CV時間: {cv_time:.2f}秒 ({cv_time/60:.1f}分)")

# targetはもう不要
del target
gc.collect()

# テストデータで予測
print("\n" + "=" * 80)
print("[STEP 6/7] 🔮 テストデータで予測")
print("=" * 80)
pred_start = time.time()

predictions = predict_with_models(models, test_features, cat_features, apply_expm1=True)

pred_time = time.time() - pred_start
print(f"  ✓ 予測完了")
print(f"  ⏱️  予測時間: {pred_time:.2f}秒")

# test_featuresはもう不要
del test_features
gc.collect()

# Submission作成
print("\n" + "=" * 80)
print("[STEP 7/7] 📝 Submission作成")
print("=" * 80)
submission_start = time.time()

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

print(f"  ✓ Submission saved: {output_path}")

# 特徴量重要度の保存
feature_importance = pd.DataFrame(
    {"feature": train_features.columns, "importance": models[0].feature_importances_}
).sort_values("importance", ascending=False)

importance_path = OUTPUT_DIR / f"feature_importance_{timestamp}.csv"
feature_importance.to_csv(importance_path, index=False)
print(f"  ✓ Feature importance saved: {importance_path}")

submission_time = time.time() - submission_start
print(f"  ⏱️  Submission作成時間: {submission_time:.2f}秒")

# 結果サマリー
print("\n" + "=" * 80)
print("🎉 完了!")
print("=" * 80)
print(f"📊 CV結果:")
print(f"  - MAPE: {np.mean(cv_scores):.4f}% (± {np.std(cv_scores):.4f}%)")
print(f"  - Fold scores: {[f'{s:.4f}%' for s in cv_scores]}")
print(f"\n📈 予測値の統計:")
stats = submission["money_room"].describe()
print(f"  - Count: {int(stats['count']):,}")
print(f"  - Mean:  ¥{int(stats['mean']):,}")
print(f"  - Std:   ¥{int(stats['std']):,}")
print(f"  - Min:   ¥{int(stats['min']):,}")
print(f"  - Max:   ¥{int(stats['max']):,}")
print(f"\n📂 出力ファイル:")
print(f"  - Submission: {output_path.name}")
print(f"  - Feature importance: {importance_path.name}")
print(
    f"\n⏱️  総実行時間: {time.time() - overall_start_time:.2f}秒 ({(time.time() - overall_start_time)/60:.1f}分)"
)
print("\nTop 30 重要な特徴量:")
print(feature_importance.head(30).to_string(index=False))
print("\n" + "=" * 80)
print("✅ Ready to submit! 🚀")
print("=" * 80)

# 最終的なクリーンアップ
del train_features, models, feature_importance, submission
gc.collect()
