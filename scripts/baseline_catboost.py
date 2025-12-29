"""
CatBoost Baseline Model
最小限の前処理でサクッと1sub
"""

import os
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# パス設定
TRAIN_PATH = "data/raw/train.csv"
TEST_PATH = "data/raw/test.csv"
SAMPLE_PATH = "data/raw/sample_submit.csv"
OUTPUT_DIR = "submissions/exp001_baseline"

print("=" * 60)
print("CatBoost Baseline - 最小限構成")
print("=" * 60)

# データ読み込み
print("\n[1] データ読み込み中...")
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
# sample_submit.csvはヘッダーなし
sample_sub = pd.read_csv(SAMPLE_PATH, header=None, names=['id', 'money_room'])

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Sample submission shape: {sample_sub.shape}")

# 目的変数
target = train["money_room"]
print(f"\nTarget statistics:")
print(target.describe())

# 削除するカラム
drop_cols = [
    "money_room",  # 目的変数
    "building_id",  # ID系
    "unit_id",
    "bukken_id",
    # 日付系（とりあえず削除）
    "building_create_date",
    "building_modify_date",
    "reform_exterior_date",
    "reform_common_area_date",
    "reform_date",
    "reform_wet_area_date",
    "reform_interior_date",
    "renovation_date",
    "snapshot_create_date",
    "new_date",
    "snapshot_modify_date",
    "timelimit_date",
    "usable_date",
    # テキスト系
    "building_name",
    "building_name_ruby",
    "homes_building_name",
    "homes_building_name_ruby",
    "unit_name",
    "name_ruby",
    "full_address",
    "addr2_name",
    "addr3_name",
    "rosen_name1",
    "eki_name1",
    "bus_stop1",
    "rosen_name2",
    "eki_name2",
    "bus_stop2",
    "traffic_other",
    "traffic_car",
    "parking_memo",
    "school_ele_name",
    "school_jun_name",
    "est_other_name",
    "reform_exterior_other",
    "reform_place_other",
    "reform_wet_area_other",
    "reform_interior_other",
    "reform_etc",
    "renovation_etc",
    "empty_contents",
    "money_sonota_str1",
    "money_sonota_str2",
    "money_sonota_str3",
]

# 特徴量作成
print("\n[2] 特徴量作成...")
train_features = train.drop(columns=drop_cols, errors="ignore")
test_features = test.drop(columns=drop_cols, errors="ignore")

# カテゴリカル変数の自動検出と変換
cat_features = []
for col in train_features.columns:
    if train_features[col].dtype == "object":
        cat_features.append(col)
        # NaNを文字列に変換
        train_features[col] = train_features[col].fillna("missing").astype(str)
        test_features[col] = test_features[col].fillna("missing").astype(str)
    elif train_features[col].nunique() < 50 and train_features[col].dtype in ['int64', 'int32']:
        # 整数型でユニーク数が少ないものをカテゴリカルに
        cat_features.append(col)
        train_features[col] = train_features[col].fillna(-999).astype(str)
        test_features[col] = test_features[col].fillna(-999).astype(str)

print(f"特徴量数: {len(train_features.columns)}")
print(f"カテゴリカル特徴量数: {len(cat_features)}")
print(f"カテゴリカル特徴量: {cat_features[:10]}...")  # 最初の10個を表示

# 数値特徴量の欠損値を-999で埋める（CatBoostは欠損値を扱えるが念のため）
numeric_cols = train_features.select_dtypes(include=[np.number]).columns
train_features[numeric_cols] = train_features[numeric_cols].fillna(-999)
test_features[numeric_cols] = test_features[numeric_cols].fillna(-999)

# CatBoost用のPool作成
print("\n[3] CatBoostモデル学習...")
pool_train = Pool(train_features, target, cat_features=cat_features)

# モデル定義
model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    loss_function="MAPE",  # 評価指標に合わせる
    eval_metric="MAPE",
    random_seed=42,
    verbose=100,
    early_stopping_rounds=50,
)

# 学習
model.fit(pool_train)

"""
# CV Score確認（簡易版）
print("\n[4] Cross Validation...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, valid_idx) in enumerate(kf.split(train_features), 1):
    X_train, X_valid = train_features.iloc[train_idx], train_features.iloc[valid_idx]
    y_train, y_valid = target.iloc[train_idx], target.iloc[valid_idx]

    pool_tr = Pool(X_train, y_train, cat_features=cat_features)
    pool_val = Pool(X_valid, y_valid, cat_features=cat_features)

    cv_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        loss_function="MAPE",
        eval_metric="MAPE",
        random_seed=42,
        verbose=0,
        early_stopping_rounds=50,
    )

    cv_model.fit(pool_tr, eval_set=pool_val)

    # MAPE計算
    y_pred = cv_model.predict(X_valid)
    mape = np.mean(np.abs((y_valid - y_pred) / y_valid)) * 100
    cv_scores.append(mape)
    print(f"Fold {fold} MAPE: {mape:.4f}%")

print(f"\nCV MAPE: {np.mean(cv_scores):.4f}% (+/- {np.std(cv_scores):.4f}%)")
"""

# テストデータで予測
print("\n[5] 予測...")
pool_test = Pool(test_features, cat_features=cat_features)
predictions = model.predict(pool_test)

# Submission作成
submission = sample_sub.copy()
submission["money_room"] = predictions

# 保存
os.makedirs(OUTPUT_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = f"{OUTPUT_DIR}/submission_{timestamp}.csv"
submission.to_csv(output_path, index=False, header=False)

print(f"\n[6] 完了!")
print(f"Submission saved to: {output_path}")
print(f"Prediction stats:")
print(submission["money_room"].describe())
print("\n" + "=" * 60)
print("Ready to submit! 🚀")
print("=" * 60)

