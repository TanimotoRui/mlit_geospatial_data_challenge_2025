"""
改善版ベースライン（Zenn記事参考）

参考: https://zenn.dev/mmrbulbul/articles/signate-geospatial-challenge-2025-01-baseline

主な改善点:
1. スラッシュ区切り特徴量のone-hot展開
2. log変換 + MAE損失
3. 日付特徴量の処理
4. 住所特徴量の抽出
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime
import warnings

from src.data.preprocess import preprocess_for_catboost
from src.models.train_catboost import train_catboost_cv, predict_with_models

warnings.filterwarnings('ignore')

# パス設定
TRAIN_PATH = "data/raw/train.csv"
TEST_PATH = "data/raw/test.csv"
SAMPLE_PATH = "data/raw/sample_submit.csv"
OUTPUT_DIR = "submissions/exp002_improved"

print("=" * 60)
print("改善版ベースライン - CatBoost + 特徴量エンジニアリング")
print("参考: Zenn記事")
print("=" * 60)

# データ読み込み
print("\n[1] データ読み込み...")
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
sample_sub = pd.read_csv(SAMPLE_PATH, header=None, names=['id', 'money_room'])

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# 前処理
print("\n[2] 前処理...")
train_features, test_features, target, cat_features = preprocess_for_catboost(
    train, test, target_col='money_room', apply_log=True
)

# モデルパラメータ
params = {
    'iterations': 2000,
    'learning_rate': 0.05,
    'depth': 6,
    'loss_function': 'MAE',  # log変換後はMAEが効果的
    'eval_metric': 'MAE',
    'random_seed': 42,
    'verbose': 100,
    'early_stopping_rounds': 50,
}

# Cross Validation
print("\n[3] Cross Validation...")
models, cv_scores = train_catboost_cv(
    train_features, target, cat_features,
    n_splits=5, params=params, verbose=100
)

# テストデータで予測
print("\n[4] テストデータで予測...")
predictions = predict_with_models(
    models, test_features, cat_features, apply_expm1=True
)

# Submission作成
print("\n[5] Submission作成...")
submission = sample_sub.copy()
submission['money_room'] = predictions.astype(int)

# 保存
os.makedirs(OUTPUT_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = f"{OUTPUT_DIR}/submission_{timestamp}.csv"
submission.to_csv(output_path, index=False, header=False)

print(f"\n{'='*60}")
print(f"完了!")
print(f"{'='*60}")
print(f"Submission saved to: {output_path}")
print(f"\n予測値の統計:")
print(submission['money_room'].describe())
print(f"\nCV MAPE: {np.mean(cv_scores):.4f}% (+/- {np.std(cv_scores):.4f}%)")
print(f"{'='*60}")
print("Ready to submit! 🚀")
print(f"{'='*60}")

# 特徴量重要度の保存
print("\n[6] 特徴量重要度の保存...")
feature_importance = pd.DataFrame({
    'feature': train_features.columns,
    'importance': models[0].feature_importances_
}).sort_values('importance', ascending=False)

importance_path = f"{OUTPUT_DIR}/feature_importance_{timestamp}.csv"
feature_importance.to_csv(importance_path, index=False)
print(f"Feature importance saved to: {importance_path}")

print("\nTop 20 重要な特徴量:")
print(feature_importance.head(20).to_string(index=False))

