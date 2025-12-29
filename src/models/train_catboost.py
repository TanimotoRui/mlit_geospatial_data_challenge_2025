"""
CatBoostモデルの学習
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    MAPEを計算（log変換された値から元に戻す）

    Args:
        y_true: 真値（log変換済み）
        y_pred: 予測値（log変換済み）

    Returns:
        MAPE
    """
    # log1pを元に戻す
    y_true_original = np.expm1(y_true)
    y_pred_original = np.expm1(y_pred)

    # MAPEを計算
    mape = np.mean(np.abs((y_true_original - y_pred_original) / y_true_original)) * 100
    return mape


def train_catboost_cv(
    X: pd.DataFrame,
    y: pd.Series,
    cat_features: List[str],
    n_splits: int = 5,
    params: Optional[dict] = None,
    verbose: int = 100,
) -> Tuple[List[CatBoostRegressor], List[float]]:
    """
    Cross ValidationでCatBoostを学習

    Args:
        X: 特徴量
        y: 目的変数（log変換済み）
        cat_features: カテゴリカル特徴量のリスト
        n_splits: CV分割数
        params: モデルパラメータ
        verbose: 学習ログの表示間隔

    Returns:
        models, cv_scores
    """
    if params is None:
        params = {
            "iterations": 1000,
            "learning_rate": 0.05,
            "depth": 6,
            "loss_function": "MAE",  # log変換後はMAEが効果的
            "eval_metric": "MAE",
            "random_seed": 42,
            "verbose": verbose,
            "early_stopping_rounds": 50,
        }

    print("=" * 60)
    print("Cross Validation 開始")
    print("=" * 60)
    print(f"パラメータ: {params}")
    print(f"CV分割数: {n_splits}")
    print()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    models = []
    cv_scores = []

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X), 1):
        print(f"\n{'='*60}")
        print(f"Fold {fold}/{n_splits}")
        print(f"{'='*60}")

        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        # Pool作成
        pool_train = Pool(X_train, y_train, cat_features=cat_features)
        pool_valid = Pool(X_valid, y_valid, cat_features=cat_features)

        # モデル学習
        model = CatBoostRegressor(**params)
        model.fit(pool_train, eval_set=pool_valid)

        # 予測（log空間）
        y_pred_log = model.predict(X_valid)

        # MAPEを計算（元のスケールで）
        mape = calculate_mape(y_valid.values, y_pred_log)
        cv_scores.append(mape)

        print(f"\nFold {fold} MAPE: {mape:.4f}%")

        models.append(model)

    print(f"\n{'='*60}")
    print(f"CV MAPE: {np.mean(cv_scores):.4f}% (+/- {np.std(cv_scores):.4f}%)")
    print(f"{'='*60}")

    return models, cv_scores


def train_catboost_full(
    X: pd.DataFrame,
    y: pd.Series,
    cat_features: List[str],
    params: Optional[dict] = None,
    verbose: int = 100,
) -> CatBoostRegressor:
    """
    全データでCatBoostを学習

    Args:
        X: 特徴量
        y: 目的変数（log変換済み）
        cat_features: カテゴリカル特徴量のリスト
        params: モデルパラメータ
        verbose: 学習ログの表示間隔

    Returns:
        学習済みモデル
    """
    if params is None:
        params = {
            "iterations": 1000,
            "learning_rate": 0.05,
            "depth": 6,
            "loss_function": "MAE",
            "eval_metric": "MAE",
            "random_seed": 42,
            "verbose": verbose,
        }

    print("=" * 60)
    print("全データでモデル学習")
    print("=" * 60)

    pool_train = Pool(X, y, cat_features=cat_features)
    model = CatBoostRegressor(**params)
    model.fit(pool_train)

    print("学習完了")
    print("=" * 60)

    return model


def predict_with_models(
    models: List[CatBoostRegressor],
    X: pd.DataFrame,
    cat_features: List[str],
    apply_expm1: bool = True,
) -> np.ndarray:
    """
    複数モデルで予測して平均

    Args:
        models: モデルのリスト
        X: 特徴量
        cat_features: カテゴリカル特徴量のリスト
        apply_expm1: log1pを元に戻すか

    Returns:
        予測値
    """
    pool = Pool(X, cat_features=cat_features)
    predictions = []

    for model in models:
        pred = model.predict(pool)
        predictions.append(pred)

    # 平均
    pred_mean = np.mean(predictions, axis=0)

    # log1pを元に戻す
    if apply_expm1:
        pred_mean = np.expm1(pred_mean)

    return pred_mean
