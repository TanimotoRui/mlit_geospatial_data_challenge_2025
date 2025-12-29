"""
データ前処理モジュール
"""

from typing import List, Tuple

import numpy as np
import pandas as pd


def expand_slash_features(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    スラッシュ区切りの特徴量をone-hot展開

    Args:
        df: DataFrame
        columns: 展開する列名のリスト

    Returns:
        展開後のDataFrame
    """
    df_expanded = df.copy()

    for col in columns:
        if col not in df.columns:
            continue

        # スラッシュで分割してone-hot化
        unique_values = set()
        for val in df[col].dropna():
            if isinstance(val, str) and "/" in val:
                unique_values.update(val.split("/"))

        # 各値のone-hot特徴量を作成
        for value in sorted(unique_values):
            new_col = f"{col}_{value}"
            df_expanded[new_col] = df[col].apply(
                lambda x: 1 if isinstance(x, str) and value in x.split("/") else 0
            )

    return df_expanded


def process_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    日付特徴量の処理

    Args:
        df: DataFrame

    Returns:
        処理後のDataFrame
    """
    df_processed = df.copy()

    date_columns = [
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
    ]

    for col in date_columns:
        if col not in df.columns:
            continue

        # 日付をdatetimeに変換
        df_processed[col] = pd.to_datetime(df_processed[col], errors="coerce")

        # 年、月、年月を抽出
        df_processed[f"{col}_year"] = df_processed[col].dt.year
        df_processed[f"{col}_month"] = df_processed[col].dt.month
        df_processed[f"{col}_ym"] = (
            df_processed[col].dt.year * 100 + df_processed[col].dt.month
        )

        # 元の列を削除
        df_processed = df_processed.drop(columns=[col])

    return df_processed


def process_address_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    住所特徴量の処理

    Args:
        df: DataFrame

    Returns:
        処理後のDataFrame
    """
    df_processed = df.copy()

    # 都道府県と市区町村を抽出
    if "full_address" in df.columns:
        df_processed["prefecture"] = df_processed["full_address"].str[:3]

        # 市区町村の抽出（簡易版）
        df_processed["city"] = df_processed["full_address"].apply(
            lambda x: (
                x.split("市")[0] + "市"
                if "市" in str(x)
                else (
                    x.split("区")[0] + "区"
                    if "区" in str(x)
                    else (
                        x.split("町")[0] + "町"
                        if "町" in str(x)
                        else (
                            x.split("村")[0] + "村" if "村" in str(x) else str(x)[:10]
                        )
                    )
                )
            )
        )

    return df_processed


def preprocess_for_catboost(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str = "money_room",
    apply_log: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, List[str]]:
    """
    CatBoost用の前処理

    Args:
        train: 学習データ
        test: テストデータ
        target_col: 目的変数のカラム名
        apply_log: 目的変数にlog変換を適用するか

    Returns:
        train_features, test_features, target, cat_features
    """
    print("=" * 60)
    print("前処理開始")
    print("=" * 60)

    # 目的変数の取得
    target = train[target_col].copy()
    if apply_log:
        print("目的変数にlog1p変換を適用")
        target = np.log1p(target)

    # 削除する列
    drop_cols = [
        target_col,
        "building_id",
        "unit_id",
        "bukken_id",  # ID系
    ]

    # テキスト系の列（完全に削除せず、一部は処理）
    text_cols_to_drop = [
        "building_name",
        "building_name_ruby",
        "homes_building_name",
        "homes_building_name_ruby",
        "unit_name",
        "name_ruby",
        "empty_contents",
        "parking_memo",
        "reform_place_other",
        "reform_wet_area_other",
        "reform_interior_other",
        "reform_exterior_other",
        "reform_etc",
        "renovation_etc",
        "money_sonota_str1",
        "money_sonota_str2",
        "money_sonota_str3",
    ]
    drop_cols.extend(text_cols_to_drop)

    # trainとtestを結合
    train_len = len(train)
    combined = pd.concat([train, test], axis=0, ignore_index=True)
    print(f"結合データ shape: {combined.shape}")

    # スラッシュ区切り特徴量の展開
    print("\n[1] スラッシュ区切り特徴量の展開...")
    slash_cols = [
        "building_tag_id",
        "unit_tag_id",
        "reform_interior",
        "reform_exterior",
        "reform_wet_area",
        "statuses",
    ]
    combined = expand_slash_features(combined, slash_cols)
    print(f"展開後 shape: {combined.shape}")

    # 住所特徴量の処理
    print("\n[2] 住所特徴量の処理...")
    combined = process_address_features(combined)

    # 日付特徴量の処理
    print("\n[3] 日付特徴量の処理...")
    combined = process_date_features(combined)
    print(f"日付処理後 shape: {combined.shape}")

    # 不要な列を削除
    combined = combined.drop(
        columns=[col for col in drop_cols if col in combined.columns], errors="ignore"
    )

    # 完全に欠損している列を削除
    null_ratio = combined.isnull().sum() / len(combined)
    cols_to_drop = null_ratio[null_ratio == 1.0].index.tolist()
    if cols_to_drop:
        print(f"\n完全に欠損している列を削除: {len(cols_to_drop)}列")
        combined = combined.drop(columns=cols_to_drop)

    # trainとtestに分割
    train_processed = combined.iloc[:train_len].reset_index(drop=True)
    test_processed = combined.iloc[train_len:].reset_index(drop=True)

    # カテゴリカル特徴量の検出と変換
    print("\n[4] カテゴリカル特徴量の検出...")
    cat_features = []

    for col in train_processed.columns:
        if train_processed[col].dtype == "object":
            cat_features.append(col)
            # NaNを文字列に変換
            train_processed[col] = train_processed[col].fillna("missing").astype(str)
            test_processed[col] = test_processed[col].fillna("missing").astype(str)
        elif (
            train_processed[col].dtype in ["int64", "int32"]
            and train_processed[col].nunique() < 50
        ):
            # 整数型でユニーク数が少ない → カテゴリカルに
            cat_features.append(col)
            train_processed[col] = train_processed[col].fillna(-999).astype(str)
            test_processed[col] = test_processed[col].fillna(-999).astype(str)

    # 数値特徴量の欠損値を埋める
    numeric_cols = train_processed.select_dtypes(include=[np.number]).columns
    train_processed[numeric_cols] = train_processed[numeric_cols].fillna(-999)
    test_processed[numeric_cols] = test_processed[numeric_cols].fillna(-999)

    print(f"\n最終的な特徴量数: {len(train_processed.columns)}")
    print(f"カテゴリカル特徴量数: {len(cat_features)}")
    print(f"数値特徴量数: {len(train_processed.columns) - len(cat_features)}")
    print("=" * 60)

    return train_processed, test_processed, target, cat_features
