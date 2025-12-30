"""
地理空間特徴量エンジニアリング

参考:
- https://qiita.com/mountaincat/items/53a71c3b75d6ec8a01c8
- 不動産価格予測コンペのベストプラクティス
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def create_kmeans_clusters(
    train: pd.DataFrame,
    test: pd.DataFrame,
    lat_col: str = "lat",
    lon_col: str = "lon",
    n_clusters: int = 50,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, KMeans]:
    """
    緯度経度でK-meansクラスタリング

    Args:
        train: 学習データ
        test: テストデータ
        lat_col: 緯度のカラム名
        lon_col: 経度のカラム名
        n_clusters: クラスタ数
        random_state: 乱数シード

    Returns:
        train, test, kmeans_model
    """
    print(f"\n[K-means Clustering] n_clusters={n_clusters}")

    # 緯度経度の結合
    train_copy = train.copy()
    test_copy = test.copy()

    # 欠損値を除外してクラスタリング
    train_valid = train_copy[[lat_col, lon_col]].dropna()

    # 標準化
    scaler = StandardScaler()
    lat_lon_scaled = scaler.fit_transform(train_valid)

    # K-meansクラスタリング
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    train_valid_clusters = kmeans.fit_predict(lat_lon_scaled)

    # 学習データにクラスタを割り当て
    train_copy["geo_cluster"] = -1
    train_copy.loc[train_valid.index, "geo_cluster"] = train_valid_clusters

    # テストデータにクラスタを割り当て
    test_valid = test_copy[[lat_col, lon_col]].dropna()
    if len(test_valid) > 0:
        test_lat_lon_scaled = scaler.transform(test_valid)
        test_valid_clusters = kmeans.predict(test_lat_lon_scaled)
        test_copy["geo_cluster"] = -1
        test_copy.loc[test_valid.index, "geo_cluster"] = test_valid_clusters
    else:
        test_copy["geo_cluster"] = -1

    print(f"Clusters created: {train_copy['geo_cluster'].nunique()} unique clusters")

    return train_copy, test_copy, kmeans


def create_cluster_aggregation_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str = "money_room",
    cluster_col: str = "geo_cluster",
    agg_cols: List[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    クラスターごとの集約特徴量を作成

    Args:
        train: 学習データ
        test: テストデータ
        target_col: 目的変数のカラム名
        cluster_col: クラスタのカラム名
        agg_cols: 集約する数値カラムのリスト

    Returns:
        train, test
    """
    print("\n[Cluster Aggregation Features]")

    if agg_cols is None:
        agg_cols = ["house_area", "year_built", "walk_distance1", "money_kyoueki"]

    # 有効なカラムのみを使用
    agg_cols = [col for col in agg_cols if col in train.columns]

    train_copy = train.copy()
    test_copy = test.copy()

    # クラスターごとの目的変数の統計量
    if target_col in train.columns:
        cluster_target_stats = (
            train_copy.groupby(cluster_col)[target_col]
            .agg(
                [
                    ("cluster_target_mean", "mean"),
                    ("cluster_target_median", "median"),
                    ("cluster_target_std", "std"),
                    ("cluster_target_min", "min"),
                    ("cluster_target_max", "max"),
                ]
            )
            .reset_index()
        )

        train_copy = train_copy.merge(cluster_target_stats, on=cluster_col, how="left")
        test_copy = test_copy.merge(cluster_target_stats, on=cluster_col, how="left")

        print("  - Target aggregation: 5 features")

    # クラスターごとの物件数
    cluster_counts = (
        train_copy.groupby(cluster_col).size().reset_index(name="cluster_count")
    )
    train_copy = train_copy.merge(cluster_counts, on=cluster_col, how="left")
    test_copy = test_copy.merge(cluster_counts, on=cluster_col, how="left")
    print("  - Cluster count: 1 feature")

    # その他の数値特徴量の集約
    for col in agg_cols:
        cluster_agg = (
            train_copy.groupby(cluster_col)[col]
            .agg(
                [
                    (f"cluster_{col}_mean", "mean"),
                    (f"cluster_{col}_median", "median"),
                ]
            )
            .reset_index()
        )

        train_copy = train_copy.merge(cluster_agg, on=cluster_col, how="left")
        test_copy = test_copy.merge(cluster_agg, on=cluster_col, how="left")

    print(f"  - Other aggregations: {len(agg_cols) * 2} features")

    return train_copy, test_copy


def create_target_encoding_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str = "money_room",
    categorical_cols: List[str] = None,
    smoothing: float = 10.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Target Encoding（カテゴリごとの目的変数の平均など）

    Args:
        train: 学習データ
        test: テストデータ
        target_col: 目的変数のカラム名
        categorical_cols: Target Encodingするカテゴリカルカラム
        smoothing: 平滑化パラメータ

    Returns:
        train, test
    """
    print("\n[Target Encoding Features]")

    if categorical_cols is None:
        categorical_cols = ["city", "prefecture", "eki_name1"]

    # 有効なカラムのみを使用
    categorical_cols = [col for col in categorical_cols if col in train.columns]

    if target_col not in train.columns:
        print("  - Target column not found, skipping")
        return train, test

    train_copy = train.copy()
    test_copy = test.copy()

    global_mean = train_copy[target_col].mean()

    for col in categorical_cols:
        # カテゴリごとの統計量
        cat_stats = (
            train_copy.groupby(col)[target_col].agg(["mean", "count"]).reset_index()
        )
        cat_stats.columns = [col, f"{col}_target_mean", f"{col}_count"]

        # スムージング
        cat_stats[f"{col}_target_encoded"] = (
            cat_stats[f"{col}_target_mean"] * cat_stats[f"{col}_count"]
            + global_mean * smoothing
        ) / (cat_stats[f"{col}_count"] + smoothing)

        # マージ
        train_copy = train_copy.merge(
            cat_stats[[col, f"{col}_target_encoded", f"{col}_count"]],
            on=col,
            how="left",
        )
        test_copy = test_copy.merge(
            cat_stats[[col, f"{col}_target_encoded", f"{col}_count"]],
            on=col,
            how="left",
        )

        # 欠損値を全体平均で埋める
        train_copy[f"{col}_target_encoded"].fillna(global_mean, inplace=True)
        test_copy[f"{col}_target_encoded"].fillna(global_mean, inplace=True)
        train_copy[f"{col}_count"].fillna(0, inplace=True)
        test_copy[f"{col}_count"].fillna(0, inplace=True)

    print(f"  - Target encoding: {len(categorical_cols) * 2} features")

    return train_copy, test_copy


def create_distance_features(
    df: pd.DataFrame, lat_col: str = "lat", lon_col: str = "lon"
) -> pd.DataFrame:
    """
    距離関連の特徴量を作成

    Args:
        df: DataFrame
        lat_col: 緯度のカラム名
        lon_col: 経度のカラム名

    Returns:
        DataFrame
    """
    print("\n[Distance Features]")

    # 主要都市の緯度経度（例）
    major_cities = {
        "tokyo": (35.6762, 139.6503),
        "osaka": (34.6937, 135.5023),
        "nagoya": (35.1815, 136.9066),
    }

    # 各主要都市までの距離を一度に計算
    new_columns = {}
    for city_name, (city_lat, city_lon) in major_cities.items():
        new_columns[f"distance_to_{city_name}"] = np.sqrt(
            (df[lat_col] - city_lat) ** 2 + (df[lon_col] - city_lon) ** 2
        )

    # 新しい列を一度に結合
    new_df = pd.DataFrame(new_columns, index=df.index)
    df_copy = pd.concat([df, new_df], axis=1)

    print(f"  - Distance to major cities: {len(major_cities)} features")

    return df_copy


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    派生特徴量の作成

    Args:
        df: DataFrame

    Returns:
        DataFrame
    """
    print("\n[Derived Features]")

    new_columns = {}

    # 築年数の計算
    if "year_built" in df.columns:
        building_age = 2023 - df["year_built"]
        new_columns["building_age"] = building_age.clip(lower=0)

    # 単価（価格/面積）
    if "money_room" in df.columns and "house_area" in df.columns:
        new_columns["price_per_area"] = df["money_room"] / (df["house_area"] + 1)

    # 共益費の割合
    if "money_kyoueki" in df.columns and "money_room" in df.columns:
        new_columns["kyoueki_ratio"] = df["money_kyoueki"] / (df["money_room"] + 1)

    # 時系列特徴量
    if "target_ym" in df.columns:
        target_ym_int = df["target_ym"].astype(int)
        new_columns["target_year"] = target_ym_int // 100
        new_columns["target_month"] = target_ym_int % 100
        new_columns["is_january"] = (new_columns["target_month"] == 1).astype(int)
        new_columns["is_july"] = (new_columns["target_month"] == 7).astype(int)

    # 駅距離の対数変換（外れ値に頑健）
    if "walk_distance1" in df.columns:
        new_columns["log_walk_distance1"] = np.log1p(df["walk_distance1"])

    # 新しい列を一度に結合
    feature_count = len(new_columns)
    if new_columns:
        new_df = pd.DataFrame(new_columns, index=df.index)
        df_copy = pd.concat([df, new_df], axis=1)
    else:
        df_copy = df.copy()

    print(f"  - Derived features: {feature_count} features")

    return df_copy
