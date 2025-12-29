# 地理空間特徴量エンジニアリング

## 概要

特徴量重要度の分析から、地理情報（city, full_address, eki_name1など）が上位にランクインしていることが判明。
これを受けて、地理空間特徴量を強化したモデルを構築。

## サーベイ結果

類似コンペ（不動産価格予測）のベストプラクティスを調査した結果、以下が有効であることが確認された：

### 1. K-meansクラスタリング + 集約特徴量 ✅
- **参考**: [Qiita - K-means法でエリア分類](https://qiita.com/mountaincat/items/53a71c3b75d6ec8a01c8)
- 緯度経度でクラスタリングし、クラスター別の統計量を特徴量化
- 地理的な近接性を考慮した集約が可能

### 2. Target Encoding ✅
- カテゴリカル変数（市区町村、駅名など）ごとの平均価格
- スムージングにより過学習を防止

### 3. 距離特徴量 ✅
- 主要都市までの距離
- 最寄り駅までの距離（数値化）

### 4. 派生特徴量 ✅
- 築年数
- 単価（価格/面積）
- 時系列特徴量

## 実装内容

### ファイル構成

```
src/features/
└── geo_features.py          # 地理空間特徴量モジュール

scripts/
└── baseline_with_geo_features.py  # 実行スクリプト
```

### 作成される特徴量

#### 1. K-meansクラスタリング（n_clusters=50）

```python
- geo_cluster: クラスタID
```

#### 2. クラスター別集約特徴量

**目的変数の統計量**:
- `cluster_target_mean`: クラスター内平均価格
- `cluster_target_median`: クラスター内中央値価格
- `cluster_target_std`: クラスター内価格標準偏差
- `cluster_target_min`: クラスター内最小価格
- `cluster_target_max`: クラスター内最大価格

**物件数**:
- `cluster_count`: クラスター内物件数

**その他の特徴量の集約** (house_area, year_built, walk_distance1, money_kyouekiなど):
- `cluster_{feature}_mean`: クラスター内平均
- `cluster_{feature}_median`: クラスター内中央値

#### 3. Target Encoding

対象カテゴリカル変数: city, prefecture, eki_name1

- `{category}_target_encoded`: スムージング適用済み平均価格
- `{category}_count`: カテゴリ内の物件数

#### 4. 距離特徴量

- `distance_to_tokyo`: 東京都心までの距離
- `distance_to_osaka`: 大阪までの距離
- `distance_to_nagoya`: 名古屋までの距離

#### 5. 派生特徴量

- `building_age`: 築年数（2023 - year_built）
- `price_per_area`: 単価（money_room / house_area）
- `kyoueki_ratio`: 共益費の割合（money_kyoueki / money_room）
- `target_year`: 年
- `target_month`: 月
- `is_january`: 1月フラグ
- `is_july`: 7月フラグ
- `log_walk_distance1`: 駅距離の対数変換

## 実行方法

```bash
# 地理空間特徴量版の実行
make baseline-geo

# または
uv run python scripts/baseline_with_geo_features.py
```

## 期待される効果

### 特徴量数の増加
- **exp002（改善版）**: 約500特徴量
- **exp003（地理空間版）**: 約550-600特徴量

### 予測精度の向上
- クラスター別集約により、地域特性を捉える
- Target Encodingにより、カテゴリカル変数の情報を効率的に利用
- 距離特徴量により、立地の良さを数値化

## K-meansのパラメータ

### クラスタ数の選択
- **現在**: n_clusters=50
- **根拠**: 
  - 全国規模のデータなので、ある程度細かく分割
  - 各クラスタに十分なサンプル数を確保
  - 過度に細かくすると過学習のリスク

### 最適化の余地
- エルボー法やシルエットスコアで最適なクラスタ数を探索
- 都道府県別にクラスタリング（階層的クラスタリング）

## 注意点

### 1. Target Leakage
Target Encodingでは、学習データの目的変数を使用するため、リークのリスクがある。

**対策**:
- スムージングの適用（smoothing=10.0）
- CVの各foldで独立に計算（今後実装予定）

### 2. 欠損値処理
緯度経度が欠損している場合、クラスタリングできない。

**対策**:
- geo_cluster = -1 として別扱い
- 欠損値専用の集約特徴量

### 3. テストデータへの適用
学習データで作成した統計量をテストデータにマージ。

**注意**:
- 未知のカテゴリには全体平均を使用
- クラスタの割り当ては学習済みK-meansモデルで予測

## 今後の改善案

### 1. 階層的クラスタリング
```
都道府県 → 市区町村 → K-means
```

### 2. DBSCANなどの密度ベースクラスタリング
- 密集地域と郊外で異なる特性を捉える

### 3. 半径内の物件統計量
```python
# 例: 半径500m以内の平均価格
def calculate_nearby_stats(lat, lon, radius=0.005):
    ...
```

### 4. 駅からの実際の徒歩距離
- walk_distance1を活用
- 欠損値の補完

### 5. 時系列による集約特徴量の変化
- 2019年の平均価格 vs 2023年の平均価格
- トレンドの把握

## 参考文献

- [Qiita - K-means法による不動産価格予測](https://qiita.com/mountaincat/items/53a71c3b75d6ec8a01c8)
- Kaggle House Prices Advanced Regression Techniques
- 地理空間データを使った機械学習のベストプラクティス

