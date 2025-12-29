# 実験ログ

## 実験管理

各実験には以下の情報を記録する：
- 実験番号（exp001, exp002, ...）
- 実験日時
- 使用した特徴量
- モデルとハイパーパラメータ
- CV Score
- Public LB Score
- Private LB Score
- 実験の目的とメモ

---

## exp001 - ベースライン (CatBoost)

**日時**: 2025-12-29

**目的**: 最小限の前処理でサクッと1sub

**特徴量**:
- 数値・カテゴリカル特徴量（自動判定）
- ID系、日付系、テキスト系は除外
- 特徴量数: 約60-70個（自動選択）

**モデル**:
- CatBoost Regressor
  - iterations: 1000
  - learning_rate: 0.05
  - depth: 6
  - loss_function: MAPE
  - early_stopping_rounds: 50

**検証**:
- 5-Fold CV

**結果**:
- CV MAPE: TBD
- Public LB Score: TBD

**メモ**:
- CatBoostは欠損値・カテゴリカル変数をそのまま扱える
- 前処理ほぼなしでも動く
- これをベースに改善していく
- スクリプト: `scripts/baseline_catboost.py`

---

## exp002 - 改善版ベースライン（特徴量エンジニアリング）

**日時**: 2025-12-29

**目的**: Zenn記事を参考にした特徴量エンジニアリング版

**参考**: https://zenn.dev/mmrbulbul/articles/signate-geospatial-challenge-2025-01-baseline

**特徴量**:
- スラッシュ区切り特徴量のone-hot展開（149 → 約500特徴量）
  - `building_tag_id`, `unit_tag_id`, `reform_*`, `statuses`
- 日付特徴量から年・月・年月を抽出
- 住所から都道府県・市区町村を抽出
- 特徴量数: 約500個

**前処理**:
- 目的変数にlog1p変換
- カテゴリカル変数は文字列化
- 数値変数の欠損値は-999で埋める

**モデル**:
- CatBoost Regressor
  - iterations: 1000
  - learning_rate: 0.05
  - depth: 6
  - loss_function: MAE（log変換後）
  - early_stopping_rounds: 50

**検証**:
- 5-Fold CV
- MAPE評価（元のスケールで計算）

**結果**:
- CV MAPE: TBD
- Public LB Score: TBD

**メモ**:
- log変換により相対誤差を学習しやすくした
- 特徴量の大幅増加（exp001の約60個 → 約500個）
- Zenn記事では Public LB: 17.64 MAPE
- スクリプト: `scripts/baseline_improved.py`
- 特徴量重要度も出力

**改善点（exp001との違い）**:
1. スラッシュ区切り特徴量の展開
2. log変換 + MAE損失
3. 日付・住所特徴量の抽出
4. モジュール化（再利用可能）

---

## exp003 - 地理空間特徴量版（K-means + 集約特徴量）

**日時**: 2025-12-30

**目的**: 地理空間特徴量を大幅強化（K-meansクラスタリング + 集約特徴量）

**サーベイ**: 
- 類似コンペのベストプラクティスを調査
- K-meansクラスタリング + 集約特徴量が有効であることを確認
- 参考: https://qiita.com/mountaincat/items/53a71c3b75d6ec8a01c8

**新規追加特徴量**:

1. **K-meansクラスタリング** (n_clusters=50)
   - 緯度経度でクラスタリング
   - クラスター別の集約特徴量（約15個）
     - 目的変数の統計量（mean, median, std, min, max）
     - 物件数
     - その他特徴量の統計量

2. **Target Encoding** (3カテゴリ)
   - city, prefecture, eki_name1
   - スムージング適用（smoothing=10.0）
   - 約6特徴量

3. **距離特徴量** (3個)
   - 東京、大阪、名古屋までの距離

4. **派生特徴量** (9個)
   - 築年数、単価、共益費比率
   - 時系列特徴量（年、月、フラグ）
   - 駅距離の対数変換

**特徴量数**: 
- exp002: 約500 → exp003: 約550-600

**前処理**:
- exp002と同様（log変換、スラッシュ区切り展開など）
- 地理空間特徴量を追加

**モデル**:
- CatBoost Regressor（exp002と同じパラメータ）

**検証**:
- 5-Fold CV

**結果**:
- CV MAPE: TBD
- Public LB Score: TBD

**メモ**:
- 地理情報が重要度上位だったため、地理空間特徴量を強化
- K-meansにより地理的近接性を考慮した集約が可能に
- Target Encodingで地域別価格情報を効率的に利用
- スクリプト: `scripts/baseline_with_geo_features.py`
- 詳細: `docs/geo_features.md`

**改善点（exp002との違い）**:
1. K-meansクラスタリング（50クラスタ）
2. クラスター別集約特徴量
3. Target Encoding
4. 距離・派生特徴量

---

## exp004 - 

**日時**: YYYY-MM-DD

**目的**: 

**特徴量**:
- 

**モデル**:
- 

**検証**:
- 

**結果**:
- CV Score: 
- Public LB Score: 

**メモ**:
- 

---

