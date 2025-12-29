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

## exp003 - 

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

