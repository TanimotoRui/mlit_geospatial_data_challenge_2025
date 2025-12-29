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

## exp002 - 

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

