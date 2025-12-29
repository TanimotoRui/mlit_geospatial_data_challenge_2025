# ベースラインモデル - クイックスタート 🚀

## 最速で1sub する方法

### 1. 環境セットアップ（初回のみ）

```bash
# uvのインストール
curl -LsSf https://astral.sh/uv/install.sh | sh

# 依存関係のインストール
uv sync --extra dev
```

### 2. ベースラインモデル実行

```bash
# 1コマンドで実行
make baseline

# または
uv run python scripts/baseline_catboost.py
```

### 3. 提出

1. `submissions/` ディレクトリ内の最新のCSVファイルを確認
2. SIGNATEにアップロード

## ベースラインの仕様

### モデル
- **CatBoostRegressor**
  - Loss: MAPE（コンペの評価指標に合わせる）
  - Iterations: 1000
  - Learning rate: 0.05
  - Depth: 6
  - Early stopping: 50 rounds

### 特徴量
- **使用**: 数値・カテゴリカル特徴量（自動判定）
- **除外**: 
  - ID系（building_id, unit_id, bukken_id）
  - 日付系（そのまま使用不可のため）
  - テキスト系（住所、駅名など）

### 前処理
- **最小限**: CatBoostは欠損値やカテゴリカルをそのまま扱えるため
- 削除のみ（エンコーディング不要）

## 実行時間の目安

- データ読み込み: 数秒
- モデル学習: 5-10分（マシンスペックによる）
- CV: 20-30分（5-fold）
- 予測: 数秒

**合計**: 約30-40分

## 出力

```
submissions/YYYYMMDD_HHMMSS_baseline_catboost.csv
```

形式: `target_ym,money_room`のCSV

## 期待されるスコア

初回ベースラインなので、LBスコアは参考値として：
- **目標**: MAPE 20-30%程度
- これを起点に改善していきます

## 改善の方向性

ベースライン後の改善案：

1. **特徴量エンジニアリング**
   - 日付から年・月を抽出
   - 築年数の計算
   - 駅距離などの数値化
   - GIS特徴量（国土数値情報を活用）

2. **モデルチューニング**
   - ハイパーパラメータ最適化（Optuna）
   - 別のモデル（LightGBM, XGBoost）
   - アンサンブル

3. **データクレンジング**
   - 外れ値処理
   - 欠損値の補完

4. **CV戦略**
   - 時系列分割
   - Stratified KFold

## トラブルシューティング

### エラーが出る場合

```bash
# 依存関係を再インストール
uv sync --extra dev

# キャッシュクリア
uv cache clean
```

### メモリ不足の場合

スクリプト内の`iterations`を500に減らす：

```python
iterations=500,  # 1000から減らす
```

## 次のステップ

1. ✅ ベースライン実行
2. ⬜ LBスコア確認
3. ⬜ EDAで改善ポイント探索
4. ⬜ 特徴量エンジニアリング
5. ⬜ モデル改善

詳しい実験管理は `docs/experiments.md` に記録しましょう！

