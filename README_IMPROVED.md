# 改善版ベースライン 🚀

参考: [Zenn記事 - SIGNATE 第2回 国土交通省 地理空間情報データチャレンジ](https://zenn.dev/mmrbulbul/articles/signate-geospatial-challenge-2025-01-baseline)

## 主な改善点

### 1. スラッシュ区切り特徴量のone-hot展開
**対象カラム**:
- `building_tag_id`
- `unit_tag_id`
- `reform_interior`
- `reform_exterior`
- `reform_wet_area`
- `statuses`

これらのカラムは「値1/値2/値3」のようにスラッシュで区切られており、複数の属性を持っています。
これをone-hot化することで、**149特徴量 → 約500特徴量** に増加。

### 2. log変換 + MAE損失
- **評価指標**: MAPE（相対誤差）
- **学習時**: log1p変換 + MAE損失
  - MAPEは学習に使いにくいため、log変換してMAEで学習
  - 推論時にexpm1で元に戻す

### 3. 日付特徴量の処理
日付カラムから以下を抽出:
- 年 (`_year`)
- 月 (`_month`)
- 年月 (`_ym`)

### 4. 住所特徴量の抽出
`full_address`から:
- 都道府県
- 市区町村

を抽出してカテゴリカル特徴量として使用。

## ファイル構成

```
src/
├── data/
│   └── preprocess.py          # 前処理モジュール
└── models/
    └── train_catboost.py      # モデル学習モジュール

scripts/
├── baseline_catboost.py       # シンプル版（既存）
└── baseline_improved.py       # 改善版（新規）
```

## 実行方法

```bash
# 改善版ベースラインの実行
make baseline-improved

# または
uv run python scripts/baseline_improved.py
```

## 期待される改善

### シンプル版との比較
- **シンプル版**: 約60-70特徴量、基本的な前処理のみ
- **改善版**: 約500特徴量、log変換 + 特徴量エンジニアリング

### Zenn記事の結果
- **Public LB Score**: 17.64 MAPE

## 出力ファイル

```
submissions/
├── YYYYMMDD_HHMMSS_improved_catboost.csv      # 提出ファイル
└── YYYYMMDD_HHMMSS_feature_importance.csv     # 特徴量重要度
```

## モデルパラメータ

```python
{
    'iterations': 1000,
    'learning_rate': 0.05,
    'depth': 6,
    'loss_function': 'MAE',      # log変換後はMAEが効果的
    'eval_metric': 'MAE',
    'early_stopping_rounds': 50,
}
```

## Cross Validation

- **手法**: 5-Fold CV
- **評価**: MAPE（元のスケールで計算）

## 重要な特徴量（参考）

Zenn記事より、以下が重要:
1. `house_area` - 物件面積
2. `year_built` - 築年
3. 市区町村
4. `full_address` - 住所
5. `madori_kind_all` - 間取り
6. 築年関連の日付特徴量
7. 駅情報（駅名、距離）
8. 経緯度

## 今後の改善方向性

1. **時系列分割CV**
   - train: 2019-2022、test: 2023
   - 時系列を考慮したCV戦略

2. **地理空間特徴量**（必須）
   - 国土数値情報の活用
   - 周辺施設との距離
   - 地価情報

3. **さらなる特徴量エンジニアリング**
   - 築年数の計算
   - 駅距離の数値化
   - 集約特徴量（市区町村別平均価格など）

4. **モデルの高度化**
   - LightGBM、XGBoostとのアンサンブル
   - ハイパーパラメータチューニング

## トラブルシューティング

### メモリ不足の場合
スラッシュ区切り特徴量の展開で特徴量が大幅に増えるため、メモリ使用量が増加します。

対処法:
- `iterations`を減らす（1000 → 500）
- 不要な特徴量を削除
- サンプリングして実験

### 実行時間
- CV: 約40-60分
- 全体: 約1時間

## 参考リンク

- [Zenn記事](https://zenn.dev/mmrbulbul/articles/signate-geospatial-challenge-2025-01-baseline)
- [SIGNATE コンペページ](https://signate.jp/)

