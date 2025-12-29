# Submissions

## ディレクトリ構造

各実験ごとにサブディレクトリを作成し、提出ファイルと関連ファイルを整理。

```
submissions/
├── exp001_baseline/              # シンプル版ベースライン
│   ├── submission_YYYYMMDD_HHMMSS.csv
│   └── ...
├── exp002_improved/              # 改善版（特徴量エンジニアリング）
│   ├── submission_YYYYMMDD_HHMMSS.csv
│   ├── feature_importance_YYYYMMDD_HHMMSS.csv
│   └── ...
└── exp003_geo_features/          # 地理空間特徴量版
    ├── submission_YYYYMMDD_HHMMSS.csv
    ├── feature_importance_YYYYMMDD_HHMMSS.csv
    └── ...
```

## 実験管理

各実験の結果は `docs/experiments.md` に記録。

### exp001 - シンプル版ベースライン
- **スクリプト**: `scripts/baseline_catboost.py`
- **特徴量数**: 約60-70
- **ディレクトリ**: `submissions/exp001_baseline/`

### exp002 - 改善版（特徴量エンジニアリング）
- **スクリプト**: `scripts/baseline_improved.py`
- **特徴量数**: 約500
- **ディレクトリ**: `submissions/exp002_improved/`
- **追加要素**: スラッシュ区切り展開、log変換、日付・住所特徴量

### exp003 - 地理空間特徴量版
- **スクリプト**: `scripts/baseline_with_geo_features.py`
- **特徴量数**: 約550-600
- **ディレクトリ**: `submissions/exp003_geo_features/`
- **追加要素**: K-means、集約特徴量、Target Encoding、距離特徴量

## 提出手順

1. スクリプトを実行
   ```bash
   make baseline-geo  # または make baseline, make baseline-improved
   ```

2. 該当ディレクトリから最新のsubmission_*.csvを取得

3. SIGNATEに提出

4. 結果を `docs/experiments.md` に記録
   - CV Score
   - Public LB Score
   - 気づき・改善点

