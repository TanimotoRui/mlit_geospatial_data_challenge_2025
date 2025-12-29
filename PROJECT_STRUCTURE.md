# プロジェクト構造

## ディレクトリ構成

```
mlit_geospatial_data_challenge_2025/
│
├── data/                      # データディレクトリ（Gitには含めない）
│   ├── raw/                   # 元データ（変更厳禁）
│   │   ├── kokudo_suuchi/    # 国土数値情報
│   │   ├── touki_data/       # 登記所備付地図データ
│   │   └── lifull/           # LIFULL物件データ
│   ├── interim/              # 中間処理データ
│   ├── processed/            # 前処理済みデータ
│   └── external/             # 外部から追加で取得したデータ
│
├── notebooks/                 # Jupyterノートブック
│   ├── eda/                  # 探索的データ分析
│   ├── features/             # 特徴量エンジニアリング
│   └── models/               # モデル開発・実験
│
├── src/                      # 再利用可能なPythonコード
│   ├── __init__.py
│   ├── data/                 # データ取得・前処理
│   │   ├── __init__.py
│   │   ├── load.py          # データ読み込み
│   │   └── preprocess.py    # 前処理
│   ├── features/             # 特徴量エンジニアリング
│   │   ├── __init__.py
│   │   ├── gis_features.py  # GIS特徴量
│   │   └── property_features.py  # 物件特徴量
│   ├── models/               # モデル定義・学習・予測
│   │   ├── __init__.py
│   │   ├── train.py         # モデル学習
│   │   └── predict.py       # 予測
│   ├── visualization/        # 可視化
│   │   ├── __init__.py
│   │   └── plot.py
│   └── utils/                # ユーティリティ関数
│       ├── __init__.py
│       ├── logger.py        # ログ管理
│       └── metrics.py       # 評価指標
│
├── experiments/              # 実験管理
│   ├── configs/             # 実験設定ファイル（YAML/JSON）
│   ├── results/             # 実験結果（メトリクス、ログ）
│   └── models/              # 学習済みモデル
│
├── submissions/             # 提出ファイル
│   └── {datetime}_submission.csv
│
├── tests/                   # テストコード
│   ├── test_data.py
│   ├── test_features.py
│   └── test_models.py
│
├── docs/                    # ドキュメント
│   ├── data_description.md  # データ説明
│   ├── features.md         # 特徴量説明
│   └── experiments.md      # 実験ログ
│
├── .gitignore              # Git除外設定
├── requirements.txt        # Python依存パッケージ
├── setup.py               # パッケージ設定
├── README.md              # プロジェクト概要
└── PROJECT_STRUCTURE.md   # このファイル
```

## ディレクトリの使い方

### 1. data/
- **raw/**: ダウンロードした元データをそのまま保存。**絶対に編集しない**
- **interim/**: 中間処理データ（GISデータの結合など）
- **processed/**: 学習・予測に使用する最終的な前処理済みデータ
- **external/**: 自分で追加取得したデータ

### 2. notebooks/
- **eda/**: データの理解、可視化、統計分析
  - 例: `01_initial_eda.ipynb`, `02_gis_analysis.ipynb`
- **features/**: 特徴量の探索と作成
  - 例: `01_basic_features.ipynb`, `02_distance_features.ipynb`
- **models/**: モデルの実験とチューニング
  - 例: `01_baseline_model.ipynb`, `02_lightgbm_tuning.ipynb`

ノートブックは番号付けして順序を明確にする。

### 3. src/
再利用可能なコードをモジュール化。ノートブックで試したコードをここに移行。

```python
# 使用例
from src.data.load import load_lifull_data
from src.features.gis_features import calculate_distance_features
from src.models.train import train_model
```

### 4. experiments/
実験の再現性を確保するためのディレクトリ。

- **configs/**: 実験設定（ハイパーパラメータ、特徴量リストなど）
  ```yaml
  # experiments/configs/exp001.yaml
  model: lightgbm
  features: [...]
  params:
    num_leaves: 31
    learning_rate: 0.05
  ```

- **results/**: 実験結果を記録
  ```
  experiments/results/exp001/
  ├── metrics.json
  ├── cv_scores.csv
  └── feature_importance.png
  ```

- **models/**: 学習済みモデルを保存
  ```
  experiments/models/exp001_model.pkl
  ```

### 5. submissions/
SIGNATE提出用のCSVファイルを保存。タイムスタンプを含めてバージョン管理。

```
20260105_120000_exp001_submission.csv
20260107_150000_exp015_submission.csv
```

### 6. tests/
コードの正常性を確認するテスト。

```bash
pytest tests/
```

### 7. docs/
実験ログやメモ、データの説明などを記録。

## ワークフロー例

1. **データ探索** (`notebooks/eda/`)
   - データの理解、欠損値確認、可視化

2. **データ前処理** (`src/data/`)
   - raw → interim → processed

3. **特徴量エンジニアリング** (`notebooks/features/` → `src/features/`)
   - ノートブックで試行錯誤 → 有効な処理をsrcに移行

4. **モデル実験** (`notebooks/models/`)
   - ベースライン作成 → 改善実験

5. **実験管理** (`experiments/`)
   - 設定ファイル作成 → 学習実行 → 結果記録

6. **提出** (`submissions/`)
   - 予測結果をCSVで保存 → SIGNATE提出

## 実験管理のベストプラクティス

### 1. 実験番号管理
すべての実験に番号を付ける: `exp001`, `exp002`, ...

### 2. 設定ファイルによる管理
ハイパーパラメータなどはYAML/JSONで管理し、コードにハードコーディングしない。

### 3. ログの記録
- 実験日時
- 使用した特徴量
- CV Score
- Public LB Score
- 実験の目的とメモ

### 4. 再現性の確保
- random seed固定
- 設定ファイル保存
- モデル保存
- requirements.txtで環境管理

### 5. バージョン管理
- コードはGit管理
- データとモデルは`.gitignore`で除外（大容量のため）
- DVC等の使用も検討

## 推奨ツール

- **データ処理**: pandas, geopandas, shapely
- **可視化**: matplotlib, seaborn, folium
- **機械学習**: scikit-learn, lightgbm, xgboost, catboost
- **実験管理**: mlflow, wandb, optuna
- **環境管理**: uv (高速パッケージマネージャー)

## 次のステップ

1. ✅ `.gitignore`の設定
2. ✅ `pyproject.toml`の作成（uvでパッケージ管理）
3. ✅ `src/`配下のパッケージ設定
4. データのダウンロードと配置（既に配置済み）
5. EDAの開始

### uvを使った開発の開始

```bash
# 依存関係のインストール
uv sync --extra dev

# Jupyter Labの起動
uv run jupyter lab

# 詳しい使い方は docs/uv_usage.md を参照
```

