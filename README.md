# mlit_geospatial_data_challenge_2025

## コンペティション概要

### 基本情報

**大会名**: 第2回 国土交通省 地理空間情報データチャレンジ - 不動産の売買価格を予測しよう!

**主催**: 国土交通省 地理空間情報課

**締切**: 2026年1月9日（金）23:59

**参加者数**: 1,553人 | **投稿数**: 9,996件

### 賞金・報酬

| 順位 | 賞金 |
|------|------|
| 🥇 1位 | 10万円 |
| 🥈 2位 | 5万円 |
| 🥉 3位 | 3万円 |

**注意**: 入賞の条件として、表彰式@G空間EXPO（2026年1月28日予定）への参加が必須です。

---

## コンペの目的

国土交通省が整備するオープンデータ「**国土数値情報**」と民間企業のデータを活用し、**不動産の売買価格の予測モデル**を構築することが目的です。

### 国土数値情報とは？

- 整備開始より50年、一般公開して20年以上の歴史を持つGISデータ
- 約190の項目から構成され、高い品質を確保
- 年間ダウンロード数: 200万件以上（令和5年度）
- **本コンペでは国土数値情報の利用が必須**

---

## 課題内容

### タスク

提供される国土数値情報や民間企業のデータを活用し、**不動産の売買価格を予測**するモデルを構築してください。

### 評価指標: MAPE (Mean Absolute Percentage Error)

$$
MAPE = \frac{100}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|
$$

- $y_i$: 正解値（実際の売買価格）
- $\hat{y}_i$: 予測値
- $n$: サンプル数

**小さいほど精度が高い**（パーセンテージ誤差の平均）

---

## スケジュール

| 日程 | イベント |
|------|----------|
| 2025年11月14日（金） | コンペティション開始 |
| 2025年12月15日（月）23:59 | チーム結成期限 |
| 2026年1月9日（金）23:59 | **コンペティション終了** |
| 2026年1月14日（水） | 入賞者決定（予定） |
| 2026年1月28日（水） | 表彰式@G空間EXPO 2026 |

---

## 提供データ

| データ名 | 内容 | 提供元 |
|----------|------|--------|
| **国土数値情報** | 全国の地理空間情報を含むGISデータ | 国土交通省 |
| **登記所備付地図データ** | 不動産登記に関連する地図・図面の電子データ | 法務省 |
| **物件データ** | 2019年～2023年（1月、7月）の全国の中古マンション・戸建の売買価格と物件情報 | 株式会社LIFULL |

### 分析環境（協賛社提供）

- **Databricks**: 日本語でのデータ解析、予測モデル構築環境（データブリックス・ジャパン株式会社）
- **Snowflake**: AIデータクラウド環境（Snowflake合同会社）

※ 分析環境には利用人数に上限があります

---

## 入賞の流れ

### 評価方法

1. **暫定評価**: コンテスト期間中は評価用データの一部でスコア算出
2. **最終評価**: コンテスト終了後、残りの評価用データでスコア再算出
   - 投稿ファイルは最大2つまで選択可能
   - 最終評価のスコアが高い方が自動的に採用される

### 入賞決定プロセス

1. 最終評価による順位が上位の方を入賞候補者とし、事務局から連絡
2. 入賞候補者は「国土数値情報の利用に関する具体的な方法と効果」のレポートを提出
3. 必要に応じて、作成したモデルの提出を求める場合あり
4. 上記を満たした方の中から入賞者を確定

### 失格条件

以下のいずれかに該当する場合、入賞の資格を失います：

- 事務局からの連絡に期限内に対応しない
- 参加条件やルールを満たしていない
- 表彰式または振り返り会に参加できない
- その他、事務局が不当と判断した場合

---

## 勉強会・サポート

コンペ期間中、以下の勉強会が開催されました：

| 日程 | テーマ | 提供 |
|------|--------|------|
| 11月14日（金）19:00-21:00 | 開会式・勉強会① | GIS・不動産・国土数値情報の基礎 |
| 12月3日（水）19:00-21:00 | Snowflake編 | 初心者向け使い方＆活用例 |
| 12月18日（木）19:00-21:00 | ゼンリン編 | 地理空間情報活用Tips |
| 12月26日（金）19:00-21:00 | 最後の追い込み編 | GA technologies |

---

## 問い合わせ

**専用Slack**: データREADME.pdfに記載の招待リンクから参加

**メール**: dcjimukyoku@pcdua.org（Slackを利用できない場合）

---

## 参考リンク

- [国土交通省 今後の国土数値情報の整備のあり方に関する検討会](https://www.mlit.go.jp/)
- [第1回コンペ表彰式アフタームービー](https://www.youtube.com/)
- [SIGNATE Competition Page](https://signate.jp/)

---

## プロジェクトセットアップ

### 必要環境

- Python 3.9以上
- [uv](https://github.com/astral-sh/uv) (高速パッケージマネージャー)
- Git

### uvのインストール

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# または Homebrew (macOS)
brew install uv
```

### セットアップ手順

```bash
# 1. リポジトリのクローン（既にクローン済みの場合はスキップ）
git clone <repository-url>
cd mlit_geospatial_data_challenge_2025

# 2. 依存パッケージのインストール（仮想環境も自動作成）
uv sync

# 3. 開発用パッケージも含めてインストールする場合
uv sync --extra dev

# 4. データのダウンロード
# SIGNATEからデータをダウンロードし、data/raw/ に配置
```

### 基本的な使い方

```bash
# パッケージの追加
uv add <package-name>

# 開発用パッケージの追加
uv add --dev <package-name>

# Jupyter Labの起動
uv run jupyter lab

# Pythonスクリプトの実行
uv run python src/data/load.py

# テストの実行
uv run pytest

# インタラクティブシェル
uv run python
```

### Makefileを使った便利なコマンド

```bash
# ヘルプの表示
make help

# 依存関係のインストール
make install-dev

# テストの実行
make test

# コードフォーマット
make format

# リント実行
make lint

# すべてのチェック（フォーマット、リント、テスト）
make check

# Jupyter Labの起動
make notebook

# 一時ファイルのクリーンアップ
make clean
```

### ディレクトリ構造

詳細なプロジェクト構造とワークフローについては、[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) を参照してください。

```
mlit_geospatial_data_challenge_2025/
├── data/                      # データ（Gitには含めない）
│   ├── raw/                   # 元データ
│   ├── interim/              # 中間処理データ
│   ├── processed/            # 前処理済みデータ
│   └── external/             # 外部データ
├── notebooks/                 # Jupyter Notebook
│   ├── eda/                  # 探索的データ分析
│   ├── features/             # 特徴量エンジニアリング
│   └── models/               # モデル開発
├── src/                      # 再利用可能なコード
│   ├── data/                 # データ処理
│   ├── features/             # 特徴量エンジニアリング
│   ├── models/               # モデル定義・学習・予測
│   ├── visualization/        # 可視化
│   └── utils/                # ユーティリティ
├── experiments/              # 実験管理
│   ├── configs/             # 実験設定ファイル
│   ├── results/             # 実験結果
│   └── models/              # 学習済みモデル
├── submissions/             # 提出ファイル
├── tests/                   # テストコード
└── docs/                    # ドキュメント
```

### ワークフロー

1. **データ探索**: `notebooks/eda/` でデータの理解と可視化
2. **特徴量エンジニアリング**: `notebooks/features/` で特徴量の試作 → `src/features/` に実装
3. **モデル開発**: `notebooks/models/` で実験 → `experiments/` で管理
4. **提出**: `submissions/` に予測結果を保存して提出

### CI/CD

プロジェクトにはGitHub Actionsによる自動チェックを設定済み：

```bash
# pre-commitフックのインストール（コミット前に自動チェック）
make install-dev

# 手動でチェック実行
make pre-commit

# プッシュ前に全チェック実行
make check
```

詳細は[docs/cicd.md](docs/cicd.md)を参照。

### ドキュメント

- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - プロジェクト構造の詳細
- [docs/uv_usage.md](docs/uv_usage.md) - uvの使い方ガイド
- [docs/cicd.md](docs/cicd.md) - CI/CDガイド
- [docs/data_description.md](docs/data_description.md) - データの説明
- [docs/features.md](docs/features.md) - 特徴量一覧
- [docs/experiments.md](docs/experiments.md) - 実験ログ

### テスト

```bash
# テストの実行
uv run pytest

# カバレッジ付きでテスト
uv run pytest --cov=src --cov-report=html

# 特定のテストファイルを実行
uv run pytest tests/test_metrics.py
```

### コード品質

```bash
# コードフォーマット
uv run black src/ tests/

# import文のソート
uv run isort src/ tests/

# リント
uv run flake8 src/ tests/

# 型チェック
uv run mypy src/

# すべてをまとめて実行
uv run black src/ tests/ && uv run isort src/ tests/ && uv run flake8 src/ tests/
```

## クイックスタート - 最速で1sub 🚀

```bash
# 1. 依存関係のインストール
uv sync --extra dev

# 2. ベースラインモデル実行
make baseline

# 3. submissions/ の最新CSVをSIGNATEに提出
```

詳細は [README_BASELINE.md](README_BASELINE.md) を参照。

## タスク管理

- [x] データのダウンロードと配置
- [x] ベースラインモデルの作成（CatBoost）
- [ ] EDA（探索的データ分析）の実施
- [ ] 特徴量エンジニアリング
- [ ] GIS特徴量の作成
- [ ] モデルの改善
- [ ] アンサンブル

## 参考リンク

- [SIGNATE コンペページ](https://signate.jp/)
- [国土数値情報ダウンロードサイト](https://nlftp.mlit.go.jp/)
- Slackワークスペース: データREADME.pdfを参照

## ライセンス

データの利用規約については、SIGNATE上の参加規約を参照してください。
