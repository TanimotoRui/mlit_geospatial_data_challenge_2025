# UV パッケージマネージャー 使い方ガイド

## uvとは？

`uv`はRustで書かれた、高速で信頼性の高いPythonパッケージマネージャーです。pipやpip-toolsよりも10-100倍高速で、プロジェクトの依存関係管理を簡単にします。

## 特徴

- 🚀 **高速**: pipより10-100倍高速なパッケージインストール
- 🔒 **ロックファイル**: `uv.lock`で依存関係の正確なバージョンを記録
- 🎯 **シンプル**: 仮想環境の自動管理
- 🔄 **互換性**: pipと同じパッケージエコシステム

## 基本コマンド

### プロジェクトのセットアップ

```bash
# 依存関係のインストール（仮想環境も自動作成）
uv sync

# 開発用依存関係も含めてインストール
uv sync --extra dev

# 実験追跡ツールも含める
uv sync --extra experiment-tracking

# すべてのオプション依存関係
uv sync --all-extras
```

### パッケージの管理

```bash
# パッケージの追加
uv add pandas numpy

# 開発用パッケージの追加
uv add --dev pytest black

# 特定のバージョンを指定
uv add "pandas>=2.0.0,<3.0.0"

# パッケージの削除
uv remove package-name

# パッケージのアップデート
uv sync --upgrade-package pandas
```

### スクリプト・コマンドの実行

```bash
# Pythonスクリプトの実行
uv run python script.py

# Jupyter Notebookの起動
uv run jupyter notebook

# Jupyter Labの起動
uv run jupyter lab

# テストの実行
uv run pytest

# フォーマッター実行
uv run black src/

# インタラクティブシェル
uv run python
uv run ipython  # ipythonがインストールされている場合
```

### 環境管理

```bash
# 仮想環境の場所を確認
uv venv --help

# 特定のPythonバージョンを使用
uv venv --python 3.11

# 仮想環境の削除と再作成
rm -rf .venv
uv sync
```

### ロックファイル

```bash
# ロックファイルの更新（新しいバージョンがあれば）
uv lock --upgrade

# 特定のパッケージのみアップデート
uv lock --upgrade-package pandas
```

## プロジェクト設定ファイル

### pyproject.toml

プロジェクトの設定と依存関係は`pyproject.toml`で管理します。

```toml
[project]
name = "mlit-geospatial-challenge"
version = "0.1.0"
requires-python = ">=3.9"

dependencies = [
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    # ...
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "black>=23.10.0",
    # ...
]
```

### uv.lock

`uv.lock`ファイルは自動生成され、すべての依存関係の正確なバージョンを記録します。
このファイルはGitにコミットすることで、チームメンバー全員が同じ環境を再現できます。

## ワークフロー例

### データサイエンスプロジェクトの典型的な流れ

```bash
# 1. プロジェクトのセットアップ
uv sync --extra dev

# 2. Jupyter Notebookで分析
uv run jupyter lab

# 3. 新しいパッケージが必要になったら追加
uv add seaborn plotly

# 4. コードをスクリプト化
uv run python src/data/preprocess.py

# 5. テストを実行
uv run pytest tests/

# 6. コード品質チェック
uv run black src/
uv run flake8 src/
```

### 実験を実行する

```bash
# ベースラインモデルの学習
uv run python src/models/train.py --config experiments/configs/exp001_baseline.yaml

# ノートブックで結果を確認
uv run jupyter notebook notebooks/models/01_baseline_lightgbm.ipynb
```

## トラブルシューティング

### パッケージのインストールに失敗する

```bash
# キャッシュをクリアして再試行
rm -rf ~/.cache/uv
uv sync

# より詳細なログを表示
uv sync -v
```

### 仮想環境をリセットしたい

```bash
# .venvディレクトリを削除して再作成
rm -rf .venv
uv sync
```

### 特定のPythonバージョンを使いたい

```bash
# .python-versionファイルで指定
echo "3.11" > .python-version
uv sync
```

## pipとの比較

| 操作 | pip | uv |
|------|-----|-----|
| インストール | `pip install -r requirements.txt` | `uv sync` |
| パッケージ追加 | `pip install pandas` | `uv add pandas` |
| スクリプト実行 | `python script.py` | `uv run python script.py` |
| 仮想環境 | `python -m venv venv && source venv/bin/activate` | 自動管理 |
| ロックファイル | pip-tools必要 | 標準機能 |

## 参考リンク

- [uv 公式ドキュメント](https://github.com/astral-sh/uv)
- [pyproject.toml リファレンス](https://packaging.python.org/en/latest/specifications/declaring-project-metadata/)

## よくある質問

### Q: requirements.txtは残しておくべき？

A: `pyproject.toml`があれば不要ですが、CI/CDや他のツールとの互換性のために残しておくこともできます。その場合、以下のコマンドで生成できます：

```bash
uv pip compile pyproject.toml -o requirements.txt
```

### Q: JupyterでインストールしたパッケージをすぐNinに使いたい

A: ノートブック内で以下を実行：

```python
!uv add package-name
```

その後、カーネルを再起動してください。

### Q: チームメンバーと環境を共有したい

A: `pyproject.toml`と`uv.lock`の両方をGitにコミットしてください。メンバーは`uv sync`を実行するだけで同じ環境が再現されます。

