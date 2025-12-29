.PHONY: help install install-dev sync test format lint clean notebook

help:  ## このヘルプメッセージを表示
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## 依存パッケージのインストール
	uv sync

install-dev:  ## 開発用パッケージも含めてインストール
	uv sync --extra dev
	uv run pre-commit install

install-all:  ## すべてのオプション依存関係をインストール
	uv sync --all-extras

sync:  ## 依存関係の同期
	uv sync

test:  ## テストの実行
	uv run pytest

test-cov:  ## カバレッジ付きでテスト実行
	uv run pytest --cov=src --cov-report=html --cov-report=term-missing

format:  ## コードのフォーマット
	uv run black src/ tests/
	uv run isort src/ tests/

lint:  ## リント実行
	uv run flake8 src/ tests/
	uv run mypy src/

check:  ## フォーマット、リント、テストを実行
	make format
	make lint
	make test

pre-commit:  ## pre-commitフックを実行
	uv run pre-commit run --all-files

baseline:  ## ベースラインモデルを実行
	uv run python scripts/baseline_catboost.py

notebook:  ## Jupyter Labを起動
	uv run jupyter lab

notebook-classic:  ## Jupyter Notebookを起動（クラシック）
	uv run jupyter notebook

clean:  ## 一時ファイルとキャッシュを削除
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name htmlcov -exec rm -rf {} +
	find . -type f -name .coverage -delete
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete

clean-all: clean  ## すべてのビルド成果物を削除
	rm -rf .venv
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info

.DEFAULT_GOAL := help

