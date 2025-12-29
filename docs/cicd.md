# CI/CD ガイド

## 概要

このプロジェクトでは、GitHub Actionsを使用した最小限のCI/CDパイプラインを構築しています。

## CI/CDワークフロー

### 1. CI (継続的インテグレーション)

**ファイル**: `.github/workflows/ci.yml`

**トリガー**:
- `main`または`develop`ブランチへのpush
- `main`または`develop`ブランチへのPull Request

**実行内容**:
1. ✅ コードフォーマットチェック（black）
2. ✅ import順序チェック（isort）
3. ✅ リント（flake8）
4. ✅ テスト実行（pytest）
5. ✅ カバレッジレポート生成

### 2. Auto Format (自動フォーマット)

**ファイル**: `.github/workflows/format.yml`

**トリガー**:
- Pull Requestの作成・更新

**実行内容**:
- コードを自動的にblackとisortでフォーマット
- 変更があれば自動的にコミット・プッシュ

## Pre-commit フック

コミット前に自動的にチェックを実行します。

### セットアップ

```bash
# pre-commitフックのインストール
make install-dev

# または
uv run pre-commit install
```

### 手動実行

```bash
# すべてのファイルに対して実行
make pre-commit

# または
uv run pre-commit run --all-files
```

### 設定内容

`.pre-commit-config.yaml`で以下をチェック：

- 末尾の空白除去
- ファイル末尾の改行
- YAML/JSON/TOMLの構文チェック
- 大きなファイルの検出（10MB以上）
- コードフォーマット（black）
- import順序（isort）
- リント（flake8）

## ローカルでのチェック

プッシュ前にローカルで確認：

```bash
# すべてのチェックを実行
make check

# 個別に実行
make format    # フォーマット
make lint      # リント
make test      # テスト
```

## ワークフロー

### 新機能開発の流れ

1. **ブランチ作成**
```bash
git checkout -b feature/new-feature
```

2. **開発**
```bash
# コードを書く
# ...

# フォーマット
make format
```

3. **コミット**
```bash
git add .
git commit -m "feat: add new feature"
# pre-commitフックが自動実行される
```

4. **プッシュ前チェック**
```bash
make check
```

5. **プッシュ**
```bash
git push origin feature/new-feature
```

6. **Pull Request作成**
- GitHubでPRを作成
- CIが自動実行される
- 必要に応じて自動フォーマットが適用される

7. **レビュー・マージ**
- CIが全て通ったらマージ

## CI/CDのステータス確認

### GitHub Actionsでの確認

1. GitHubリポジトリの「Actions」タブを開く
2. 各ワークフローの実行状況を確認
3. 失敗した場合はログを確認して修正

### ローカルでの模擬実行

```bash
# CIと同じチェックをローカルで実行
make check
```

## トラブルシューティング

### CIが失敗する場合

1. **フォーマットエラー**
```bash
make format
git add .
git commit --amend --no-edit
git push --force-with-lease
```

2. **テストエラー**
```bash
make test
# エラーを修正
git add .
git commit -m "fix: resolve test errors"
git push
```

3. **リントエラー**
```bash
make lint
# 指摘された箇所を修正
```

### Pre-commitフックをスキップ

緊急時のみ使用（非推奨）：

```bash
git commit --no-verify -m "commit message"
```

## ベストプラクティス

1. **コミット前**: `make check`を実行
2. **プッシュ前**: テストが全て通ることを確認
3. **PR作成時**: テンプレートに従って記入
4. **小さなコミット**: 頻繁に小さい単位でコミット
5. **意味のあるコミットメッセージ**: プレフィックスを使用
   - `feat:` - 新機能
   - `fix:` - バグ修正
   - `docs:` - ドキュメント
   - `style:` - フォーマット
   - `refactor:` - リファクタリング
   - `test:` - テスト追加・修正
   - `chore:` - その他

## 設定ファイル

- `.github/workflows/ci.yml` - メインのCIワークフロー
- `.github/workflows/format.yml` - 自動フォーマット
- `.pre-commit-config.yaml` - pre-commit設定
- `.flake8` - flake8設定
- `pyproject.toml` - black, isort, pytest, mypy設定

## カバレッジレポート

テスト実行後、`htmlcov/index.html`でカバレッジを確認：

```bash
make test-cov
open htmlcov/index.html  # macOS
# または
xdg-open htmlcov/index.html  # Linux
```

## 今後の拡張案

必要に応じて追加可能：

- [ ] 依存関係の脆弱性チェック（Dependabot）
- [ ] ドキュメント自動生成
- [ ] Dockerイメージのビルド
- [ ] モデルのバージョン管理
- [ ] デプロイメント自動化

