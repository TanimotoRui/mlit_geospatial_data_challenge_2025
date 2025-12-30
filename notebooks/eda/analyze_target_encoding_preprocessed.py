"""
Target Encodingの効果を分析（前処理済みデータ使用）
trainとtestでのカテゴリカラムの重複率をチェック
"""
import pickle
from pathlib import Path

import pandas as pd

# パス設定
project_root = Path(__file__).resolve().parent.parent
processed_dir = project_root / "data" / "processed"

# 前処理済みデータが存在するかチェック
if not (processed_dir / "train_processed.parquet").exists():
    print("❌ 前処理済みデータが見つかりません")
    print("先に baseline_with_geo_features.py を実行してください")
    exit(1)

print("前処理済みデータ読み込み中...")
train = pd.read_parquet(processed_dir / "train_processed.parquet")
test = pd.read_parquet(processed_dir / "test_processed.parquet")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# Target Encoding対象のカラム
te_columns = ["city", "prefecture", "eki_name1"]

# 実際に存在するカラムのみを使用
te_columns = [col for col in te_columns if col in train.columns]
print(f"\n対象カラム: {te_columns}")

print("\n" + "=" * 80)
print("Target Encoding カラムの重複分析")
print("=" * 80)

for col in te_columns:
    # ユニーク値の数
    train_unique = set(train[col].dropna().unique())
    test_unique = set(test[col].dropna().unique())

    # 重複
    overlap = train_unique & test_unique

    # testにしか存在しない値（trainで学習できない）
    test_only = test_unique - train_unique

    # trainにしか存在しない値
    train_only = train_unique - test_unique

    # カバレッジ率（testの値のうち、trainにも存在する割合）
    coverage = len(overlap) / len(test_unique) * 100 if len(test_unique) > 0 else 0

    # testデータでのカバレッジ（レコード数ベース）
    test_covered = test[test[col].isin(train_unique)]
    record_coverage = len(test_covered) / len(test) * 100

    print(f"\n📊 {col}:")
    print(f"  Train unique values: {len(train_unique):,}")
    print(f"  Test unique values:  {len(test_unique):,}")
    print(f"  Overlap:             {len(overlap):,}")
    print(f"  Test only:           {len(test_only):,}")
    print(f"  Train only:          {len(train_only):,}")
    print(f"  ✅ Coverage (値):     {coverage:.2f}%")
    print(f"  ✅ Coverage (レコード): {record_coverage:.2f}%")

    if len(test_only) > 0 and len(test_only) <= 20:
        print(f"  Test only values: {sorted(list(test_only))}")

    # Target Encodingの効果予測
    if coverage >= 95 and record_coverage >= 95:
        print("  💚 TEが非常に効果的")
    elif coverage >= 80 and record_coverage >= 90:
        print("  💛 TEが効果的（一部未知カテゴリあり）")
    elif coverage >= 50:
        print("  🟠 TEの効果は限定的")
    else:
        print("  🔴 TEはあまり効果的でない可能性")

# 頻度の分析
print("\n" + "=" * 80)
print("カテゴリの頻度分布（Top 10）")
print("=" * 80)

for col in te_columns:
    print(f"\n📈 {col} - Top 10 (Train):")
    top_10 = train[col].value_counts().head(10)
    print(top_10)

    # testでの出現回数
    test_counts = test[col].value_counts()

    print(f"\n   同じカテゴリのTest出現回数:")
    for cat in top_10.index:
        test_count = test_counts.get(cat, 0)
        train_count = top_10[cat]
        ratio = test_count / train_count * 100 if train_count > 0 else 0
        print(f"   {cat}: Train={train_count:,}, Test={test_count:,} ({ratio:.1f}%)")

print("\n" + "=" * 80)
print("総合評価")
print("=" * 80)

print("""
✅ 結論:
  - 前処理済みデータで正しくカラムが作成されています
  - Target Encodingの効果を上記の結果から判断してください
  
推奨アクション:
  - Coverage (レコード) が 95% 以上: そのカラムは非常に有効
  - Coverage (レコード) が 80-95%: そのカラムは有効だが要注意
  - Coverage (レコード) が 80% 未満: 他の特徴量を検討
""")

print("\n✅ 分析完了")

