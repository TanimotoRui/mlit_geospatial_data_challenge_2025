"""
Target Encodingの効果を分析（前処理を実行して分析）
trainとtestでのカテゴリカラムの重複率をチェック
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd  # noqa: E402

from src.data.preprocess import preprocess_for_catboost  # noqa: E402

print("=" * 80)
print("Target Encoding 効果分析")
print("=" * 80)

# データ読み込み
print("\n[1] 生データ読み込み中...")
train = pd.read_csv(project_root / "data" / "raw" / "train.csv")
test = pd.read_csv(project_root / "data" / "raw" / "test.csv")

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# 前処理実行
print("\n[2] 前処理実行中...")
train_features, test_features, target, cat_features = preprocess_for_catboost(
    train, test, target_col="money_room", apply_log=True
)

print(f"前処理後 Train shape: {train_features.shape}")
print(f"前処理後 Test shape: {test_features.shape}")

# Target Encoding対象のカラム
te_columns = ["city", "prefecture", "eki_name1"]

# 実際に存在するカラムのみを使用
te_columns = [col for col in te_columns if col in train_features.columns]
print(f"\n[3] 対象カラム: {te_columns}")

print("\n" + "=" * 80)
print("Target Encoding カラムの重複分析")
print("=" * 80)

for col in te_columns:
    # ユニーク値の数
    train_unique = set(train_features[col].dropna().unique())
    test_unique = set(test_features[col].dropna().unique())

    # 重複
    overlap = train_unique & test_unique

    # testにしか存在しない値（trainで学習できない）
    test_only = test_unique - train_unique

    # trainにしか存在しない値
    train_only = train_unique - test_unique

    # カバレッジ率（testの値のうち、trainにも存在する割合）
    coverage = len(overlap) / len(test_unique) * 100 if len(test_unique) > 0 else 0

    # testデータでのカバレッジ（レコード数ベース）
    test_covered = test_features[test_features[col].isin(train_unique)]
    record_coverage = len(test_covered) / len(test_features) * 100

    print(f"\n📊 {col}:")
    print(f"  Train unique values: {len(train_unique):,}")
    print(f"  Test unique values:  {len(test_unique):,}")
    print(f"  Overlap:             {len(overlap):,}")
    print(f"  Test only:           {len(test_only):,}")
    print(f"  Train only:          {len(train_only):,}")
    print(f"  ✅ Coverage (値):     {coverage:.2f}%")
    print(f"  ✅ Coverage (レコード): {record_coverage:.2f}%")

    if len(test_only) > 0 and len(test_only) <= 20:
        test_only_sorted = sorted(list(test_only))
        print(f"  Test only values: {test_only_sorted}")

    # Target Encodingの効果予測
    if coverage >= 95 and record_coverage >= 95:
        print("  💚 TEが非常に効果的 → 使うべき")
    elif coverage >= 80 and record_coverage >= 90:
        print("  💛 TEが効果的（一部未知カテゴリあり） → 使ってOK")
    elif coverage >= 50:
        print("  🟠 TEの効果は限定的 → 慎重に判断")
    else:
        print("  🔴 TEはあまり効果的でない可能性 → 他の方法を検討")

# 頻度の分析
print("\n" + "=" * 80)
print("カテゴリの頻度分布（Top 10）")
print("=" * 80)

for col in te_columns:
    print(f"\n📈 {col} - Top 10 (Train):")
    top_10 = train_features[col].value_counts().head(10)
    print(top_10)

    # testでの出現回数
    test_counts = test_features[col].value_counts()

    print(f"\n   同じカテゴリのTest出現回数:")
    for cat in top_10.index:
        test_count = test_counts.get(cat, 0)
        train_count = top_10[cat]
        ratio = test_count / train_count * 100 if train_count > 0 else 0
        print(f"   {cat}: Train={train_count:,}, Test={test_count:,} ({ratio:.1f}%)")

print("\n" + "=" * 80)
print("📋 結論")
print("=" * 80)

print("""
✅ Target Encoding の推奨:
  - 💚 マークのカラム → 絶対使うべき
  - 💛 マークのカラム → 使ってOK
  - 🟠 マークのカラム → 慎重に判断
  - 🔴 マークのカラム → 他の方法を検討
  
現在の設定:
  baseline_with_geo_features.py では以下を使用中:
  ["city", "prefecture", "eki_name1"]
""")

print("\n✅ 分析完了")

