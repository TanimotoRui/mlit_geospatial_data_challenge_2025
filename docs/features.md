# 特徴量一覧

## 基本方針

- 特徴量は再現性を確保するため、すべて `src/features/` にモジュール化する
- 新しい特徴量を作成したら、このドキュメントに記録する
- 有効性の評価結果も記載する

---

## 1. 基本特徴量

### 1.1 物件基本情報
- `area`: 面積（㎡）
- `age`: 築年数
- `floor`: 階数
- `rooms`: 部屋数
- `structure`: 構造タイプ（木造、RC造など）

### 1.2 位置情報
- `latitude`: 緯度
- `longitude`: 経度
- `prefecture`: 都道府県
- `city`: 市区町村

---

## 2. 時系列特徴量

### 2.1 時間特徴量
- `year`: 年
- `month`: 月
- `is_january`: 1月フラグ
- `is_july`: 7月フラグ

---

## 3. GIS特徴量

### 3.1 距離特徴量
- `distance_to_station`: 最寄駅までの距離
- `distance_to_city_center`: 都心（主要駅）までの距離
- `distance_to_school`: 最寄学校までの距離
- `distance_to_hospital`: 最寄病院までの距離
- `distance_to_park`: 最寄公園までの距離

### 3.2 周辺施設カウント
- `num_stations_within_1km`: 1km圏内の駅数
- `num_schools_within_500m`: 500m圏内の学校数
- `num_convenience_stores_within_500m`: 500m圏内のコンビニ数

### 3.3 地域特性
- `land_use_category`: 土地利用区分
- `zoning`: 用途地域
- `building_coverage_ratio`: 建蔽率
- `floor_area_ratio`: 容積率

---

## 4. 集約特徴量

### 4.1 地域別統計量
- `mean_price_by_city`: 市区町村別平均価格
- `median_price_by_city`: 市区町村別中央値価格
- `std_price_by_city`: 市区町村別価格標準偏差

### 4.2 物件タイプ別統計量
- `mean_price_by_structure`: 構造別平均価格
- `mean_age_by_structure`: 構造別平均築年数

---

## 5. 派生特徴量

### 5.1 比率特徴量
- `price_per_sqm`: 単価（円/㎡）
- `age_ratio`: 築年数/耐用年数

### 5.2 交互作用特徴量
- `area_x_station_distance`: 面積 × 駅距離
- `age_x_structure`: 築年数 × 構造

---

## 特徴量評価

| 特徴量 | 重要度 | CV Score改善 | 備考 |
|--------|--------|--------------|------|
| area | TBD | TBD | 基本特徴量 |
| age | TBD | TBD | 基本特徴量 |
| distance_to_station | TBD | TBD | GIS特徴量 |
| ... | ... | ... | ... |

---

## TODO

- [ ] 基本特徴量の実装
- [ ] GIS特徴量の実装
- [ ] 集約特徴量の実装
- [ ] 特徴量重要度の分析
- [ ] 相関の高い特徴量の削除

