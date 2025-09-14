# GEO MVP — Data & API Spec (P0)

> **Цель:** единый, практичный и презентуемый спецификатор для MVP на основе обезличенных геотреков. Охватывает формат данных, приватность, общие метаметки и все функции P0 + `model-status`.

---

## 1) Формат входных данных (CSV → internal cache)

**Файл:** `trips.csv` (~100 МБ)

**Колонки (жёсткий контракт):**
- `randomized_id:int64` — анонимный идентификатор поездки/устройства (сырым наружу не отдаётся)
- `lat:float`, `lng:float` — координаты WGS84, градусы
- `alt:float` — высота, метры (опционально используется)
- `spd:float` — скорость (авто-детект единиц → приводим к km/h)
- `azm:float` — азимут движения, градусы `[0,360)`

**Валидация:** `-90≤lat≤90`, `-180≤lng≤180`, `spd≥0`, `0≤azm<360`, `|alt|<10000`; drop NaN/дубликаты.

**Авто-детект единиц скорости:**
- Если `p95(spd)∈[10,40]` → трактуем как **m/s**, конвертируем в **km/h** для отчётов
- Если `p95(spd)∈[40,140]` → трактуем как **km/h**

**Internal cache:** после ингеста данные сохраняются в `data/cache/points.parquet` (или DuckDB). Все downstream‑модули читают из кэша.

---

## 2) Приватность и безопасность данных

- **Агрегирование по умолчанию**: фронту не отдаются сырые точки и сырые `randomized_id`.
- **k-анонимность**: подавление корзин/кластеров/коридоров с `count < K_ANON` (по умолчанию `K_ANON=5`).
- **Псевдонимизация ID**: `anon_id = hash(SALT || randomized_id)`, `SALT` хранится только в ENV.
- **Дифференциальный шум (опц.)**: Лаплас‑шум `ε∈[1,3]` накладывается **только** при вынужденном показе сырых точек (в MVP не требуется).

---

## 3) Общие метаметки ответа (`meta`)

Каждый эндпоинт (кроме `model-status`) возвращает **единый блок метаданных**:

```json
"meta": {
  "generated_at": "2025-09-14T10:32:05Z",
  "data_version": "trips_csv_v1",
  "query": { /* echo входных параметров */ },
  "privacy": { "k_anon": 5, "epsilon": null, "suppressed": 12 },
  "speed_unit": "km/h",
  "unit_detection_method": "auto-p95",
  "timings_ms": { "compute": 480, "read_cache": 90 }
}
```

---

## 4) Фичи/признаки по поездкам (база для ML и метрик)

**Расчёт на уровне `randomized_id` → `trip_features.parquet`:**
- `path_len_km` — сумма Haversine по последовательным точкам
- `straight_len_km` — Haversine от первой до последней точки
- `detour_ratio = path_len_km / straight_len_km` (guard: если `straight_len_km<0.05` → `1.0`)
- `mean_spd_kmh`, `p95_spd_kmh`, `std_spd_kmh` — по приведённой скорости
- `circ_std_azm` — круговая дисперсия азимутов (детект «зигзага»)

> **Нет timestamp** — не считаем длительность/стопы. Это честно отражается в презентации.

---

## 5) Эндпоинты и схемы

### 5.1. Heatmap — теплокарта спроса
**Route:** `POST /api/v1/analytics/heatmap`

**Вход:**
```json
{
  "bbox": [71.35, 51.06, 71.50, 51.11],
  "time_window": null,
  "grid": {"type": "h3", "level": 8},
  "metrics": ["points", "unique_ids"]
}
```

**Алгоритм:**
1) Каждой точке присваивается H3‑ячейка указанного уровня
2) Агрегации по ячейке: `points`, `unique_ids`, (опц.) `avg_speed_kmh`
3) Нормировка `intensity` на [0..1] по выдаче
4) Применение k‑anon: фильтрация ячеек `<K_ANON`

**Выход:**
```json
{
  "cells": [
    {
      "cell_id": "882a1006b9fffff",
      "centroid": { "lat": 51.0982, "lng": 71.4129 },
      "trips": 412,
      "unique_ids": 127,
      "avg_speed_kmh": 38.4,
      "intensity": 0.92
    }
  ],
  "tiles": {
    "mvt_url": "https://api.example.com/tiles/h3/{z}/{x}/{y}.mvt?level=8",
    "legend_breaks": [5, 20, 50, 100, 200]
  },
  "level": 8,
  "meta": { /* общий meta-блок */ }
}
```

---

### 5.2. Demand Zones — зоны высокого спроса
**Route:** `GET /api/v1/analytics/demand-zones`

**Query:** `?eps_m=250&min_samples=10&bbox=71.35,51.06,71.50,51.11`

**Алгоритм:**
1) Проекция точек в метры (WebMercator)
2) Кластеризация DBSCAN(`eps=eps_m`, `min_samples`)
3) Для каждого кластера: центроид (lat/lng), `radius_p95_m`, `count`, плотность `density_km2`
4) Рассчёт `demand_score∈[0,1]` (min‑max по выдаче)
5) k‑anon фильтрация кластеров `<K_ANON`

**Выход:**
```json
{
  "clusters": [
    {
      "cluster_id": 1,
      "center": { "lat": 51.0982, "lng": 71.4129 },
      "radius_p95_m": 180,
      "count": 63,
      "density_km2": 1120.5,
      "demand_score": 0.87
    }
  ],
  "suppressed_below_k": 5,
  "meta": { /* общий meta-блок */ }
}
```

---

### 5.3. Popular Routes — популярные «коридоры»
**Route:** `POST /api/v1/routes/analyze/popular`

**Вход:**
```json
{
  "bbox": [71.35, 51.06, 71.50, 51.11],
  "top_n": 10,
  "simplify_tolerance_m": 50
}
```

**Алгоритм:**
1) По каждому `randomized_id` строится ломаная траектория
2) Упрощение траектории (Douglas–Peucker, толеранс в метрах)
3) Сегментный хэш: квантование угла (например, 10°) и длины (например, 100 м) → группировка похожих маршрутов
4) Для группы строится репрезентативная polyline, считаются `trips`, `width_p90_m`, `median_speed_kmh`, `median_detour_ratio`
5) k‑anon фильтрация редких коридоров

**Выход:**
```json
{
  "corridors": [
    {
      "corridor_id": "c1",
      "polyline": "_ibE_seK...",
      "trips": 87,
      "width_p90_m": 95,
      "median_speed_kmh": 41.2,
      "median_detour_ratio": 1.18,
      "confidence": 0.81
    }
  ],
  "suppressed_below_k": 5,
  "meta": { /* общий meta-блок */ }
}
```

---

### 5.4. Detect Anomalies — аномалии/сейфти
**Route:** `POST /api/v1/ml/detect-anomalies`

**Вход:**
```json
{
  "top_n": 100,
  "filters": { "bbox": [71.35, 51.06, 71.50, 51.11] },
  "return": { "explanations": true }
}
```

**Алгоритм:**
1) Фичи на поездку из `trip_features.parquet` (см. раздел 4)
2) Эвристики: `overspeed (p95>120)`, `high_detour (ratio>1.6)`, `zigzag (circ_std_azm>60)`
3) IsolationForest (`n_estimators≈100`, `contamination≈0.02`) по фичам
4) Интегральный скор: `final_score = max(rule_severity, iforest_score)`
5) «Почему»: `why` с ключевыми фичами и вкладами (упрощённо: top‑reasons)

**Выход:**
```json
{
  "items": [
    {
      "trip_id": "3f5c19d8...",        
      "score": 0.97,
      "severity": "high",
      "flags": ["overspeed","high_detour"],
      "why": {
        "detour_ratio": 1.85,
        "p95_spd_kmh": 132.4,
        "circ_std_azm": 18.7,
        "top_reasons": [
          { "feature": "detour_ratio", "contrib": 0.46 },
          { "feature": "p95_spd_kmh",  "contrib": 0.39 }
        ]
      },
      "policy_violation": true,
      "action_suggestion": "Flag for review; show driver guidance on safe routing."
    }
  ],
  "model": { "name": "iforest_anomaly", "version": "2025-09-14" },
  "meta": { /* общий meta-блок */ }
}
```

---

### 5.5. Efficiency Metrics — метрики эффективности
**Route:** `GET /api/v1/analytics/efficiency-metrics`

**Запросы:**
- по списку id: `?ids=7637058049336049989,5965568696283616614`
- по bbox: `?bbox=71.35,51.06,71.50,51.11`

**Алгоритм:**
- Из `trip_features.parquet` выбираются строки по фильтру; при `scope=bbox` возвращаются также **распределения** для мгновенных графиков.

**Выход (scope=ids):**
```json
{
  "scope": { "type": "ids", "value": ["3f5c19d8...", "a4b0f1c2..."] },
  "items": [
    {
      "trip_id": "3f5c19d8...",
      "path_len_km": 8.42,
      "straight_len_km": 6.11,
      "detour_ratio": 1.38,
      "mean_spd_kmh": 46.7,
      "p95_spd_kmh": 92.3,
      "std_spd_kmh": 18.4
    }
  ],
  "distributions": null,
  "speed_unit": "km/h",
  "unit_detection_method": "auto-p95",
  "meta": { /* общий meta-блок */ }
}
```

**Выход (scope=bbox):**
```json
{
  "scope": { "type": "bbox", "value": [71.35, 51.06, 71.50, 51.11] },
  "items": [],
  "distributions": {
    "detour_ratio": { "bins": [1.0,1.1,1.2,1.4,1.6,2.0], "counts": [12,45,60,31,9,3] },
    "p95_spd_kmh":  { "bins": [20,40,60,80,100,120,140], "counts": [5,22,48,44,19,4,1] }
  },
  "speed_unit": "km/h",
  "unit_detection_method": "auto-p95",
  "meta": { /* общий meta-блок */ }
}
```

---

### 5.6. Model Status — статус/паспорт ML‑моделей
**Route:** `GET /api/v1/ml/model-status`

**Алгоритм:**
- Чтение `models/meta.json` и атрибутов артефактов (размер, sha256). Возвращается список моделей.

**Выход:**
```json
{
  "models": [
    {
      "name": "iforest_anomaly",
      "ready": true,
      "version": "2025-09-14T10:05:33Z",
      "trained_at": "2025-09-14T10:05:33Z",
      "artifact_path": "models/anom_iforest.joblib",
      "artifact_size_mb": 3.1,
      "artifact_sha256": "c1f0...ab9",
      "features": [
        "path_len_km","straight_len_km","detour_ratio",
        "mean_spd_kmh","p95_spd_kmh","std_spd_kmh","circ_std_azm"
      ],
      "training_data": { "source": "trip_features.parquet", "rows": 18342, "time_window": null }
    }
  ]
}
```

---

## 6) Мини‑KPI/«вау»-маркеры для презентации

- **Heatmap**: `intensity` + `legend_breaks` + MVT‑тайлы → «летает», выглядит как продуктивный слой.
- **Demand Zones**: `radius_p95_m`, `density_km2`, `demand_score` → сразу ясно, где и насколько «горячо».
- **Popular Routes**: `width_p90_m`, `confidence` → визуально читаемые «трубки».
- **Anomalies**: `severity`, `why.top_reasons`, `action_suggestion` → понятная история «что делать».
- **Efficiency**: агрегатные **distributions** для мгновенных графиков.
- **Model Status**: `artifact_sha256`, `artifact_size_mb` → доверие и воспроизводимость.

---

## 7) Ограничения MVP и что дальше

- Нет `timestamp` → не считаем длительности/стопы/час‑пик (добавим при расширении датасета).
- Map‑matching, прогноз спроса, VRP — в roadmap; каркасы эндпоинтов можно оставить.
- Пороговые значения эвристик выносятся в конфиг, могут калиброваться по городу.

---

**Готово.** Этот документ можно использовать как исходник для фронтовых типов (TS), pydantic‑схем на бэке и как основу для разделов презентации (приватность, архитектура, демо).

