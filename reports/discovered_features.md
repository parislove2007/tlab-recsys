# Признаки, найденные поиском (llm_recorded, сид 0)

Принято 50 программ из 421 предложенных. Ниже — 12 с наибольшим вкладом, в виде SQL. `:T` — дата предсказания; обратите внимание, что во всех запросах условие по времени строго одностороннее.

### `item_sta_w3_count_z_in_group`
вклад по сплитам: 38 134

```sql
-- item_sta_w3_count_z_in_group
SELECT e.item_id, COUNT(*) AS v
FROM events e
WHERE e.ts < :T
  AND e.ts >= :T - INTERVAL '3' DAY
GROUP BY e.item_id
-- then normalised with: z_in_group
```

### `item_sta_w3_nuniq_user_per_day`
вклад по сплитам: 21 221

```sql
-- item_sta_w3_nuniq_user_per_day
SELECT e.item_id, COUNT(DISTINCT e.user_id) AS v
FROM events e
WHERE e.ts < :T
  AND e.ts >= :T - INTERVAL '3' DAY
GROUP BY e.item_id
-- then normalised with: per_day
```

### `item_sta_w7_count_z_in_group`
вклад по сплитам: 21 125

```sql
-- item_sta_w7_count_z_in_group
SELECT e.item_id, COUNT(*) AS v
FROM events e
WHERE e.ts < :T
  AND e.ts >= :T - INTERVAL '7' DAY
GROUP BY e.item_id
-- then normalised with: z_in_group
```

### `item_sta_w3_count_rank_in_group`
вклад по сплитам: 20 799

```sql
-- item_sta_w3_count_rank_in_group
SELECT e.item_id, COUNT(*) AS v
FROM events e
WHERE e.ts < :T
  AND e.ts >= :T - INTERVAL '3' DAY
GROUP BY e.item_id
-- then normalised with: rank_in_group
```

### `mul__user_sta_all_count_log1p__item_sta_w3_count_z_in_group`
вклад по сплитам: 15 162

```sql
-- mul__user_sta_all_count_log1p__item_sta_w3_count_z_in_group
WITH l AS (
-- user_sta_all_count_log1p
SELECT e.user_id, COUNT(*) AS v
FROM events e
WHERE e.ts < :T
GROUP BY e.user_id
-- then normalised with: log1p
), r AS (
-- item_sta_w3_count_z_in_group
SELECT e.item_id, COUNT(*) AS v
FROM events e
WHERE e.ts < :T
  AND e.ts >= :T - INTERVAL '3' DAY
GROUP BY e.item_id
-- then normalised with: z_in_group
)
SELECT c.user_id, c.item_id, l.v * r.v AS value
FROM candidates c
LEFT JOIN l ON l.user_id = c.user_id
LEFT JOIN r ON r.item_id = c.item_id
```

### `item_sta_w7_count_rank_in_group`
вклад по сплитам: 14 362

```sql
-- item_sta_w7_count_rank_in_group
SELECT e.item_id, COUNT(*) AS v
FROM events e
WHERE e.ts < :T
  AND e.ts >= :T - INTERVAL '7' DAY
GROUP BY e.item_id
-- then normalised with: rank_in_group
```

### `item_sta_w3_trend`
вклад по сплитам: 7 326

```sql
-- item_sta_w3_trend
SELECT e.item_id, CAST(COUNT_IF(DATE_DIFF('day', e.ts, :T) < 1.5) AS DOUBLE) / (COUNT_IF(DATE_DIFF('day', e.ts, :T) >= 1.5) + 1) AS v
FROM events e
WHERE e.ts < :T
  AND e.ts >= :T - INTERVAL '3' DAY
GROUP BY e.item_id
```

### `item_sta_w7_trend`
вклад по сплитам: 7 076

```sql
-- item_sta_w7_trend
SELECT e.item_id, CAST(COUNT_IF(DATE_DIFF('day', e.ts, :T) < 3.5) AS DOUBLE) / (COUNT_IF(DATE_DIFF('day', e.ts, :T) >= 3.5) + 1) AS v
FROM events e
WHERE e.ts < :T
  AND e.ts >= :T - INTERVAL '7' DAY
GROUP BY e.item_id
```

### `item_sta_w3_sum_dur_z_in_group`
вклад по сплитам: 4 591

```sql
-- item_sta_w3_sum_dur_z_in_group
SELECT e.item_id, SUM(e.total_dur) AS v
FROM events e
WHERE e.ts < :T
  AND e.ts >= :T - INTERVAL '3' DAY
GROUP BY e.item_id
-- then normalised with: z_in_group
```

### `ratio__item_sta_w3_count__item_sta_w7_count`
вклад по сплитам: 4 352

```sql
-- ratio__item_sta_w3_count__item_sta_w7_count
WITH l AS (
-- item_sta_w3_count
SELECT e.item_id, COUNT(*) AS v
FROM events e
WHERE e.ts < :T
  AND e.ts >= :T - INTERVAL '3' DAY
GROUP BY e.item_id
), r AS (
-- item_sta_w7_count
SELECT e.item_id, COUNT(*) AS v
FROM events e
WHERE e.ts < :T
  AND e.ts >= :T - INTERVAL '7' DAY
GROUP BY e.item_id
)
SELECT c.user_id, c.item_id, l.v / (r.v + 1) AS value
FROM candidates c
LEFT JOIN l ON l.item_id = c.item_id
LEFT JOIN r ON r.item_id = c.item_id
```

### `attr_acto_sta_all_count_sum_share_user`
вклад по сплитам: 4 214

```sql
-- attr_acto_sta_all_count_sum_share_user
SELECT e.user_id, a.value, COUNT(*) AS v
FROM events e
JOIN item_attr a ON a.item_id = e.item_id AND a.attr = 'actors'
WHERE e.ts < :T
GROUP BY e.user_id, a.value
-- then, per candidate (user, item): SUM of v over the item's 'actors' values
-- then normalised with: share_user
```

### `ratio__item_sta_w3_count__item_sta_w30_count`
вклад по сплитам: 3 918

```sql
-- ratio__item_sta_w3_count__item_sta_w30_count
WITH l AS (
-- item_sta_w3_count
SELECT e.item_id, COUNT(*) AS v
FROM events e
WHERE e.ts < :T
  AND e.ts >= :T - INTERVAL '3' DAY
GROUP BY e.item_id
), r AS (
-- item_sta_w30_count
SELECT e.item_id, COUNT(*) AS v
FROM events e
WHERE e.ts < :T
  AND e.ts >= :T - INTERVAL '30' DAY
GROUP BY e.item_id
)
SELECT c.user_id, c.item_id, l.v / (r.v + 1) AS value
FROM candidates c
LEFT JOIN l ON l.item_id = c.item_id
LEFT JOIN r ON r.item_id = c.item_id
```
