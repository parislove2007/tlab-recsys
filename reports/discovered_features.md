# Признаки, найденные поиском (llm_recorded, сид 0)

Принято 50 программ из 421 предложенных. Ниже — 3 с наибольшим вкладом, в виде SQL. `:T` — дата предсказания; обратите внимание, что во всех запросах условие по времени строго одностороннее.

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
