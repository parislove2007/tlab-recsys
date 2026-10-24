import polars as pl

RAW = "/home/platon/tlab-recsys/data_raw/kion_train"

it = pl.read_csv(f"{RAW}/interactions.csv", try_parse_dates=True)
print("interactions", it.shape)
print(it.head(3))
print(it.describe())
print("date range:", it["last_watch_dt"].min(), "->", it["last_watch_dt"].max())
print("n_users", it["user_id"].n_unique(), "n_items", it["item_id"].n_unique())
print("dup (u,i,d):", it.shape[0] - it.select(["user_id","item_id","last_watch_dt"]).unique().shape[0])
print("dup (u,i):", it.shape[0] - it.select(["user_id","item_id"]).unique().shape[0])

print("\n-- events per week --")
w = (it.with_columns(pl.col("last_watch_dt").dt.truncate("1w").alias("wk"))
       .group_by("wk").agg(pl.len().alias("n"), pl.col("user_id").n_unique().alias("u"))
       .sort("wk"))
print(w)

print("\n-- watched_pct distribution --")
print(it["watched_pct"].describe())
print(it.select(pl.col("watched_pct").is_null().sum().alias("nulls")))
for q in [0.1,0.25,0.5,0.75,0.9]:
    print(q, it["watched_pct"].quantile(q))

print("\n-- user history length --")
h = it.group_by("user_id").agg(pl.len().alias("n"))
print(h["n"].describe())
for q in [0.5,0.9,0.99]:
    print(q, h["n"].quantile(q))

us = pl.read_csv(f"{RAW}/users.csv")
print("\nusers", us.shape, us.columns)
for c in ["age","income","sex","kids_flg"]:
    print(c, us[c].value_counts().sort("count", descending=True).head(8).to_dicts())

itm = pl.read_csv(f"{RAW}/items.csv")
print("\nitems", itm.shape, itm.columns)
print(itm["content_type"].value_counts().to_dicts())
print("release_year", itm["release_year"].min(), itm["release_year"].max(), "nulls", itm["release_year"].is_null().sum())
for c in ["genres","countries","studios","directors","actors","keywords","age_rating","for_kids"]:
    print(c, "nulls:", itm[c].is_null().sum())
print("\ngenres sample:", itm["genres"].drop_nulls().head(5).to_list())
