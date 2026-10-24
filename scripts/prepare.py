"""Turn the raw CSVs into a compact parquet event log plus item/user attribute tables."""
import sys
import polars as pl

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
from tlab.config import RAW, DATA, WATCH_PCT, FINISH_PCT, SAMPLE_CUTOFF, MIN_HISTORY

ev = (
    pl.read_csv(RAW / "interactions.csv", try_parse_dates=True)
    .rename({"last_watch_dt": "ts"})
    .with_columns(
        pl.col("watched_pct").fill_null(0.0),
        pl.col("total_dur").fill_null(0),
    )
    .with_columns(
        (pl.col("watched_pct") >= WATCH_PCT).alias("is_watch"),
        (pl.col("watched_pct") >= FINISH_PCT).alias("is_finish"),
    )
    .sort("ts")
)
print("events", ev.shape, ev["ts"].min(), ev["ts"].max())

active = (
    ev.filter(pl.col("ts") < SAMPLE_CUTOFF)
    .group_by("user_id").agg(pl.len().alias("n"))
    .filter(pl.col("n") >= MIN_HISTORY)
)
print(f"users with >={MIN_HISTORY} events before {SAMPLE_CUTOFF}: {active.height}")
active.select("user_id").write_parquet(DATA / "eligible_users.parquet")
ev.write_parquet(DATA / "events.parquet")

items = pl.read_csv(RAW / "items.csv")

def explode_attr(col, sep=", "):
    return (
        items.select("item_id", col)
        .with_columns(pl.col(col).fill_null("__na__").str.split(sep))
        .explode(col)
        .with_columns(pl.col(col).str.strip_chars())
        .filter(pl.col(col).str.len_chars() > 0)
        .rename({col: "value"})
        .with_columns(pl.lit(col).alias("attr"))
        .unique()
    )

multi = pl.concat([explode_attr(c) for c in ["genres", "countries", "studios", "directors"]])
# actors and keywords are long tails; keep only the frequent values so joins stay cheap
for col, keep in [("actors", 3000), ("keywords", 3000)]:
    e = explode_attr(col)
    top = e.group_by("value").agg(pl.len().alias("n")).sort("n", descending=True).head(keep).select("value")
    multi = pl.concat([multi, e.join(top, on="value")])

single = items.select(
    "item_id",
    pl.col("content_type"),
    pl.col("release_year").fill_null(-1).cast(pl.Int32),
    pl.col("age_rating").fill_null(-1).cast(pl.Int32),
    pl.col("for_kids").fill_null(0).cast(pl.Int32),
    pl.col("genres").str.split(", ").list.len().cast(pl.Int32).alias("n_genres"),
    pl.col("title").str.len_chars().cast(pl.Int32).alias("title_len"),
    pl.col("description").fill_null("").str.len_chars().cast(pl.Int32).alias("descr_len"),
    pl.col("keywords").fill_null("").str.split(", ").list.len().cast(pl.Int32).alias("n_keywords"),
)
# primary genre = the most frequent genre of the item, used as a plain categorical
gen = multi.filter(pl.col("attr") == "genres")
gfreq = gen.group_by("value").agg(pl.len().alias("gn"))
primary = (
    gen.join(gfreq, on="value").sort("gn", descending=True)
    .group_by("item_id").agg(pl.col("value").first().alias("primary_genre"))
)
single = single.join(primary, on="item_id", how="left").with_columns(
    pl.col("primary_genre").fill_null("__na__")
)
single.write_parquet(DATA / "items.parquet")
multi.write_parquet(DATA / "item_attrs.parquet")
print("item_attrs", multi.shape)
print(multi.group_by("attr").agg(pl.len(), pl.col("value").n_unique().alias("uniq")))

users = (
    pl.read_csv(RAW / "users.csv")
    .with_columns(
        pl.col("age").fill_null("__na__"),
        pl.col("income").fill_null("__na__"),
        pl.col("sex").fill_null("__na__"),
        pl.col("kids_flg").fill_null(0).cast(pl.Int32),
    )
)
users.write_parquet(DATA / "users.parquet")
print("users", users.shape)
