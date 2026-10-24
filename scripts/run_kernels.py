"""Fit the decay rate of each relation instead of choosing a window."""
import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import polars as pl
import torch
from tlab import dataset as D, kernels, rank
from tlab.config import RUNS, SCALES, TRAIN_DATES, VAL_DATES, TEST_DATES
from tlab.human import human_programs
from tlab.metrics import ranking_metrics
from tlab.pipeline import Bundle

ap = argparse.ArgumentParser()
ap.add_argument("--scale", default="search")
ap.add_argument("--epochs", type=int, default=120)
ap.add_argument("--lr-lam", type=float, default=0.05)
ap.add_argument("--lr-head", type=float, default=0.01)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

torch.manual_seed(a.seed)
torch.set_num_threads(8)
users = D.sample_users(SCALES[a.scale])
b = Bundle(users, verbose=False)
print(f"scale={a.scale}: {len(users)} users", flush=True)

t0 = time.time()
tens = {}
for T in b.dates:
    tens[T] = kernels.FoldTensors(b.folds[T]["engine"], b.folds[T]["cand"])
    print(f"  tensors {T} rows={tens[T].n_rows} groups={tens[T].n_groups}", flush=True)
print(f"tensor build {time.time()-t0:.0f}s", flush=True)

model = kernels.KernelRanker(n_rel=len(kernels.RELATIONS))
opt = torch.optim.Adam([
    {"params": [model.raw], "lr": a.lr_lam},
    {"params": model.head.parameters(), "lr": a.lr_head},
])


def ndcg_on(dates):
    model.eval()
    vals = []
    with torch.no_grad():
        for T in dates:
            s = model(tens[T]).numpy()
            df = b.folds[T]["cand"].select("user_id", "label").with_columns(
                pl.Series("score", s.astype(np.float64)))
            vals.append(ranking_metrics(df, pos_table=b.folds[T]["pos"])[0])
    return {k: float(np.mean([v[k] for v in vals])) for k in vals[0] if isinstance(vals[0][k], float)}


hist, best, best_state, patience = [], -1, None, 0
for ep in range(1, a.epochs + 1):
    model.train()
    tot = 0.0
    for T in TRAIN_DATES:
        t = tens[T]
        opt.zero_grad()
        loss = kernels.listwise_loss(model(t), t.y, t.gid, t.n_groups)
        loss.backward()
        opt.step()
        tot += float(loss)
    if ep % 5 == 0 or ep == 1:
        v = ndcg_on(VAL_DATES)["ndcg@10"]
        hl = model.half_lives()
        hist.append({"epoch": ep, "loss": tot / len(TRAIN_DATES), "val_ndcg": v,
                     "half_lives": hl.tolist()})
        print(f"  ep{ep:3d} loss={tot/len(TRAIN_DATES):.4f} val_ndcg={v:.4f}", flush=True)
        if v > best:
            best, best_state, patience = v, {k: x.clone() for k, x in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 6:
                break

model.load_state_dict(best_state)
hl = model.half_lives()
learned = {name: {"half_life_days": [round(float(x), 2) for x in hl[i]]}
           for i, (name, *_rest) in enumerate(kernels.RELATIONS)}
print("\nlearned half-lives (days):")
for k, v in learned.items():
    print(f"  {k:12} {v['half_life_days']}")

standalone = {"val": ndcg_on(VAL_DATES), "test": ndcg_on(TEST_DATES)}
print(f"\nkernel model alone: val={standalone['val']['ndcg@10']:.4f} "
      f"test={standalone['test']['ndcg@10']:.4f}", flush=True)

# hand the learned kernel features to the same gradient boosting ranker
kcols = {}
with torch.no_grad():
    for T in b.dates:
        F = model.features(tens[T]).numpy()
        names = [f"kern_{n}_{m}" for n, *_ in kernels.RELATIONS for m in range(F.shape[1] // len(kernels.RELATIONS))]
        kcols[T] = pl.DataFrame({nm: F[:, i].astype(np.float32) for i, nm in enumerate(names)})

results = {}
for label, progs, with_kern in [("BASE", [], False), ("BASE+KERNEL", [], True),
                                ("BASE+HUMAN", human_programs(), False),
                                ("BASE+HUMAN+KERNEL", human_programs(), True)]:
    parts = b.matrix(progs)
    if with_kern:
        parts = {s: [(T, pl.concat([X, kcols[T]], how="horizontal"), c) for T, X, c in v]
                 for s, v in parts.items()}
    m, ttrain, _ = rank.train_rank(parts)
    val, _, _ = rank.evaluate(m, b, parts, "val")
    test, _, pu = rank.evaluate(m, b, parts, "test")
    results[label] = {"n_features": len(m.feature_name()), "val": val, "test": test,
                      "train_seconds": ttrain}
    pu.write_parquet(RUNS / f"peruser_kern_{label.replace('+','_')}_{a.scale}.parquet")
    print(f"  {label:20} nf={len(m.feature_name()):3d} val={val['ndcg@10']:.4f} test={test['ndcg@10']:.4f}", flush=True)

(RUNS / f"kernels_{a.scale}.json").write_text(json.dumps({
    "scale": a.scale, "n_users": int(len(users)), "relations": [r[0] for r in kernels.RELATIONS],
    "learned_half_lives": learned, "training": hist, "standalone": standalone,
    "gbdt": results}, indent=1, ensure_ascii=False))
print("saved", RUNS / f"kernels_{a.scale}.json")
