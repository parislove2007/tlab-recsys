"""Run one feature-search strategy under the shared evaluation protocol.

Every strategy gets the same budget (generations x block newly compiled
programs), the same proxy ranker, the same folds and the same candidate lists.
Only the proposal mechanism differs. The test folds are not read here.
"""
import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from tlab import dataset as D, space, search, llm
from tlab.config import RUNS, SCALES
from tlab.human import human_programs
from tlab.pipeline import Bundle

ap = argparse.ArgumentParser()
ap.add_argument("--strategy", required=True,
                choices=["random", "evolution", "llm_local", "llm_recorded", "human_pool"])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--generations", type=int, default=15)
ap.add_argument("--block", type=int, default=32)
ap.add_argument("--keep", type=int, default=10)
ap.add_argument("--scale", default="search")
ap.add_argument("--base", default="none", choices=["none", "human"])
ap.add_argument("--model", default="qwen2.5-coder:7b-instruct")
ap.add_argument("--transcripts", default=None)
ap.add_argument("--wait", type=int, default=0,
                help="seconds to wait for each recorded response instead of exiting")
ap.add_argument("--tag", default=None)
a = ap.parse_args()

base_progs = human_programs() if a.base == "human" else []
users = D.sample_users(SCALES[a.scale])
print(f"{a.strategy} seed={a.seed} scale={a.scale}({len(users)} users) base={a.base}", flush=True)
bundle = Bundle(users, verbose=False)
ev = search.Evaluator(bundle, base_progs=base_progs)

proposer_cost = {}
if a.strategy == "human_pool":
    # fairness control: the hand-written programs go through exactly the same
    # accept-or-reject loop, so the searched sets' advantage cannot be explained
    # by the search having been allowed to tune on the validation fold
    proposer = search.pool_proposer(human_programs())
elif a.strategy == "random":
    proposer = search.random_proposer(space)
elif a.strategy == "evolution":
    proposer = search.evolution_proposer(space)
elif a.strategy == "llm_local":
    d = a.transcripts or (RUNS / f"llm_local_{a.base}_s{a.seed}")
    proposer = llm.OllamaProposer(model=a.model, batch=a.block, log_dir=d, seed=a.seed * 1000)
else:
    d = a.transcripts or (RUNS / f"llm_recorded_{a.base}_s{a.seed}")
    proposer = llm.TranscriptProposer(d, batch=a.block, wait_seconds=a.wait)

t0 = time.time()
try:
    st, hist = search.run_search(proposer, ev, generations=a.generations, block=a.block,
                                 keep=a.keep, log=True, seed=a.seed)
except StopIteration as e:
    print(f"PAUSED: {e}")
    sys.exit(2)
wall = round(time.time() - t0, 1)

if hasattr(proposer, "cost"):
    proposer_cost = proposer.cost()

tag = a.tag or f"{a.strategy}_{a.base}_s{a.seed}"
out = {
    "strategy": a.strategy, "seed": a.seed, "base": a.base, "scale": a.scale,
    "n_users": int(len(users)), "generations": a.generations, "block": a.block,
    "keep": a.keep, "wall_seconds": wall,
    "evaluator_cost": ev.cost(), "proposer_cost": proposer_cost,
    "val_trace": st.val_trace,
    "accepted": st.accepted,
    "tried": list(st.tried.values()),
    "gains": {k: v for k, v in st.gains.items()},
    "history": hist,
}
(RUNS / f"search_{tag}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(f"saved runs/search_{tag}.json  best_val={st.val_trace[-1]['val_ndcg']:.4f} "
      f"|F|={len(st.accepted)} wall={wall}s cost={ev.cost()} llm={proposer_cost}", flush=True)
