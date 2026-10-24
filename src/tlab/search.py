"""Feature search over the DSL, with one evaluation protocol shared by every
strategy so that a comparison between them is a comparison of proposal quality
and nothing else.

The budget unit is one newly proposed, compiled and screened program. Each
generation costs two proxy trainings regardless of strategy: one to read the
gain of the new block, one to decide whether to keep it. The test folds are
never touched here; selection uses the validation fold only.
"""
import gc
import json
import random
import time

import numpy as np
import polars as pl

from . import rank
from .config import TOPK
from .dsl import canon, name, validate, InvalidProgram

PROXY = {
    "learning_rate": 0.12,
    "num_leaves": 31,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.75,
}


class Evaluator:
    def __init__(self, bundle, base_progs=(), num_boost_round=300, early_stopping=30):
        self.b = bundle
        self.base_progs = list(base_progs)
        self.rounds = num_boost_round
        self.es = early_stopping
        self.n_trainings = 0
        self.n_programs = 0
        self.t_compile = 0.0
        self.t_train = 0.0

    def _fit(self, progs):
        t0 = time.time()
        parts = self.b.matrix(self.base_progs + list(progs))
        self.t_compile += time.time() - t0
        t0 = time.time()
        model, _, _ = rank.train_rank(parts, params=PROXY,
                                      num_boost_round=self.rounds,
                                      early_stopping=self.es)
        self.t_train += time.time() - t0
        self.n_trainings += 1
        gc.collect()
        return model, parts

    def val_ndcg(self, progs):
        model, parts = self._fit(progs)
        m, _, _ = rank.evaluate(model, self.b, parts, "val")
        return m["ndcg@10"], model

    def screen(self, current, new):
        """Gain importance of each newly added program, in the presence of the
        current set. One training for the whole block."""
        self.n_programs += len(new)
        model, parts = self._fit(list(current) + list(new))
        imp = dict(zip(model.feature_name(),
                       model.feature_importance("gain").astype(float)))
        out = {canon(p): imp.get(name(p), 0.0) for p in new}
        del model, parts
        gc.collect()
        return out

    def cost(self):
        return {"trainings": self.n_trainings, "programs": self.n_programs,
                "compile_s": round(self.t_compile, 1), "train_s": round(self.t_train, 1)}


class SearchState:
    """What a proposer is allowed to see: the accepted set, every program tried
    with its screening gain, and the validation trace."""

    def __init__(self):
        self.accepted = []
        self.tried = {}
        self.gains = {}
        self.val_trace = []
        self.rejected_blocks = 0

    def seen(self, p):
        return canon(p) in self.tried


def run_search(proposer, ev, generations=20, block=24, keep=8, log=None, seed=0):
    rng = random.Random(seed)
    st = SearchState()
    best, _ = ev.val_ndcg([])
    st.val_trace.append({"gen": 0, "programs": 0, "val_ndcg": best,
                         "n_features": 0, "trainings": ev.n_trainings})
    history = []

    for gen in range(1, generations + 1):
        # batching proposers need to know a generation started, otherwise an
        # under-full batch silently shifts them out of step with the loop
        if hasattr(proposer, "new_generation"):
            proposer.new_generation(st)
        want, tries = [], 0
        while len(want) < block and tries < block * 25:
            tries += 1
            try:
                p = validate(proposer(st, rng))
            except (InvalidProgram, KeyError, TypeError):
                continue
            if st.seen(p) or any(canon(p) == canon(q) for q in want):
                continue
            want.append(p)
        if not want:
            # a generation that produced nothing parseable costs an LLM call but
            # no budget; record it and move on rather than ending the run
            st.val_trace.append({"gen": gen, "programs": ev.n_programs, "val_ndcg": best,
                                 "n_features": len(st.accepted), "trainings": ev.n_trainings,
                                 "accepted_now": 0, "empty_generation": True})
            if log:
                print(f"  gen {gen:2d}: nothing parseable, skipped", flush=True)
            continue

        gains = ev.screen(st.accepted, want)
        for p in want:
            st.tried[canon(p)] = p
            st.gains[canon(p)] = gains[canon(p)]

        ranked = sorted(want, key=lambda p: -gains[canon(p)])[:keep]
        trial = st.accepted + ranked
        score, _ = ev.val_ndcg(trial)
        if score > best + 1e-6:
            best = score
            st.accepted = trial
            kept = [name(p) for p in ranked]
        else:
            st.rejected_blocks += 1
            kept = []

        st.val_trace.append({"gen": gen, "programs": ev.n_programs, "val_ndcg": best,
                             "n_features": len(st.accepted), "trainings": ev.n_trainings,
                             "accepted_now": len(kept)})
        history.append({"gen": gen, "proposed": [name(p) for p in want],
                        "gains": {name(p): gains[canon(p)] for p in want},
                        "kept": kept, "val_ndcg": score, "best": best})
        if log:
            print(f"  gen {gen:2d}: tried {ev.n_programs:4d} progs, "
                  f"|F|={len(st.accepted):3d}, val={score:.4f} best={best:.4f}"
                  f" {'(kept)' if kept else '(rejected)'}", flush=True)
    return st, history


def random_proposer(space_mod):
    def f(st, rng):
        return space_mod.random_program(rng)
    return f


def evolution_proposer(space_mod, elite=12, p_cross=0.3, p_random=0.15):
    """Mutate and recombine the programs with the highest screening gain."""
    def f(st, rng):
        if not st.gains or rng.random() < p_random:
            return space_mod.random_program(rng)
        top = sorted(st.gains.items(), key=lambda kv: -kv[1])[:elite]
        top = [st.tried[k] for k, _ in top if st.gains[k] > 0]
        if not top:
            return space_mod.random_program(rng)
        if len(top) > 1 and rng.random() < p_cross:
            a, b = rng.sample(top, 2)
            return space_mod.crossover(a, b, rng)
        return space_mod.mutate(rng.choice(top), rng)
    return f


def pool_proposer(pool):
    """Draws from a fixed pool of programs instead of inventing them."""
    import itertools
    from .dsl import canon as _canon

    def f(st, rng):
        left = [p for p in pool if not st.seen(p)]
        if not left:
            raise InvalidProgram("pool exhausted")
        return rng.choice(left)
    return f
