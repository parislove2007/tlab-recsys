"""Sampling and mutation over the DSL program space."""
import itertools
import random

from .dsl import (ANCHORS, VIA, EVENTS, WINDOWS, AGGS, NORMS, POOLS,
                  InvalidProgram, canon, validate)


def enumerate_aggs():
    out = []
    for anchor in ANCHORS:
        vias = VIA if anchor in ("attr", "attr_item") else [None]
        pools = POOLS if anchor in ("attr", "attr_item") else [None]
        for via, ev, w, ag, nrm, pool in itertools.product(
            vias, EVENTS, WINDOWS, AGGS, NORMS, pools
        ):
            p = {"kind": "agg", "anchor": anchor, "via": via, "event": ev,
                 "window": w, "agg": ag, "norm": nrm, "pool": pool}
            try:
                validate(p)
            except InvalidProgram:
                continue
            out.append(p)
    return out


def space_size():
    n = len(enumerate_aggs())
    return {"agg": n, "ratio": n * (n - 1), "mul": n * (n - 1) // 2,
            "total": n + n * (n - 1) + n * (n - 1) // 2}


def random_agg(rng):
    while True:
        anchor = rng.choice(ANCHORS)
        is_attr = anchor in ("attr", "attr_item")
        p = {
            "kind": "agg", "anchor": anchor,
            "via": rng.choice(VIA) if is_attr else None,
            "event": rng.choice(EVENTS),
            "window": rng.choice(WINDOWS),
            "agg": rng.choice(AGGS),
            "norm": rng.choice(NORMS),
            "pool": rng.choice(POOLS) if is_attr else None,
        }
        try:
            return validate(p)
        except InvalidProgram:
            continue


def random_program(rng, p_compound=0.25):
    r = rng.random()
    if r < p_compound / 2:
        a, b = random_agg(rng), random_agg(rng)
        if canon(a) != canon(b):
            try:
                return validate({"kind": "ratio", "a": a, "b": b, "smooth": 1.0})
            except InvalidProgram:
                pass
    elif r < p_compound:
        a, b = random_agg(rng), random_agg(rng)
        if canon(a) != canon(b):
            try:
                return validate({"kind": "mul", "a": a, "b": b})
            except InvalidProgram:
                pass
    return random_agg(rng)


_FIELDS = ["anchor", "via", "event", "window", "agg", "norm", "pool"]
_CHOICES = {"anchor": ANCHORS, "via": VIA, "event": EVENTS, "window": WINDOWS,
            "agg": AGGS, "norm": NORMS, "pool": POOLS}


def mutate_agg(p, rng, tries=40):
    for _ in range(tries):
        q = dict(p)
        f = rng.choice(_FIELDS)
        q[f] = rng.choice(_CHOICES[f])
        if q["anchor"] in ("attr", "attr_item"):
            q["via"] = q["via"] or rng.choice(VIA)
            q["pool"] = q["pool"] or rng.choice(POOLS)
        else:
            q["via"], q["pool"] = None, None
        try:
            return validate(q)
        except InvalidProgram:
            continue
    return p


def mutate(p, rng):
    if p["kind"] == "agg":
        if rng.random() < 0.15:
            other = random_agg(rng)
            kind = "ratio" if rng.random() < 0.6 else "mul"
            cand = {"kind": kind, "a": p, "b": other}
            if kind == "ratio":
                cand["smooth"] = 1.0
            try:
                return validate(cand)
            except InvalidProgram:
                return mutate_agg(p, rng)
        return mutate_agg(p, rng)
    q = dict(p)
    side = "a" if rng.random() < 0.5 else "b"
    q[side] = mutate_agg(q[side], rng)
    try:
        return validate(q)
    except InvalidProgram:
        return q[side]


def crossover(p, q, rng):
    if p["kind"] == "agg" and q["kind"] == "agg":
        child = dict(p)
        for f in _FIELDS:
            if rng.random() < 0.5:
                child[f] = q.get(f)
        if child["anchor"] in ("attr", "attr_item"):
            child["via"] = child["via"] or rng.choice(VIA)
            child["pool"] = child["pool"] or rng.choice(POOLS)
        else:
            child["via"], child["pool"] = None, None
        try:
            return validate(child)
        except InvalidProgram:
            return mutate_agg(p, rng)
    base = p if p["kind"] == "agg" else p["a"]
    other = q if q["kind"] == "agg" else q["a"]
    return crossover(base, other, rng)
