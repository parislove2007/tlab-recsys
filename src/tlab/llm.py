"""Language-model proposers for the feature search.

The model never writes code. It emits programs in the DSL, which are validated
against the grammar before anything is compiled, so a hallucinated field is a
rejected proposal rather than a silent bug or a leak.

Two backends share one prompt builder:
  * `OllamaProposer` talks to a local model and runs unattended.
  * `TranscriptProposer` replays recorded responses, which is how a run driven
    by a large hosted model stays reproducible for someone re-running the repo.
"""
import json
import pathlib
import re
import time
import urllib.request

from .dsl import (ANCHORS, VIA, EVENTS, WINDOWS, AGGS, NORMS, POOLS,
                  canon, name, validate, InvalidProgram)

GRAMMAR = f"""A feature is a JSON object in this grammar.

agg program:
{{"kind":"agg","anchor":A,"via":V,"event":E,"window":W,"agg":G,"norm":N,"pool":P}}
  anchor A  {ANCHORS}
      user       - aggregate the user's own past events            -> one value per user
      item       - aggregate the candidate item's past events      -> one value per item
      attr       - the user's past events on items that share an attribute value
                   with the candidate item (the indirect user-item link)
      attr_item  - everyone's past events on items sharing the attribute value
                   (how hot that genre / decade / director is right now)
  via V     {VIA}            (required for attr and attr_item, else null)
  event E   {EVENTS}  (nested funnel: start = opened, watch = >=30% seen,
                               finish = >=90% seen)
  window W  {WINDOWS}   (days back from the prediction date; null = all history)
  agg G     {AGGS}
      recency  = days since the most recent matching event
      mean_age = mean age in days of matching events
      trend    = count in the recent half of the window / count in the older half
  norm N    {NORMS}
      per_day        divide by the window length
      share_user     divide by the user's total event count in the same window
      share_item     divide by the item's total event count in the same window
      z_in_group     standardise within the candidate list of one user
      rank_in_group  relative rank within the candidate list of one user
  pool P    {POOLS}    (required for attr and attr_item: an item has several
                               attribute values, this pools over them)

compound programs:
{{"kind":"ratio","a":<agg>,"b":<agg>,"smooth":1.0}}   a / (b + 1)
{{"kind":"mul","a":<agg>,"b":<agg>}}                  a * b

Rules the validator enforces: nuniq_user is rejected for anchor user/attr,
nuniq_item for anchor item; per_day, share_user and share_item only apply to
counting aggregates (count, nuniq_item, nuniq_user, nuniq_day, sum_dur);
share_user needs anchor user or attr, share_item needs item or attr_item."""

DOMAIN = """Domain: KION, a Russian film and series streaming service. One row
per (user, item) in the log, so a user never watches the same item twice and
already-watched items are removed from the candidate list. Direct user-item
history is therefore useless here and all personalisation has to come through
indirect links. Task: for a prediction date T, rank ~100 candidate items for
each user by whether they will be watched in the following seven days.
Candidates come from popularity, ALS, item-item co-visitation and
genre-popularity generators; their ranks and scores are already in the model,
as are user demographics and item metadata."""

TASK = """Propose {n} NEW feature programs that are likely to add signal on top
of what is already in the model. Spend the budget well:
 - vary the time window deliberately; the point of the search is to find which
   horizon each relation actually wants
 - prefer relations and normalisations that are not yet represented
 - a few deliberate long shots are fine, but most proposals should be plausible
Return ONLY a JSON array of {n} objects, each the program itself plus one extra
key "why" with at most 12 words. No markdown fences, no prose outside the JSON."""


def _fmt_state(st, top=18, worst=8):
    if not st.gains:
        return "Nothing has been tried yet."
    items = sorted(st.gains.items(), key=lambda kv: -kv[1])
    lines = ["Already in the model (accepted): "
             + (", ".join(name(p) for p in st.accepted) or "none")]
    lines.append("\nHighest screening gain so far (gain = LightGBM split gain):")
    for k, v in items[:top]:
        lines.append(f"  {v:10.1f}  {canon(st.tried[k])}")
    dead = [kv for kv in items if kv[1] <= 0.0][:worst]
    if dead:
        lines.append("\nScored exactly zero gain (do not propose these shapes again):")
        for k, v in dead:
            lines.append(f"  {canon(st.tried[k])}")
    if st.val_trace:
        t = st.val_trace[-1]
        lines.append(f"\nValidation NDCG@10 is {t['val_ndcg']:.4f} with "
                     f"{t['n_features']} searched features after {t['programs']} proposals.")
    if st.rejected_blocks:
        lines.append(f"{st.rejected_blocks} block(s) were rejected for not improving validation.")
    return "\n".join(lines)


def build_prompt(st, n):
    return "\n\n".join([GRAMMAR, DOMAIN, _fmt_state(st), TASK.format(n=n)])


NULLISH = {"none", "null", "nan", "", "na"}


def _clean(p):
    """Deserialisation hygiene only: a model writing the string "None" where
    JSON wants null is a formatting slip, not a grammar error."""
    for f in ("via", "norm", "pool", "window"):
        v = p.get(f)
        if isinstance(v, str) and v.strip().lower() in NULLISH:
            p[f] = None
    w = p.get("window")
    if isinstance(w, str):
        try:
            p["window"] = int(w)
        except ValueError:
            p["window"] = None
    return p


def parse_programs(text, errors=None):
    """Pull a JSON array of programs out of a model response, tolerantly.

    Grammar violations are rejected, not repaired; when `errors` is given the
    reasons are collected so the caller can show them to the model once."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.MULTILINE).strip()
    start = t.find("[")
    if start < 0:
        return []
    depth, end = 0, None
    for i, ch in enumerate(t[start:], start):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        t = t[start:] + "]" * depth
    else:
        t = t[start:end]
    try:
        raw = json.loads(t)
    except json.JSONDecodeError:
        raw = []
        for m in re.finditer(r"\{(?:[^{}]|\{[^{}]*\})*\}", t):
            try:
                raw.append(json.loads(m.group(0)))
            except json.JSONDecodeError:
                continue
    out = []
    for p in raw:
        if not isinstance(p, dict):
            continue
        p.pop("why", None)
        p.setdefault("kind", "agg")
        if p["kind"] == "agg":
            for f in ("via", "norm", "pool", "window"):
                p.setdefault(f, None)
            _clean(p)
        for side in ("a", "b", "inner"):
            if isinstance(p.get(side), dict):
                p[side].pop("why", None)
                p[side].setdefault("kind", "agg")
                for f in ("via", "norm", "pool", "window"):
                    p[side].setdefault(f, None)
                _clean(p[side])
        try:
            out.append(validate(p))
        except InvalidProgram as e:
            if errors is not None:
                errors.append(f"{json.dumps(p, sort_keys=True)} -> {e}")
    return out


class OllamaProposer:
    def __init__(self, model="qwen2.5-coder:7b-instruct", host="http://127.0.0.1:11435",
                 batch=24, log_dir=None, temperature=0.8, seed=0):
        self.model, self.host, self.batch = model, host, batch
        self.temperature, self.seed = temperature, seed
        self.log_dir = pathlib.Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self.queue = []
        self.calls = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.seconds = 0.0
        self.gen = 0
        self.proposed = 0
        self.rejected = 0
        self.repairs = 0

    def _generate(self, prompt):
        body = json.dumps({
            "model": self.model, "prompt": prompt, "stream": False,
            "options": {"temperature": self.temperature, "num_predict": 2600,
                        "num_ctx": 8192, "seed": self.seed + self.calls},
        }).encode()
        req = urllib.request.Request(self.host + "/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=1200) as r:
            out = json.loads(r.read())
        self.seconds += time.time() - t0
        self.calls += 1
        self.tokens_in += out.get("prompt_eval_count", 0)
        self.tokens_out += out.get("eval_count", 0)
        return out.get("response", "")

    def new_generation(self, st):
        self.gen += 1
        prompt = build_prompt(st, self.batch)
        errs = []
        text = self._generate(prompt)
        progs = parse_programs(text, errs)
        self.proposed += self.batch
        self.rejected += len(errs)
        # one repair round when most of the batch broke the grammar: the model
        # is told exactly what the validator said and asked again
        if len(progs) < self.batch // 2 and errs:
            self.repairs += 1
            fix = (prompt + "\n\nYour previous answer was rejected by the validator. "
                   "Here are the first failures, verbatim:\n"
                   + "\n".join(f"  {e}" for e in errs[:10])
                   + "\n\nFix these mistakes and return a fresh JSON array of "
                   f"{self.batch} valid programs. Remember: anchor user and item take "
                   "via = null and pool = null; use JSON null, not the string \"None\".")
            errs2 = []
            text2 = self._generate(fix)
            more = parse_programs(text2, errs2)
            self.rejected += len(errs2)
            seen = {canon(x) for x in progs}
            progs = progs + [x for x in more if canon(x) not in seen]
            text = text + "\n\n=== repair round ===\n" + fix + "\n\n=== response ===\n" + text2
        if self.log_dir:
            (self.log_dir / f"gen{self.gen:02d}_prompt.txt").write_text(prompt)
            (self.log_dir / f"gen{self.gen:02d}_response.txt").write_text(text)
            (self.log_dir / f"gen{self.gen:02d}_parsed.json").write_text(
                json.dumps(progs, indent=1))
        self.queue = list(progs)
        self.parsed_last = len(progs)

    def __call__(self, st, rng):
        if not self.queue:
            raise InvalidProgram("batch exhausted for this generation")
        return self.queue.pop(0)

    def cost(self):
        return {"calls": self.calls, "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out, "seconds": round(self.seconds, 1),
                "proposed": self.proposed, "rejected_by_validator": self.rejected,
                "repair_rounds": self.repairs}


class TranscriptProposer:
    """Replays recorded model responses; in record mode it writes the prompt it
    would have sent and stops, so a hosted model can be driven by hand and the
    whole run still replays deterministically afterwards."""

    def __init__(self, log_dir, batch=24, wait_seconds=0, poll=4.0):
        self.dir = pathlib.Path(log_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.batch = batch
        self.wait_seconds = wait_seconds
        self.poll = poll
        self.queue = []
        self.gen = 0
        self.calls = 0
        self.waited = 0.0
        self.proposed = 0
        self.rejected = 0

    def new_generation(self, st):
        self.gen += 1
        resp = self.dir / f"gen{self.gen:02d}_response.txt"
        prompt = build_prompt(st, self.batch)
        (self.dir / f"gen{self.gen:02d}_prompt.txt").write_text(prompt)
        if not resp.exists() and self.wait_seconds:
            # stay alive so the fold caches survive while the response is
            # produced out of band; a hand-driven run would otherwise cost
            # O(generations^2) recomputation
            print(f"  waiting for {resp}", flush=True)
            t0 = time.time()
            while not resp.exists() and time.time() - t0 < self.wait_seconds:
                time.sleep(self.poll)
            self.waited += time.time() - t0
        if not resp.exists():
            raise StopIteration(
                f"no recorded response for generation {self.gen}; prompt written to {resp.parent}")
        self.calls += 1
        errs = []
        progs = parse_programs(resp.read_text(), errs)
        self.proposed += self.batch
        self.rejected += len(errs)
        (self.dir / f"gen{self.gen:02d}_parsed.json").write_text(json.dumps(progs, indent=1))
        self.queue = list(progs)

    def __call__(self, st, rng):
        if not self.queue:
            raise InvalidProgram("batch exhausted for this generation")
        return self.queue.pop(0)

    def cost(self):
        return {"calls": self.calls, "waited_seconds": round(self.waited, 1),
                "proposed": self.proposed, "rejected_by_validator": self.rejected}
