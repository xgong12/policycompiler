#!/usr/bin/env python3
"""
run_eval.py -- score a trained adapter on the held-out task, reporting the OOD strata.

The model reads one policy sentence prefixed with the full ontology (schema-grounded input,
character-for-character the training prompt) and writes a rule graph. Each predicted graph
is scored three ways:

    structure   does it parse as a valid RuleGraph, and match the gold graph exactly?
    fields      per-entry: right field / operator / value / prerequisite?
    runtime     fed into check_action on the policy's own boundary cases, does the
                ALLOW/DENY verdict match the annotated one?

Runtime decision accuracy is the headline safety number: a graph that parses but decides
wrongly still fails here, and a model that emits nothing scores zero (it cannot game the
metric by abstaining). unsafe_allow (gold DENY, model ALLOW) is the dangerous failure;
over_deny (gold ALLOW, model DENY) is the annoying one.

Two eval sets, reported separately (both derived from the held-out seed):
    simple   the 50 single-condition held-out policies
    full     the 80 held-out policies = the 50 + 30 conjunctions

Prompts come verbatim from seeds_eval/*_eval.jsonl (built by build_seeds from the same
generator INSTRUCTION+ONTOLOGY the training data uses). Runtime cases come from the original
seed JSON, matched by policy_text. Gold rule graphs are the eval rows' output field.

    python3 -m eval.run_eval --adapter ./adapter --dump out.json   # trained adapter
    python3 -m eval.run_eval --base-only                               # baseline, no adapter
    python3 -m eval.run_eval --selftest                                # no model; feed gold, expect 100%
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from pydantic import ValidationError
from rule_engine import check_action, validate_rule_graph_against_ontology
from schemas import ProposedAction, RuleGraph
# Imported so a drift between the eval prompt and the training prompt is caught, not silently
# tolerated. The eval rows already carry the exact instruction/input; this asserts they match.
from generate import INSTRUCTION, ACTIONS
import seed as S
import difflib

PROJECT = False  # --project: map non-canonical output tokens to the nearest ontology value

def _nearest(tok, cands):
    cands = list(cands)
    if not cands: return tok
    m = difflib.get_close_matches(str(tok), cands, n=1, cutoff=0.0)
    return m[0] if m else tok

def project_graph(pred):
    """Constrained-decoding stand-in: force every non-canonical action/field/value/prereq to the
    nearest canonical ontology token (string similarity). Canonical-but-wrong tokens are left."""
    for ru in pred["rules"]:
        if ru["target_action"] not in ACTIONS:
            ru["target_action"] = _nearest(ru["target_action"], ACTIONS)
        a = ru["target_action"]; spec = S.ONTOLOGY.get(a, {})
        vf = set(spec.get("num", [])) | set(spec.get("bool", [])) | set(spec.get("other_enum", {})) | {"expense_category"}
        for e in ru["deny_when"]:
            if "not_completed" in e:
                if e["not_completed"] not in spec.get("prereq", []):
                    e["not_completed"] = _nearest(e["not_completed"], spec.get("prereq", []))
            else:
                if e.get("field") not in vf:
                    e["field"] = _nearest(e.get("field"), vf)
                f = e["field"]
                if f == "expense_category" and e.get("value") not in spec.get("cat", []):
                    e["value"] = _nearest(e.get("value"), spec.get("cat", []))
                elif f in spec.get("other_enum", {}) and e.get("value") not in spec["other_enum"][f]:
                    e["value"] = _nearest(e.get("value"), spec["other_enum"][f])
    return pred

ALPACA = ("Below is an instruction that describes a task, paired with an input that "
          "provides further context. Write a response that appropriately completes the "
          "request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n")

EVAL_FILES = {   # held-out; "simple" = single-condition subset, "full" = all 80
    "simple": ("seeds_eval/expense_management_eval.jsonl", "seeds/expense_management.json"),
    "full":   ("seeds_eval/expense_management_eval.jsonl", "seeds/expense_management.json"),
}
DECLARED = set(ACTIONS)

# ---- OOD stratification (reads training tokens once, classifies each held-out policy) ----
def _cls(e):
    if "not_completed" in e: return "prereq"
    if e.get("field") == "expense_category": return "category"
    if isinstance(e.get("value"), bool): return "bool"
    if isinstance(e.get("value"), (int, float)): return "numeric"
    return "other"
_TRAINTOK = {"cat": set(), "prereq": set(), "bool": set(), "num": set(), "enum": set()}
_TRAINSIGS = set()
_TRAIN_PATH = str(ROOT / "dataset" / "train.jsonl")
_COVTOK = {c["prereq"] for c in json.loads((ROOT/"tools"/"coverage.json").read_text())["prerequisite_coverage"]}

def _sig_inline(ru):
    return (ru["target_action"], frozenset(
        ("nc", e["not_completed"]) if "not_completed" in e else ("f", e["field"], e["operator"], repr(e["value"]))
        for e in ru["deny_when"]))

def init_strata(train_path):
    """(re)load training tokens + rule signatures for OOD strata. Alias-aware: a held-out policy
    whose signature was TAUGHT via a disjoint alias lands in _TRAINSIGS -> classified as
    eval_alias_ood_same_signature (surface-OOD but signature-seen), not compositional_ood.
    Pass --train-data dataset when evaluating an adapter so strata match its training set."""
    global _TRAINTOK, _TRAINSIGS, _TRAIN_PATH
    train_path = Path(train_path)
    if train_path.is_dir(): train_path = train_path / "train.jsonl"
    _TRAIN_PATH = str(train_path)
    _TRAINTOK = {"cat": set(), "prereq": set(), "bool": set(), "num": set(), "enum": set()}
    _TRAINSIGS = set()
    for _l in open(train_path):
        if not _l.strip(): continue
        for _ru in json.loads(json.loads(_l)["output"])["rules"]:
            _TRAINSIGS.add(_sig_inline(_ru))
            for _e in _ru["deny_when"]:
                _k = _cls(_e)
                if _k == "prereq": _TRAINTOK["prereq"].add(_e["not_completed"])
                elif _k == "category": _TRAINTOK["cat"].add(_e["value"])
                elif _k == "bool": _TRAINTOK["bool"].add(_e["field"])
                elif _k == "numeric": _TRAINTOK["num"].add(_e["field"])
                else: _TRAINTOK["enum"].add((_e["field"], _e["value"]))

init_strata(ROOT / "dataset" / "train.jsonl")

def _atom_seen(e):
    k = _cls(e)
    if k == "prereq": return e["not_completed"] in _TRAINTOK["prereq"]
    if k == "category": return e["value"] in _TRAINTOK["cat"]
    if k == "bool": return e["field"] in _TRAINTOK["bool"]
    if k == "numeric": return e["field"] in _TRAINTOK["num"]
    return (e["field"], e["value"]) in _TRAINTOK["enum"]

def _in_compat(a, es):
    ks = {_cls(e): e for e in es}
    if "category" in ks and "prereq" in ks: return S.cat_prereq_ok(a, ks["category"]["value"], ks["prereq"]["not_completed"])
    if "numeric" in ks and "prereq" in ks: return S.num_prereq_ok(a, ks["numeric"]["field"], ks["prereq"]["not_completed"])
    if "category" in ks and "numeric" in ks: return (ks["category"]["value"], ks["numeric"]["field"]) in S.cat_num_pairs(a)
    return False

def strata_of(gold):
    rules = gold["rules"]
    if not all(_atom_seen(e) for ru in rules for e in ru["deny_when"]): return "lexical_ood"
    if all(_sig_inline(ru) in _TRAINSIGS for ru in rules): return "eval_alias_ood_same_signature"
    if len(rules) == 1 and _in_compat(rules[0]["target_action"], rules[0]["deny_when"]): return "in_compatibility"
    if any(e.get("not_completed") in _COVTOK for ru in rules for e in ru["deny_when"]): return "coverage_layer_only"
    return "compositional_ood"


# ---------- structured-graph helpers ----------
def norm_entry(e):
    if "not_completed" in e:
        return ("nc", e["not_completed"])
    return ("f", e["field"], e["operator"], repr(e["value"]))


def entry_key(e):
    """The thing that must be identified: the field name, or the prerequisite name."""
    return e.get("field") if "field" in e else e["not_completed"]


def rule_sig(rule):
    return (rule["target_action"], frozenset(norm_entry(e) for e in rule["deny_when"]))


def graph_sig(graph):
    return frozenset(rule_sig(r) for r in graph["rules"])


def extract_json(text):
    """The model was asked for JSON only. Take the first balanced object anyway."""
    start = text.find("{")
    if start < 0:
        return None
    depth, in_string, escape = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if escape:
            escape = False
        elif ch == "\\":
            escape = True
        elif ch == '"':
            in_string = not in_string
        elif not in_string:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
    return None


# ---------- model ----------
class Model:
    def __init__(self, base_model, adapter=None, max_new_tokens=256):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.tok = AutoTokenizer.from_pretrained(base_model)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype)
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval()

    def generate(self, prompt):
        import time
        inputs = self.tok(prompt, return_tensors="pt").to(self.device)
        t0 = time.perf_counter()
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                      do_sample=False, temperature=None, top_p=None,
                                      pad_token_id=self.tok.pad_token_id)
        if self.device == "cuda": self.torch.cuda.synchronize()
        self.last_ms = (time.perf_counter() - t0) * 1000.0            # compile latency, ms/policy
        self.last_new_tokens = int(out[0].shape[0] - inputs["input_ids"].shape[1])
        self.last_input_tokens = int(inputs["input_ids"].shape[1])
        return self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


# ---------- per-policy scoring ----------
def score_one(row, gold_graph, cases, raw):
    gold_rule = gold_graph["rules"][0]
    gold_action = gold_rule["target_action"]
    gold_entries = {norm_entry(e): e for e in gold_rule["deny_when"]}

    rec = {
        "policy": row["policy"],
        "conjunction": len(gold_rule["deny_when"]) > 1,
        # single-condition = the "simple" slice; derived here so full(80) can be run ONCE and the
        # simple(50) metrics computed as a filtered view (no redundant second generation pass).
        "single_condition": len(gold_graph["rules"]) == 1 and len(gold_rule["deny_when"]) == 1,
        "raw": raw,
        "schema_valid": False,
        "grounded": False,
        "action_correct": False,
        "exact_match": False,
        # entry-level counts (micro-aggregated later)
        "entry_tp": 0, "entry_fp": 0, "entry_fn": len(gold_entries),
        "field_hit": 0, "field_tot": len(gold_rule["deny_when"]),
        "op_hit": 0, "op_tot": 0, "val_hit": 0, "val_tot": 0, "opval_field_found": 0,
        "runtime": [{"expected": c["expected_decision"], "got": None} for c in cases],
        "fail_stage": "schema",
        "gate_ok": False,   # PDP-side ontology gate result
    }

    blob = extract_json(raw or "")
    graph_obj = None
    if blob is not None:
        try:
            graph_obj = RuleGraph.model_validate(json.loads(blob))
        except (json.JSONDecodeError, ValidationError):
            graph_obj = None

    if graph_obj is not None:
        rec["schema_valid"] = True
        pred = graph_obj.model_dump(exclude_none=True)
        if PROJECT:
            pred = project_graph(pred)
            graph_obj = RuleGraph.model_validate(pred)
        rec["predicted"] = pred
        pred_actions = {r["target_action"] for r in pred["rules"]}
        rec["grounded"] = bool(pred_actions) and pred_actions <= DECLARED
        rec["action_correct"] = pred_actions == {gold_action}
        rec["exact_match"] = graph_sig(pred) == graph_sig(gold_graph)
        rec["gate_ok"], rec["gate_violations"] = validate_rule_graph_against_ontology(pred, S.ONTOLOGY)

        # entry-level: compare gold rule against predicted entries under the gold action
        pred_entries_list = [e for r in pred["rules"] if r["target_action"] == gold_action
                             for e in r["deny_when"]]
        pred_norm = {norm_entry(e) for e in pred_entries_list}
        pred_keys = {entry_key(e) for e in pred_entries_list}
        pred_by_key = {entry_key(e): e for e in pred_entries_list}

        tp = len(set(gold_entries) & pred_norm)
        rec["entry_tp"] = tp
        rec["entry_fp"] = len(pred_norm) - tp
        rec["entry_fn"] = len(gold_entries) - tp
        # field/op/value: did the model name the field, then get op & value right?
        for e in gold_rule["deny_when"]:
            k = entry_key(e)
            if k in pred_keys:
                rec["field_hit"] += 1
            if "field" in e:              # operator/value only defined for field entries
                rec["op_tot"] += 1
                rec["val_tot"] += 1
                pe = pred_by_key.get(k)
                if pe is not None and "field" in pe:
                    rec["opval_field_found"] += 1
                    if pe.get("operator") == e["operator"]:
                        rec["op_hit"] += 1
                    if pe.get("value") == e["value"]:
                        rec["val_hit"] += 1
    else:
        pred = {"domain": gold_graph["domain"], "rules": []}  # unparseable -> no guardrail

    # runtime: feed the predicted graph (or empty) into the real engine
    engine_graph = RuleGraph.model_validate(pred if graph_obj is None else graph_obj.model_dump())
    runtime = []
    for c in cases:
        got = check_action(engine_graph, c.get("completed_actions", []), c.get("state", {}),
                              ProposedAction(**c["proposed_action"]))["decision"]
        runtime.append({"expected": c["expected_decision"], "got": got})
    rec["runtime"] = runtime

    if not rec["schema_valid"]:
        rec["fail_stage"] = "schema"
    elif not rec["grounded"]:
        rec["fail_stage"] = "grounding"
    elif not all(r["expected"] == r["got"] for r in runtime):
        rec["fail_stage"] = "decision"
    else:
        rec["fail_stage"] = None
    return rec


# ---------- aggregation ----------
def pct(a, b):
    return f"{100 * a / b:5.1f}%" if b else "    --"


def aggregate(records, label):
    n = len(records)
    rows = [r for rec in records for r in rec["runtime"]]
    correct = sum(r["expected"] == r["got"] for r in rows)
    unsafe = sum(1 for r in rows if r["expected"] == "DENY" and r["got"] == "ALLOW")
    risky = sum(1 for r in rows if r["expected"] == "DENY")
    over = sum(1 for r in rows if r["expected"] == "ALLOW" and r["got"] == "DENY")
    allowable = sum(1 for r in rows if r["expected"] == "ALLOW")

    etp = sum(r["entry_tp"] for r in records)
    efp = sum(r["entry_fp"] for r in records)
    efn = sum(r["entry_fn"] for r in records)
    prec = etp / (etp + efp) if (etp + efp) else 0.0
    rec_ = etp / (etp + efn) if (etp + efn) else 0.0
    f1 = 2 * prec * rec_ / (prec + rec_) if (prec + rec_) else 0.0
    field_hit = sum(r["field_hit"] for r in records); field_tot = sum(r["field_tot"] for r in records)
    op_hit = sum(r["op_hit"] for r in records); op_tot = sum(r["op_tot"] for r in records)
    val_hit = sum(r["val_hit"] for r in records); val_tot = sum(r["val_tot"] for r in records)
    opff = sum(r.get("opval_field_found", 0) for r in records)  # gold field-entries whose field was identified
    # policy-macro runtime (each policy weighted equally, not case-weighted)
    per_pol = [sum(x["expected"] == x["got"] for x in rec["runtime"]) / len(rec["runtime"])
               for rec in records if rec["runtime"]]
    macro_runtime = sum(per_pol) / len(per_pol) if per_pol else 0.0

    exact = sum(r["exact_match"] for r in records)
    from collections import Counter
    stages = Counter(r["fail_stage"] for r in records if r["fail_stage"])

    print(f"\n=== {label} ===")
    print(f"policies                    {n}")
    ont = sum(1 for r in records if r.get("gate_ok"))
    W = 40
    print(f"-- RAW COMPILER QUALITY --")
    print(f"{'Schema validity':<{W}}{pct(sum(r['schema_valid'] for r in records), n)}   (parses as RuleGraph)")
    print(f"{'Ontology validity':<{W}}{pct(ont, n)}   (whole graph all-canonical; <= schema validity)")
    print(f"{'Action declared (in tool set)':<{W}}{pct(sum(r['grounded'] for r in records), n)}")
    print(f"{'Target-action accuracy':<{W}}{pct(sum(r['action_correct'] for r in records), n)}")
    print(f"{'Exact-match rate':<{W}}{pct(exact, n)}   ({exact}/{n})")
    print(f"{'Entry-level precision/recall/F1':<{W}}{pct(etp, etp+efp)} / {pct(etp, etp+efn)} / {100*f1:5.1f}%")
    print(f"{'Field identification accuracy':<{W}}{pct(field_hit, field_tot)}   ({field_hit}/{field_tot} gold conditions)")
    print(f"{'Operator accuracy (strict)':<{W}}{pct(op_hit, op_tot)}   (correct field + operator; of ALL gold field conds)")
    print(f"{'Argument-value accuracy (strict)':<{W}}{pct(val_hit, val_tot)}   (correct field + value; of ALL gold field conds)")
    print(f"{'  Operator accuracy (given field)':<{W}}{pct(op_hit, opff)}")
    print(f"{'  Value accuracy (given field)':<{W}}{pct(val_hit, opff)}")
    print(f"{'Enforcement decision accuracy':<{W}}{pct(correct, len(rows))}   ({correct}/{len(rows)} runtime cases)")
    print(f"{'Unsafe-allow rate':<{W}}{pct(unsafe, risky)}   (gold DENY, model ALLOW; {unsafe}/{risky})")
    print(f"{'Over-deny rate':<{W}}{pct(over, allowable)}   (gold ALLOW, model DENY; {over}/{allowable})")
    print(f"{'fail stages':<{W}}{dict(stages) or 'none'}")

    return {
        "label": label, "policies": n, "runtime_cases": len(rows),
        "schema_valid": sum(r["schema_valid"] for r in records) / n if n else 0.0,
        "ontology_valid": ont / n if n else 0.0,
        "action_declared": sum(r["grounded"] for r in records) / n if n else 0.0,
        "action_accuracy": sum(r["action_correct"] for r in records) / n if n else 0.0,
        "graph_exact_match": exact / n if n else 0.0,
        "entry_precision": prec, "entry_recall": rec_, "entry_f1": f1,
        "field_accuracy": field_hit / field_tot if field_tot else 0.0,
        "field_operator_accuracy_strict": op_hit / op_tot if op_tot else 0.0,
        "field_value_accuracy_strict": val_hit / val_tot if val_tot else 0.0,
        "operator_accuracy_given_field": op_hit / opff if opff else 0.0,
        "value_accuracy_given_field": val_hit / opff if opff else 0.0,
        "runtime_decision_acc_case": correct / len(rows) if rows else 0.0,
        "runtime_decision_acc_policy": macro_runtime,
        "unsafe_allow_rate": unsafe / risky if risky else 0.0,
        "over_deny_rate": over / allowable if allowable else 0.0,
        "fail_stages": dict(stages),
    }


# ---------- data loading ----------
def load_eval(which):  # which in {simple, full}
    eval_path, seed_path = EVAL_FILES[which]
    rows = []
    for line in (ROOT / eval_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        assert r["instruction"] == INSTRUCTION, "eval prompt drifted from generator INSTRUCTION"
        r["policy"] = r["input"].split("Policy: ", 1)[1]
        r["gold"] = json.loads(r["output"])
        rows.append(r)
    if which == "simple":
        rows = [r for r in rows
                if len(r["gold"]["rules"]) == 1 and len(r["gold"]["rules"][0]["deny_when"]) == 1]
    seed = json.loads((ROOT / seed_path).read_text())
    by_text = {p["policy_text"]: p for p in seed["policies"]}
    for r in rows:
        p = by_text[r["policy"]]
        r["cases"] = p.get("runtime_cases", [])
    return rows


def run(which, gen, limit=None, model=None, lat=None):
    rows = load_eval(which)
    if limit:
        rows = rows[:limit]
    records = []
    for i, r in enumerate(rows, 1):
        raw = gen(r)
        if model is not None and lat is not None:
            lat["compile_ms"].append(model.last_ms); lat["compile_tok"].append(model.last_new_tokens); lat["input_tok"].append(model.last_input_tokens)
        rec = score_one(r, r["gold"], r["cases"], raw)
        rec["stratum"] = strata_of(r["gold"])
        records.append(rec)
        if i % 10 == 0:
            print(f"  {which}: {i}/{len(rows)}", file=sys.stderr)
    return records


# ---------- main ----------

def _pctl(xs, q):
    if not xs: return 0.0
    xs = sorted(xs); return xs[min(len(xs) - 1, int(q * len(xs)))]

def bench_engine(reps=1000, warmup=200):
    """engine_latency: wall-clock of a single check_action call, micro-benchmarked over every
    held-out runtime case x reps (deterministic, model-independent, CPU-portable)."""
    import time
    rows = load_eval("full")
    triples = []
    for r in rows:
        g = RuleGraph.model_validate(r["gold"])
        for c in r["cases"]:
            triples.append((g, c.get("completed_actions", []), c.get("state", {}),
                            ProposedAction(**c["proposed_action"])))
    for i in range(min(warmup, len(triples))):
        g, ca, st, pa = triples[i % len(triples)]; check_action(g, ca, st, pa)
    out = []
    for (g, ca, st, pa) in triples:
        t0 = time.perf_counter()
        for _ in range(reps): check_action(g, ca, st, pa)
        out.append((time.perf_counter() - t0) / reps * 1e6)  # us per call
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--base-only", action="store_true", help="evaluate the base model, no adapter")
    ap.add_argument("--selftest", action="store_true",
                    help="no model; feed gold as prediction; every metric must be perfect")
    ap.add_argument("--evals", nargs="+", default=["simple", "full"], choices=["simple", "full"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dump", default=None)
    ap.add_argument("--latency", action="store_true", help="measure compile (GPU) + engine (CPU) latency")
    ap.add_argument("--train-data", default=None, help="training jsonl for OOD strata (default dataset)")
    ap.add_argument("--nebius-platform", default=None, help="stamp into latency dump for the demo, e.g. gpu-l40s-d")
    ap.add_argument("--nebius-preset", default=None, help="stamp into latency dump, e.g. 1gpu-16vcpu-96gb")
    ap.add_argument("--project", action="store_true",
                    help="map non-canonical outputs to nearest ontology token (constrained-decoding stand-in)")
    args = ap.parse_args()
    global PROJECT; PROJECT = args.project
    if args.train_data: init_strata(args.train_data)

    model = None
    if args.selftest:
        gen = lambda r: r["output"]            # gold string as the "prediction"
        print("SELFTEST: feeding gold outputs as predictions (expect 100% / 0 unsafe / 0 over)")
    else:
        adapter = None if args.base_only else args.adapter
        if not args.base_only and adapter is None:
            ap.error("provide --adapter PATH, or --base-only, or --selftest")
        model = Model(args.base_model, adapter)
        gen = lambda r: model.generate(ALPACA.format(instruction=r["instruction"], input=r["input"]))

    # Single generation pass: run the widest requested slice ONCE, derive the narrower "simple"
    # slice as a filtered view (single-condition subset). Avoids re-generating the 50 simple
    # policies that are already inside full(80).
    lat = {"compile_ms": [], "compile_tok": [], "input_tok": []} if args.latency else None
    want_full = "full" in args.evals
    base = run("full" if want_full else "simple", gen, args.limit, model=model, lat=lat)
    record_sets = {}
    if want_full: record_sets["full"] = base
    if "simple" in args.evals:
        record_sets["simple"] = [r for r in base if r.get("single_condition")]
    summaries, dumps = {}, {}
    for which in args.evals:
        summaries[which] = aggregate(record_sets[which], f"{which} eval")
        dumps[which] = record_sets[which]

    print("\n-- held-out OOD strata (accuracy by difficulty) --")
    from collections import Counter as _C
    allrec = [rec for recs in dumps.values() for rec in recs if "full" in args.evals]
    if "full" in dumps:
        by = {}
        for rec in dumps["full"]: by.setdefault(rec["stratum"], []).append(rec)
        print(f"    (strata computed against training set: {_TRAIN_PATH.split('/')[-2]})")
        for st in ("eval_alias_ood_same_signature","in_compatibility","compositional_ood","coverage_layer_only","lexical_ood"):
            g = by.get(st, [])
            if not g: continue
            em = sum(r["exact_match"] for r in g)
            rc = [x for r in g for x in r["runtime"]]
            racc = sum(x["expected"]==x["got"] for x in rc)
            print(f"    {st:22} n={len(g):3}  exact_match {100*em/len(g):5.1f}%  runtime {100*racc/len(rc) if rc else 0:5.1f}%")

    latency_report = None
    if args.latency:
        dev = "cpu"; is_gpu = False
        try:
            import torch
            if torch.cuda.is_available(): dev = torch.cuda.get_device_name(0); is_gpu = True
        except Exception: pass
        cm = lat["compile_ms"][2:] if len(lat["compile_ms"]) > 2 else lat["compile_ms"]   # drop 2 warmup
        ct = lat["compile_tok"][2:] if len(lat["compile_tok"]) > 2 else lat["compile_tok"]
        it = lat["input_tok"][2:] if len(lat["input_tok"]) > 2 else lat["input_tok"]
        eng = bench_engine()
        latency_report = {
            "gpu_name": dev, "is_gpu": is_gpu, "base_model": args.base_model, "adapter": args.adapter,
            "nebius_platform": args.nebius_platform, "nebius_preset": args.nebius_preset,
            "engine_latency_us_p50": _pctl(eng, .5), "engine_latency_us_p95": _pctl(eng, .95),
            "engine_latency_us_p99": _pctl(eng, .99), "engine_bench_cases": len(eng),
        }
        print("\n-- [3] LATENCY --")
        if cm:
            toks = [1000.0 * t / m for t, m in zip(ct, cm) if m > 0]
            lbl = "Nebius GPU compile latency" if is_gpu else "Compile latency (CPU -- NOT deployment)"
            latency_report.update({
                "compile_latency_ms_p50": _pctl(cm, .5), "compile_latency_ms_p95": _pctl(cm, .95),
                "compile_latency_ms_p99": _pctl(cm, .99), "tokens_per_second": _pctl(toks, .5),
                "input_tokens_median": _pctl(it, .5), "output_tokens_median": _pctl(ct, .5),
                "compile_samples": len(cm),
            })
            print(f"{'device / platform':<40}{dev}  |  {args.nebius_platform or '-'} / {args.nebius_preset or '-'}")
            print(f"{'base_model / adapter':<40}{args.base_model}  |  {args.adapter}")
            print(f"{lbl:<40}p50 {_pctl(cm,.5):.1f}  p95 {_pctl(cm,.95):.1f}  p99 {_pctl(cm,.99):.1f} ms/policy  (batch=1, greedy, 256tok)")
            print(f"{'Compilation throughput':<40}{_pctl(toks,.5):.0f} tok/s (median) | in {int(_pctl(it,.5))} out {int(_pctl(ct,.5))} tok")
        else:
            print(f"{'Policy compilation latency':<40}(skipped: selftest/no model -- run on Nebius GPU for real numbers)")
        print(f"{'Engine decision latency':<40}p50 {_pctl(eng,.5):.2f}  p95 {_pctl(eng,.95):.2f}  p99 {_pctl(eng,.99):.2f} us/decision  (deterministic, CPU, no LLM; n={len(eng)} cases x1000 reps)")

    print("\n" + "=" * 52)
    print("HEADLINE")
    for which in args.evals:
        s = summaries[which]
        tag = "single-condition" if which == "simple" else "full set incl. conjunctions"
        print(f"  {which} ({tag}): graph_exact_match {100*s['graph_exact_match']:.1f}%  "
              f"runtime_acc_case {100*s['runtime_decision_acc_case']:.1f}%  "
              f"unsafe_allow {100*s['unsafe_allow_rate']:.1f}%  "
              f"over_deny {100*s['over_deny_rate']:.1f}%")
    print("=" * 52)

    if args.selftest:
        ok = all(s["graph_exact_match"] == 1.0 and s["runtime_decision_acc_case"] == 1.0
                 and s["unsafe_allow_rate"] == 0.0 and s["over_deny_rate"] == 0.0
                 and s["schema_valid"] == 1.0 and s["ontology_valid"] == 1.0 for s in summaries.values())
        print("\nSELFTEST", "PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)

    if args.dump:
        Path(args.dump).write_text(json.dumps({"summaries": summaries, "records": dumps,
                                               "latency": latency_report, "train_data": _TRAIN_PATH},
                                              indent=2, ensure_ascii=False))
        print(f"\nwrote {args.dump}")


if __name__ == "__main__":
    main()
