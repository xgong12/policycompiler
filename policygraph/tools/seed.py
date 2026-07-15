#!/usr/bin/env python3
"""
seed.py -- single source of truth loader/renderer for the held-out seed.

Reads seeds/expense_management.json and exposes, for the whole pipeline (generator, eval
builder, audit), everything derived FROM the seed so nothing drifts from it:

  ACTIONS            action -> description
  ONTOLOGY           action -> {cat, num, bool, other_enum, prereq}  (normalized keys)
  ontology_text()    the exact schema prompt block used in model inputs (canonical section labels)
  compatibility helpers: allowed_ops / cat_num_pairs / scoped_triples / cat_prereq_ok /
                         num_prereq_ok / standalone_prereqs
  held_out_guards()  (signatures, thresholds) for the leakage filter
  is_canonical(action, entry)  ontology validator

Nothing here is hand-copied from the seed; it is all read at import.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SEED_PATH = ROOT / "seeds" / "expense_management.json"

_SEED = json.loads(SEED_PATH.read_text())
DOMAIN = _SEED["domain"]
ACTIONS = _SEED["actions"]
POLICIES = _SEED["policies"]

# ontology section label -> normalized internal key
_LMAP = {"expense_categories": "cat", "numeric_fields": "num", "boolean_fields": "bool",
         "enum_fields": "other_enum", "prerequisite_actions": "prereq"}
_LABEL_ORDER = ["expense_categories", "numeric_fields", "boolean_fields", "prerequisite_actions"]

ONTOLOGY = {}
for _a, _spec in _SEED["ontology"].items():
    ONTOLOGY[_a] = {
        "cat": list(_spec.get("expense_categories", [])),
        "num": list(_spec.get("numeric_fields", [])),
        "bool": list(_spec.get("boolean_fields", [])),
        "other_enum": {k: list(v) for k, v in (_spec.get("enum_fields") or {}).items()},
        "prereq": list(_spec.get("prerequisite_actions", [])),
    }

_COMPAT = _SEED["compatibility"]


def ontology_text():
    """The schema block prefixed to every model input. Labels match the seed ontology."""
    lines = ["Schema (canonical values):"]
    for a, spec in _SEED["ontology"].items():
        lines.append(f"\n[{a}]")
        if spec.get("expense_categories"): lines.append(f"expense_categories: {', '.join(spec['expense_categories'])}")
        if spec.get("numeric_fields"):     lines.append(f"numeric_fields: {', '.join(spec['numeric_fields'])}")
        if spec.get("boolean_fields"):     lines.append(f"boolean_fields: {', '.join(spec['boolean_fields'])}")
        for f, vals in (spec.get("enum_fields") or {}).items():
            lines.append(f"{f}: {', '.join(vals)}")
        if spec.get("prerequisite_actions"): lines.append(f"prerequisite_actions: {', '.join(spec['prerequisite_actions'])}")
    return "\n".join(lines)


# ---------- compatibility helpers (all read from the seed) ----------
def allowed_ops(field):
    nd = _COMPAT["numeric_direction"]
    return nd.get("by_field", {}).get(field, nd["default"])

def cat_num_pairs(action):
    return [(c, n) for c, ns in _COMPAT["category_numeric"].get(action, {}).items() for n in ns]

def scoped_triples(action):
    return [(f, v, n)
            for f, vs in _COMPAT["scoped_enum_numeric"].get(action, {}).items()
            for v, ns in vs.items() for n in ns]

def cat_prereq_ok(action, cat, prereq):
    return prereq in _COMPAT["category_prerequisite"].get(action, {}).get(cat, [])

def num_prereq_ok(action, num, prereq):
    return prereq in _COMPAT["numeric_prerequisite"].get(action, {}).get(num, [])

def standalone_prereqs(action):
    return list(_COMPAT["standalone_prerequisites"].get(action, []))


# ---------- validators ----------
def is_canonical(action, entry):
    s = ONTOLOGY[action]
    if "not_completed" in entry:
        return entry["not_completed"] in s["prereq"]
    f = entry["field"]
    if f == "expense_category":
        return entry["value"] in s["cat"]
    if isinstance(entry.get("value"), bool):
        return f in s["bool"]
    if isinstance(entry.get("value"), (int, float)):
        return f in s["num"]
    return f in s["other_enum"] and entry["value"] in s["other_enum"][f]


# ---------- held-out guards (leakage filter) ----------
def _rule_signature(action, body):
    norm = tuple(sorted(
        ("nc:" + e["not_completed"]) if "not_completed" in e
        else f'{e["field"]}|{e["operator"]}|{e["value"]!r}'
        for e in body))
    return (action, norm)

def held_out_guards():
    sigs, thresholds = set(), set()
    for p in POLICIES:
        for r in p["rule_graph"]["rules"]:
            a, body = r["target_action"], r["deny_when"]
            sigs.add(_rule_signature(a, body))
            for e in body:
                if "field" in e and isinstance(e["value"], (int, float)) and not isinstance(e["value"], bool):
                    thresholds.add((a, e["field"], e["operator"], e["value"]))
    return sigs, thresholds


if __name__ == "__main__":
    print("actions:", list(ACTIONS))
    print("ontology_text chars:", len(ontology_text()))
    for a in ONTOLOGY:
        print(f"  {a}: cat={len(ONTOLOGY[a]['cat'])} num={len(ONTOLOGY[a]['num'])} "
              f"bool={len(ONTOLOGY[a]['bool'])} enum={len(ONTOLOGY[a]['other_enum'])} "
              f"prereq={len(ONTOLOGY[a]['prereq'])}")
    s, t = held_out_guards()
    print("held-out signatures:", len(s), " thresholds:", len(t))
