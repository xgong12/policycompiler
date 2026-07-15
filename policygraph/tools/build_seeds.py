#!/usr/bin/env python3
"""
build_seeds.py -- rebuild the held-out seed eval inputs in the schema-grounded format.

For every held-out policy, the eval input becomes:  INSTRUCTION + ONTOLOGY + "Policy: <text>",
using the SAME INSTRUCTION and ONTOLOGY the training generator emits (imported, not copied).
The expected output is the seed's existing gold rule_graph -- values/operators/fields UNCHANGED;
only the JSON key order is normalized to the training serialization ({target_action, deny_when})
so a downstream exact-match isn't penalized by key ordering. This is verified below.

    python3 tools/build_seeds.py

Writes:
    seeds_eval/expense_management_eval.jsonl   (80 rows)
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from generate import INSTRUCTION, ONTOLOGY, DOMAIN, model_input  # noqa: E402

SEEDS = ROOT / "seeds"
OUT = ROOT / "seeds_eval"


def normalize_graph(rg):
    """Re-emit the gold rule_graph in training key order. Values are untouched."""
    return {
        "domain": rg["domain"],
        "rules": [
            {"target_action": r["target_action"], "deny_when": r["deny_when"]}
            for r in rg["rules"]
        ],
    }


def entries_multiset(rg):
    """Semantic fingerprint of a rule_graph: action + sorted deny_when entries. Order-insensitive."""
    out = []
    for r in rg["rules"]:
        norm = tuple(sorted(
            ("nc:" + e["not_completed"]) if "not_completed" in e
            else f'{e["field"]}|{e["operator"]}|{e["value"]!r}'
            for e in r["deny_when"]))
        out.append((r["target_action"], norm))
    return tuple(sorted(out))


def build(seed_name):
    blob = json.loads((SEEDS / seed_name).read_text())
    rows = []
    for p in blob["policies"]:
        text = p["policy_text"]
        gold = p["rule_graph"]
        norm = normalize_graph(gold)
        # invariant: normalization changed nothing semantic
        assert entries_multiset(gold) == entries_multiset(norm), f"gold altered: {p['policy_index']}"
        row = {
            "instruction": INSTRUCTION,
            "input": model_input(text),
            "output": json.dumps(norm, ensure_ascii=False),
        }
        # invariant: input begins with the ontology and ends with the exact policy text
        assert row["input"].startswith("Schema (canonical values):"), p["policy_index"]
        assert row["input"].endswith("Policy: " + text), p["policy_index"]
        assert norm["domain"] == DOMAIN, p["policy_index"]
        rows.append(row)
    return rows


def main():
    OUT.mkdir(exist_ok=True)
    for seed_name, out_name in [
        ("expense_management.json", "expense_management_eval.jsonl"),
    ]:
        rows = build(seed_name)
        (OUT / out_name).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
        print(f"{out_name}: {len(rows)} rows")
    print("ontology chars:", len(ONTOLOGY))


if __name__ == "__main__":
    main()
