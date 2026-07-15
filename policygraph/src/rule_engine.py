#!/usr/bin/env python3
"""
rule_engine.py -- decide a proposed tool call: ALLOW or DENY.

A rule fires when every entry in its body holds, so an empty body always fires. Any rule
that fires contributes its verdict, and DENY outranks REVIEW. ALLOW is not a verdict a
rule can carry; it is what remains when nothing fires.

The two entry kinds disagree on what a missing input means, and the disagreement is the
point:

    a field absent from state          the rule does not apply, so it stays silent
    a name absent from completed_actions   the step was not done, so the rule fires

Silence on a missing fact is fail-open. Firing on a missing step is fail-closed. A policy
that constrains who may call an action, or describes a property of the claim, wants the
first. A policy that names a step which must precede the call wants the second. Choosing
the wrong one is how a guardrail lets a call through in silence.
"""
from typing import Union

from schemas import FieldEntry, NotCompletedEntry, ProposedAction, RuleGraph, Rule

COMPARISONS = {
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


def entry_holds(entry: Union[FieldEntry, NotCompletedEntry],
                state: dict, completed_actions: list[str]) -> bool:
    if isinstance(entry, NotCompletedEntry):
        name = entry.not_completed
        return name not in completed_actions and not state.get(name)
    if entry.field not in state:
        return False
    try:
        return COMPARISONS[entry.operator](state[entry.field], entry.value)
    except TypeError:
        # A number compared against a string, say. The rule cannot speak to this input.
        return False


def rule_fires(rule: Rule, state: dict, completed_actions: list[str]) -> bool:
    return all(entry_holds(e, state, completed_actions) for e in rule.body)


def validate_rule_graph_against_ontology(graph: dict, ontology: dict):
    """Executable-graph gate that sits BEFORE the PDP -- it is NOT the PDP and does not decide
    ALLOW/DENY. It only checks that a compiled rule graph references solely canonical
    action/field/value/prerequisite tokens from the ontology. No alias projection here: aliasing
    is PAP/PIP normalization and stays separate; this gate is pure safety validation.

    Returns (is_valid, violations). A caller can fail-closed (block/deny/audit) when not is_valid,
    so a malformed or non-grounded graph is never silently executed by the PDP.
    ontology: {action: {"cat":[...], "num":[...], "bool":[...], "other_enum":{f:[...]}, "prereq":[...]}}
    """
    violations = []
    for ru in graph.get("rules", []):
        a = ru.get("target_action")
        spec = ontology.get(a)
        if spec is None:
            violations.append(f"non-canonical action {a!r}")
            continue
        valid_fields = (set(spec.get("num", [])) | set(spec.get("bool", []))
                        | set(spec.get("other_enum", {})) | {"expense_category"})
        for e in ru.get("deny_when", []):
            if "not_completed" in e:
                if e["not_completed"] not in spec.get("prereq", []):
                    violations.append(f"{a}: non-canonical prerequisite {e['not_completed']!r}")
            else:
                f = e.get("field")
                if f not in valid_fields:
                    violations.append(f"{a}: non-canonical field {f!r}")
                elif f == "expense_category" and e.get("value") not in spec.get("cat", []):
                    violations.append(f"{a}: non-canonical category {e.get('value')!r}")
                elif f in spec.get("other_enum", {}) and e.get("value") not in spec["other_enum"][f]:
                    violations.append(f"{a}: non-canonical enum {f}={e.get('value')!r}")
    return (not violations, violations)


def check_action(graph: RuleGraph, completed_actions: list[str], state: dict,
                    proposed_action: ProposedAction) -> dict:
    """Any rule that fires blocks the automated call. The fired rules travel with the
    decision so a caller can show which policy blocked it -- and each rule carries the
    original policy_text upstream, which is the only explanation a person needs. There is
    no reason enum: the sentence that produced the rule is the reason."""
    fired = [r for r in graph.rules
             if r.target_action == proposed_action.name and rule_fires(r, state, completed_actions)]

    return {
        "decision": "DENY" if fired else "ALLOW",
        "fired_rules": [r.model_dump(exclude_none=True) for r in fired],
        "checked_rules": sum(1 for r in graph.rules if r.target_action == proposed_action.name),
    }
