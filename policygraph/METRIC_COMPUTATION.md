# Metric Computation (code map / reviewer note)

Metric computation is primarily implemented in `eval/run_eval.py`, with runtime decisions and
ontology validation delegated to `src/rule_engine.py`. `nebius/bench_compile.py` is a standalone
latency-only benchmark.

## Main script — `eval/run_eval.py`

Per-policy scoring happens in `score_one()`:

- Schema validity: `RuleGraph.model_validate(...)` succeeds.
- Ontology validity: `validate_rule_graph_against_ontology(pred, S.ONTOLOGY)` returns true.
- Exact-match rate: `graph_sig(pred) == graph_sig(gold_graph)`, using order-insensitive `frozenset`
  signatures over rules and `deny_when` entries.
- Runtime execution: the predicted rule is passed into `check_action()` for each annotated
  runtime case.

Aggregation happens in `aggregate()`:

- Unsafe-allow rate: gold decision is DENY, predicted engine decision is ALLOW.
- Enforcement decision accuracy: predicted engine decision equals the annotated expected decision.
- Over-deny rate: gold decision is ALLOW, predicted engine decision is DENY.
- Strict operator/value metrics use all gold field conditions as denominator, so a missed field also
  counts as an operator/value miss.

Latency:

- Nebius GPU compile latency is measured around `model.generate()` in `Model.generate()` and
  aggregated when `--latency` is enabled.
- Engine decision latency is measured by `bench_engine()`, repeatedly calling `check_action()`
  over held-out runtime cases.

## Runtime engine — `src/rule_engine.py`

- `check_action()` implements deterministic ALLOW/DENY execution.
- `validate_rule_graph_against_ontology()` checks whether the predicted rule uses only canonical
  ontology tokens.

## Standalone latency script — `nebius/bench_compile.py`

A minimal Nebius GPU benchmark for compile latency only.

## Audit focus

1. Exact-match rate is order-insensitive because rule and condition signatures use `frozenset`.
2. Raw unsafe-allow includes malformed-output fail-open behavior: if parsing fails, `score_one()`
   evaluates an empty rule, which permits all runtime cases.
3. Strict operator/value metrics use all gold field conditions as denominator, not only fields that
   were identified.
4. Exact-match rate compares entry values via `repr(value)`, so values must match exactly. In the
   current data this is stable because numeric thresholds are represented as integers.
5. Ontology validity is only computed for parsed rules; an output that fails schema parsing is
   counted as neither schema-valid nor ontology-valid, so ontology validity is bounded by schema
   validity.
