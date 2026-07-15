# Evaluation metrics

All metrics are computed on a held-out set of **80 expense-management policies**, each with a
hand-annotated gold structured JSON rule and annotated runtime test cases (**205 runtime cases**: 83
gold-DENY, 122 gold-ALLOW). Each policy is compiled into a structured rule and executed by the
deterministic engine against its runtime cases. Metrics fall into three groups: **compiler quality**,
**runtime safety**, and **latency**.

---

## Headline metrics

### 1. Compiler quality funnel

A three-stage funnel — Schema validity → Ontology validity → Exact-match rate — each stage applying
a stricter criterion to the predicted JSON policy rules. Higher is better.

- **Schema validity** — proportion of held-out policies whose model output parses into a well-formed
  `RuleGraph`.
- **Ontology validity** — proportion of held-out policies whose predicted rules have every reference
  (target action, field, value, prerequisite) canonical in the action-scoped ontology.
- **Exact-match rate** — proportion of held-out policies whose predicted JSON policy rules are
  identical to gold: the target action plus the full set of `deny_when` conditions (field, operator,
  value, or prerequisite), order-insensitive.

### 2. Runtime safety

Treating **"must block / DENY" as the positive class**, the guardrail can fail two ways; we report
both. Lower is better on both.

- **Unsafe-allow rate (false negatives)** — gold-DENY runtime cases the compiled rule incorrectly
  permits (a fail-open, the dangerous miss). The primary safety metric.
  `unsafe-allow = (gold-DENY cases predicted ALLOW) / 83`
- **Over-deny rate (false positives)** — gold-ALLOW runtime cases the compiled rule incorrectly
  denies (a false alarm). Reported alongside unsafe-allow so a low unsafe-allow rate cannot be
  achieved by simply denying everything.
  `over-deny = (gold-ALLOW cases predicted DENY) / 122`

Both are raw rates: a malformed output evaluates as an empty rule, which permits everything.

### 3. Nebius GPU compile latency

Median wall-clock to compile one policy into a structured rule on a Nebius GPU (batch 1, greedy),
ms/policy — the offline, per-policy authoring cost. Also reported at p95/p99. Lower is better.

### 4. Engine decision latency

Median wall-clock for the deterministic engine to evaluate one compiled rule against one proposed
tool call and return ALLOW/DENY, µs/decision — the online, per-tool-call cost; CPU-bound and
model-independent (not a Nebius metric). Also reported at p95/p99. Lower is better.

### Numerator / denominator

| Metric | Numerator | Denominator | Better |
|---|---|---|---|
| Schema validity | outputs that parse as a valid `RuleGraph` | 80 policies | ↑ |
| Ontology validity | rules that parse **and** are fully canonical in the ontology | 80 policies | ↑ |
| Exact-match rate | predicted rules identical to gold (target action + full `deny_when` set) | 80 policies | ↑ |
| Unsafe-allow rate (FN) | gold-DENY cases decided ALLOW | 83 gold-DENY runtime cases | ↓ |
| Over-deny rate (FP) | gold-ALLOW cases decided DENY | 122 gold-ALLOW runtime cases | ↓ |
| Compile latency p50 | — (percentile) | 78 per-policy `model.generate()` times (80 − 2 warm-up) | ↓ |
| Engine latency p50 | — (percentile) | 205 runtime cases × 1000 reps of `check_action()` (CPU, fixed gold rules) | ↓ |

Schema ≥ Ontology ≥ Exact-match rate share the 80-policy denominator and are progressively stricter
(a non-parsing output scores 0 on all three). The two latencies are percentiles, not ratios; engine
latency runs on the fixed gold rules, so it is model-independent.

---

## Results

Held-out full set (80 policies); latency on a Nebius H100. Each ratio cell shows the point estimate
with its **Wilson 95% CI** (schema/ontology/exact-match rate over 80 policies; unsafe-allow over 83
gold-DENY; over-deny over 122 gold-ALLOW). Gaps between fine-tuned sizes fall within these intervals,
so we do not claim a firm size ranking.

| Model | Schema validity ↑ | Ontology validity ↑ | Exact-match rate ↑ | Unsafe-allow (FN) ↓ | Over-deny (FP) ↓ | Compile latency p50 (ms) ↓ | Engine latency p50 (µs) ↓ |
|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | 96.2%<br>CI:[90%, 99%] | 87.5%<br>CI:[78%, 93%] | 68.8%<br>CI:[58%, 78%] | 26.5%<br>CI:[18%, 37%] | 2.5%<br>CI:[1%, 7%] | 1736 | 1.49 |
| Qwen3-1.7B | 100%<br>CI:[95%, 100%] | 98.8%<br>CI:[93%, 100%] | 88.8%<br>CI:[80%, 94%] | 10.8%<br>CI:[6%, 19%] | 0.0%<br>CI:[0%, 3%] | 1594 | 1.58 |
| Qwen3-4B | 100%<br>CI:[95%, 100%] | 98.8%<br>CI:[93%, 100%] | 88.8%<br>CI:[80%, 94%] | 9.6%<br>CI:[5%, 18%] | 0.8%<br>CI:[0%, 4%] | 2107 | 1.52 |

Headline focuses on the small, single-GPU-trainable sizes (0.6B–4B). 8B was also evaluated (86.2%
exact-match rate, 12.0% unsafe-allow) with no gain over 4B — data quality, not scale, drives
performance.

Compilation is a one-off, offline cost (~1–3 s/policy); runtime enforcement is a deterministic
~1.5 µs engine call — the two phases differ by roughly six orders of magnitude.

### Fine-tuning impact — baseline vs fine-tuned

Each cell is **baseline → fine-tuned**. The baseline is the identical `Qwen/Qwen3-<size>-Base`
checkpoint, evaluated apple-to-apple (same instruction, in-prompt ontology, greedy decoding, and
scoring) — the only variable is the LoRA adapter, so each arrow isolates the contribution of
fine-tuning.

| Size | Schema validity ↑ | Ontology validity ↑ | Exact-match rate ↑ | Unsafe-allow (FN) ↓ | Over-deny (FP) ↓ | Compile latency p50 (ms) ↓ | Engine latency p50 (µs) ↓ |
|---|---|---|---|---|---|---|---|
| 0.6B | 6.2% → 96.2% | 2.5% → 87.5% | 0.0% → 68.8% | 95.2% → 26.5% | 0.0% → 2.5% | 3350 → 1736 | 1.66 → 1.49 |
| 1.7B | 8.8% → 100% | 6.2% → 98.8% | 3.8% → 88.8% | 94.0% → 10.8% | 0.8% → 0.0% | 1363 → 1594 | 1.72 → 1.58 |
| 4B | 18.8% → 100% | 17.5% → 98.8% | 13.8% → 88.8% | 84.3% → 9.6% | 0.0% → 0.8% | 2004 → 2107 | 1.65 → 1.52 |

The untuned baseline is effectively unusable — the 0.6B baseline compiles a valid structured rule only
6.2% of the time (0/80 exact matches) and fails open on 95% of should-deny cases. **A fine-tuned
Qwen3-0.6B beats the untuned Qwen3-8B, a model 13× larger, on every metric.** The baseline →
fine-tuned jump dwarfs the spread between fine-tuned sizes: grounded data, not raw scale, buys the
safety.

### Latency, including the baseline

Every model, baseline and fine-tuned (Nebius H100; batch 1, greedy):

| Model | Compile latency p50 (ms/policy) | Engine latency p50 (µs/decision) |
|---|---|---|
| Qwen3-0.6B — fine-tuned | 1736 | 1.49 |
| Qwen3-0.6B — baseline | 3350 | 1.66 |
| Qwen3-1.7B — fine-tuned | 1594 | 1.58 |
| Qwen3-1.7B — baseline | 1363 | 1.72 |
| Qwen3-4B — fine-tuned | 2107 | 1.52 |
| Qwen3-4B — baseline | 2004 | 1.65 |

- **Engine latency is model-independent** — micro-benchmarked on the fixed gold rules, so every
  model lands at ~1.5–1.7 µs; the spread is measurement noise.
- **Compile latency tracks model size × output length.** Fine-tuned models emit short JSON and stop;
  the untuned baseline has no reliable stop behavior, so its output length is erratic — the 0.6B
  baseline rambled to 186 output tokens (~2× its compile time), while the 1.7B/4B baselines emitted
  76–83 tokens and ran at or below their fine-tuned counterparts. The baseline is not uniformly
  slower — it is unstable, within the same 1.4–3.3 s band.

---

## Supporting metrics (diagnostic)

- **Compiler quality (per-component).** Target-action accuracy; field identification; operator
  accuracy (strict); argument-value accuracy (strict); entry-level precision / recall / F1. Strict
  metrics use all gold field conditions as the denominator (a missed field counts as a failure);
  conditional variants (given the field was identified) are also available.
- **Runtime safety.** Enforcement decision accuracy — the proportion of runtime cases whose engine
  ALLOW/DENY matches the annotation; its two error directions are the unsafe-allow (FN) and over-deny
  (FP) rates reported above.

---

## Runtime completeness: a production caveat

This evaluation runs the compiled rules against the hand-annotated runtime cases (`state` and
`completed_actions`) directly, assuming the runtime context is complete and trustworthy — we do not
perform state-completeness validation here. This matters because the engine **fail-opens on an absent
field**: a condition it was not given stays silent, so the rule does not fire. An absent field has
two causes — it is genuinely not applicable to the call (correct to stay silent), or it *should* be
present but was not extracted (a pipeline failure that would silently permit).

These two are **not** distinguishable from the field alone, because fields are scoped to the *action*,
not to the specific call. One action buckets many expense types — `issue_reimbursement`, for instance,
declares `booking_lead_days` (flights), `hotel_nightly_rate` (hotels), `daily_meal_amount` (meals),
and ~10 more — so for any single call most declared fields are legitimately not applicable. Held-out
`expense_058` compiles to `{booking_lead_days < 14, not_completed business_justification}`; run
against a *meal* reimbursement, `booking_lead_days` is absent and the rule correctly stays silent
(ALLOW). Fail-open on an absent field is therefore usually the intended behavior, not a failure.

An absent field is only suspicious when a rule's **other** conditions already hold and it is one fact
away from firing — that is where an unresolvable field signals an extraction failure rather than
non-applicability. A production deployment should add an **attribute-resolution + completeness-validation**
layer before the PDP (the deterministic decision engine) that **fail-closes (deny / escalate) when a
field an otherwise-applicable rule needs cannot be resolved**, so a missing-but-required fact never
silently permits — while fields irrelevant to the call stay correctly silent and do not cause
spurious denials.
