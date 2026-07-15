# Nebius runs — reproducibility receipts

All fine-tuning and evaluation for PolicyCompiler ran as **Nebius Serverless AI Jobs** on the public
Axolotl image (`docker.io/axolotlai/axolotl:main-20260309-py3.11-cu128-2.9.1`), LoRA SFT
(r16 / α32 / 2 epochs) on the training data. Adapters and result dumps are stored in Nebius Object
Storage. The fine-tune completes in a few minutes per size on a single H100.

## Fine-tuning jobs (Qwen3)

| Model | GPU | Job ID |
|---|---|---|
| Qwen3-0.6B | Nebius H100 | `aijob-e00jnbsgtambhrrevw` |
| Qwen3-1.7B | Nebius H100 | `aijob-e00fsjwct51syq236g` |
| Qwen3-4B   | Nebius H100 | `aijob-e00yha3npja2zyaks1` |
| Qwen3-8B   | Nebius H100 | `aijob-e00h3455kqn4pcenz6` |

## Evaluation jobs (held-out 80 policies, `--latency` on H100)

Fine-tuned (LoRA adapter) vs. the untuned base checkpoint, evaluated apple-to-apple.

| Model | Job ID | Exact-match rate | Unsafe-allow | Compile latency p50 |
|---|---|---|---|---|
| Qwen3-0.6B          | `aijob-e00m2cvf81nyswey6q` | 68.8% | 26.5% | 1736 ms |
| Qwen3-1.7B          | `aijob-e00x5hk9ypxkp6313d` | 88.8% | 10.8% | 1594 ms |
| Qwen3-4B            | `aijob-e00yrfqvwqcq8sb88k` | 88.8% | 9.6%  | 2107 ms |
| Qwen3-8B            | `aijob-e00k3zfr5tg0fzhjqd` | 86.2% | 12.0% | 2261 ms |
| Qwen3-0.6B (base)   | `aijob-e00s0aew1pedq4808w` | 0.0%  | 95.2% | 3350 ms |
| Qwen3-1.7B (base)   | `aijob-e00yahz0ghq3we4kj4` | 3.8%  | 94.0% | 1363 ms |
| Qwen3-4B (base)     | `aijob-e00q3s6ygcfmankkmg` | 13.8% | 84.3% | 2004 ms |
| Qwen3-8B (base)     | `aijob-e00fw5yjg33gtk93km` | 12.5% | 83.1% | 1954 ms |

The headline results (README) focus on the small, single-GPU-trainable sizes (0.6B–4B). **8B is
included here for completeness: it gives no further improvement over 4B** (9.6% vs 12.0% unsafe-allow,
within the confidence interval) — consistent with data quality, not scale, driving performance.

Engine latency is ~1.5–1.7 µs (model-independent, benchmarked on the fixed gold rules). Compile
latency tracks output length.

Full metric definitions: [`evaluation_metrics.md`](evaluation_metrics.md).
