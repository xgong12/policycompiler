#!/usr/bin/env python3
"""Minimal Nebius-GPU compile-latency benchmark.

Loads the fine-tuned compiler (base + LoRA adapter) and times how long it takes to compile each
held-out policy (natural language -> JSON rule graph) with greedy decoding, batch size 1. Prints
median / p95 / p99 wall-clock per policy and the GPU name. First 2 policies are dropped as warmup.

    python3 bench_compile.py <base_model> <adapter_dir> <prompts_jsonl>

Only needs torch + transformers + peft (present in the axolotl image). prompts_jsonl is the
held-out eval file (each row has instruction + input).
"""
import json, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE, ADAPTER, PROMPTS = sys.argv[1], sys.argv[2], sys.argv[3]

ALPACA = ("Below is an instruction that describes a task, paired with an input that provides "
          "further context. Write a response that appropriately completes the request.\n\n"
          "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n")

tok = AutoTokenizer.from_pretrained(BASE)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16).to("cuda")
model = PeftModel.from_pretrained(model, ADAPTER).eval().to("cuda")

rows = [json.loads(l) for l in open(PROMPTS) if l.strip()]
ms, out_toks = [], []
for i, r in enumerate(rows):
    prompt = ALPACA.format(instruction=r["instruction"], input=r["input"])
    inp = tok(prompt, return_tensors="pt").to("cuda")
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=256, do_sample=False,
                             temperature=None, top_p=None, pad_token_id=tok.pad_token_id)
    torch.cuda.synchronize()
    if i >= 2:  # drop 2 warmup
        ms.append((time.perf_counter() - t0) * 1000.0)
        out_toks.append(int(out[0].shape[0] - inp["input_ids"].shape[1]))

ms.sort()
def pct(q): return ms[min(len(ms) - 1, int(q * len(ms)))]
tps = sorted(1000.0 * t / m for t, m in zip(out_toks, ms) if m > 0)
print(f"GPU: {torch.cuda.get_device_name(0)}   base={BASE}   adapter={ADAPTER}")
print(f"Nebius GPU compile latency (ms/policy):  p50 {pct(.5):.1f}   p95 {pct(.95):.1f}   p99 {pct(.99):.1f}   (n={len(ms)})")
print(f"Compilation throughput:  {tps[len(tps)//2]:.0f} tok/s (median)")
