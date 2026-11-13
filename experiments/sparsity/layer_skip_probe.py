"""Second go/no-go probe, at the granularity the residency module manages: can whole layers be left out, and does the topic say which?

Each transformer block is skipped alone (its output replaced by its input) on training text of each topic and the drop in agreement with the dense model is recorded.
Then the s least harmful blocks are skipped together, chosen from the SAME topic's ranking (``topic``), from all topics (``global``) or from another topic (``wrong_topic``),
and scored on held-out text: top-1 agreement with the dense model and KL. Skipping s of L blocks saves about s/L of the compute and of the weights that must be resident.
Boundary: 0.5B model, 4 topics, short texts, no fine-tuning (models trained for early exit / layer dropout tolerate far more).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.sparsity.neuron_sparsity_probe import TOPICS


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--out", default="results/layer_skip_probe_qwen0.5b.json")
    p.add_argument("--train-per-topic", type=int, default=4)
    args = p.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    blocks = list(model.model.layers)
    skip: set[int] = set()

    def hook(i):
        def fn(module, a, out):
            if i not in skip:
                return None
            return (a[0],) + tuple(out[1:]) if isinstance(out, tuple) else a[0]
        return fn
    for i, b in enumerate(blocks):
        b.register_forward_hook(hook(i))

    def run(text):
        with torch.no_grad():
            return model(tok(text, return_tensors="pt").input_ids).logits[0].float()

    def score(texts, dense):
        agree = kl = n = 0
        for t, d in zip(texts, dense):
            m = run(t)
            agree += (m.argmax(-1) == d.argmax(-1)).sum().item()
            kl += torch.nn.functional.kl_div(m.log_softmax(-1), d.log_softmax(-1), log_target=True, reduction="sum").item()
            n += d.shape[0]
        return agree, kl, n

    k = args.train_per_topic
    topics = list(TOPICS)
    harm = {}
    for t in topics:                                                    # single-block skips on training text
        train = TOPICS[t][:k]
        skip.clear()
        dense = [run(x) for x in train]
        harm[t] = []
        for i in range(len(blocks)):
            skip.clear(); skip.add(i)
            a, _, n = score(train, dense)
            harm[t].append(1 - a / n)
        print(t, "least harmful blocks:", sorted(range(len(blocks)), key=lambda i: harm[t][i])[:6], flush=True)
    glob = [sum(harm[t][i] for t in topics) / len(topics) for i in range(len(blocks))]
    results = {}
    skip.clear()
    dense = {t: [run(x) for x in TOPICS[t][k:]] for t in topics}
    for s in (1, 2, 4, 6):
        for cond in ("topic", "global", "wrong_topic"):
            A = K = N = 0
            for ti, t in enumerate(topics):
                rank = harm[t] if cond == "topic" else glob if cond == "global" else harm[topics[(ti + 1) % len(topics)]]
                skip.clear(); skip.update(sorted(range(len(blocks)), key=lambda i: rank[i])[:s])
                a, kl, n = score(TOPICS[t][k:], dense[t])
                A += a; K += kl; N += n
            results[f"{s}|{cond}"] = {"top1_agreement": A / N, "kl": K / N}
            print(f"skip {s}/{len(blocks)} {cond:<12} top-1 agreement {A / N:.3f}  KL {K / N:.3f}", flush=True)
    Path(args.out).write_text(json.dumps({"experiment": "layer_skip_probe", "model": args.model, "blocks": len(blocks), "single_skip_harm": harm, "results": results,
                                          "boundary": "0.5B model, 4 topics, 2 held-out texts per topic, no fine-tuning."}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
