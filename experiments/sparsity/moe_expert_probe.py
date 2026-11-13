"""Mixture-of-experts probe: is the set of experts a topic uses predictable, and can the others stay unloaded?

A mixture-of-experts model already reads only ``top_k`` of its ``E`` experts per token and layer, but WHICH ones changes per token, so all experts must stay in memory.
This probe asks whether NSA's context (the topic) predicts a per-layer "hot set" of m experts such that routing restricted to it keeps the model's answers: experts
outside the set would never be touched, so their weights need not be resident (memory saving ~ (1 - m/E) of the expert weights; compute per token is unchanged).

Per layer the router is wrapped: it records which experts each token selects, or masks the logits of experts outside an allowed set before the top-k.
Hot sets (m experts per layer, by selection frequency on profiling text): ``topic`` (same topic), ``global`` (all topics), ``wrong_topic`` (another topic), ``random``.
``+start`` variants always include the experts the start token routes to: they are used once per text, so frequency ranks them last, yet every later token attends to
the start token's state (without them the frequency sets did WORSE than random ones). ``+cached`` variants instead route position 0 without restriction, which stands
for a start token whose state comes from a KV prefix cache (NSA's pinned prefix): its experts are then never needed at run time and the whole budget goes to content.
Reported on held-out text: ``coverage`` (share of the unrestricted model's expert selections that fall inside the set) and, with routing restricted to the set,
top-1 agreement with the unrestricted model and KL. Also how concentrated usage is per topic.
Boundary: one small model (Granite 3.0 1B, 32 experts, top-8), 4 topics, short texts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.sparsity.neuron_sparsity_probe import TOPICS


class Routers:
    def __init__(self, model):
        self.routers = [m for n, m in model.named_modules() if n.endswith(".router")]
        self.E, self.k = self.routers[0].num_experts, self.routers[0].top_k
        self.allowed = None            # per-layer bool mask (E,) or None
        self.record = None             # per-layer list of (tokens, k) index tensors
        self.free_start = False        # True: position 0 is routed without restriction (stands for a start token served from the KV prefix cache)
        for i, r in enumerate(self.routers):
            r.forward = self._forward(i, r)

    def _forward(self, i, r):
        def fn(hidden_states):
            logits = F.linear(hidden_states, r.weight).float()
            if self.allowed is not None:
                block = (~self.allowed[i]).expand(logits.shape).clone()
                if self.free_start:
                    block.view(-1, block.shape[-1])[0] = False
                logits = logits.masked_fill(block, float("-inf"))
            top_logits, top_index = logits.topk(r.top_k, dim=-1)
            if self.record is not None:
                self.record[i].append(top_index.reshape(-1, r.top_k).clone())
            return top_index, torch.softmax(top_logits, dim=-1).type_as(hidden_states), logits
        return fn

    def counts(self):
        return [torch.bincount(torch.cat(rec).flatten(), minlength=self.E).float() for rec in self.record]

    def first_token(self):
        """Experts chosen at position 0 of each recorded text (the start token routes to its own experts; frequency alone would drop them)."""
        return [torch.bincount(torch.cat([r[0] for r in rec]), minlength=self.E) > 0 for rec in self.record]


def hot(counts, m, must=None):
    """Per layer: the ``must`` experts (if given), then the most frequently selected ones, m in total."""
    out = []
    for i, c in enumerate(counts):
        score = c.clone().float()
        if must is not None:
            score[must[i]] += 1e9
        mask = torch.zeros(c.numel(), dtype=torch.bool)
        mask[score.topk(m).indices] = True
        out.append(mask)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="ibm-granite/granite-3.0-1b-a400m-instruct")
    p.add_argument("--out", default="results/moe_expert_probe_granite1b.json")
    p.add_argument("--train-per-topic", type=int, default=4)
    args = p.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    R = Routers(model)
    L, E, K = len(R.routers), R.E, R.k
    k = args.train_per_topic
    topics = list(TOPICS)

    def run(text):
        with torch.no_grad():
            return model(tok(text, return_tensors="pt").input_ids).logits[0].float()

    counts, first = {}, {}
    for t in topics:                                                     # profile expert usage on training text
        R.allowed, R.record = None, [[] for _ in range(L)]
        for x in TOPICS[t][:k]:
            run(x)
        counts[t] = R.counts()
        first[t] = R.first_token()
    glob = [sum(counts[t][i] for t in topics) for i in range(L)]
    first_all = [torch.stack([first[t][i] for t in topics]).any(0) for i in range(L)]
    print("start-token experts per layer (mean):", sum(int(f.sum()) for f in first_all) / L, flush=True)
    # concentration: experts needed per layer for 90 % of a topic's selections
    need90 = {t: sum(int(((c.sort(descending=True).values.cumsum(0) / c.sum()) < 0.9).sum()) + 1 for c in counts[t]) / L for t in topics}
    need90["all_topics"] = sum(int(((c.sort(descending=True).values.cumsum(0) / c.sum()) < 0.9).sum()) + 1 for c in glob) / L

    dense, picks = {}, {}
    for t in topics:                                                     # unrestricted reference on held-out text
        R.allowed, R.record = None, [[] for _ in range(L)]
        dense[t] = [run(x) for x in TOPICS[t][k:]]
        picks[t] = [torch.cat(rec) for rec in R.record]
    R.record = None
    results = {}
    g = torch.Generator().manual_seed(1)
    R.allowed = [torch.ones(E, dtype=torch.bool) for _ in range(L)]                       # sanity: an all-allowed mask must reproduce the model exactly
    assert all((run(x).argmax(-1) == d.argmax(-1)).all() for x, d in zip(TOPICS[topics[0]][k:], dense[topics[0]]))
    R.allowed = None
    for m in (8, 12, 16, 24):
        for cond in ("topic", "random", "topic+start", "global+start", "wrong_topic+start", "random+start", "topic+cached", "global+cached", "wrong_topic+cached", "random+cached"):
            cov = tot = agree = kl = n = 0
            for ti, t in enumerate(topics):
                base = cond.split("+")[0]
                src = counts[t] if base == "topic" else glob if base == "global" else counts[topics[(ti + 1) % len(topics)]] if base == "wrong_topic" \
                    else [torch.rand(E, generator=g) for _ in range(L)]
                sets = hot(src, m, first_all if cond.endswith("+start") else None)
                for i in range(L):
                    cov += sets[i][picks[t][i]].sum().item()
                    tot += picks[t][i].numel()
                R.allowed, R.free_start = sets, cond.endswith("+cached")
                for x, d in zip(TOPICS[t][k:], dense[t]):
                    o = run(x)
                    agree += (o.argmax(-1) == d.argmax(-1)).sum().item()
                    kl += F.kl_div(o.log_softmax(-1), d.log_softmax(-1), log_target=True, reduction="sum").item()
                    n += d.shape[0]
                R.allowed = None
            results[f"{m}|{cond}"] = {"coverage": cov / tot, "top1_agreement": agree / n, "kl": kl / n}
            print(f"hot set {m:>2}/{E} {cond:<18} coverage {cov / tot:.2f}  top-1 agreement {agree / n:.3f}  KL {kl / n:.3f}", flush=True)
    print("experts per layer needed for 90% of selections:", {a: round(b, 1) for a, b in need90.items()}, f"(of {E}, top-{K} per token)")
    Path(args.out).write_text(json.dumps({"experiment": "moe_expert_probe", "model": args.model, "layers": L, "experts": E, "top_k": K, "experts_for_90pct": need90,
                                          "results": results, "boundary": "One small MoE model, 4 topics, 2 held-out texts per topic, no fine-tuning."}, indent=2) + "\n",
                                         encoding="utf-8")


if __name__ == "__main__":
    main()
