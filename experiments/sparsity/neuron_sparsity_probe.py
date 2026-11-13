"""Go/no-go probe: do feed-forward neurons need to be read for every token, and can NSA context predict which ones?

For a Hugging Face causal LM whose MLP is ``down_proj(act(gate_proj(x)) * up_proj(x))`` (Qwen2/Llama style), the hidden vector h feeding ``down_proj`` has one entry
per neuron; neuron i's contribution to the layer output is ``|h_i| * ||W_down[:, i]||``. A hook on ``down_proj`` zeroes all but a kept set of neurons, in every layer, and
the masked model's next-token prediction is compared with the dense model's at every position of held-out text.

Kept sets (each at a keep fraction k of neurons per layer):
* ``oracle``      - the top-k contributors of THIS token (upper bound: needs the full computation to know, so not a speedup by itself)
* ``prev_token``  - the top-k contributors of the previous token (what token-local predictors exploit)
* ``topic``       - a STATIC set per topic: the top-k by mean contribution on other text of the same topic (what a context-keyed residency set would preload)
* ``global``      - one static set from all topics
* ``wrong_topic`` - another topic's static set (control: is the topic information real?)
* ``random``      - a random static set (control)
* ``topic_mean`` / ``global_mean`` - as topic / global, but a dropped neuron is replaced by its average value (a constant per layer, free at run time) instead of zero

Metrics: top-1 agreement with the dense model and mean KL(dense || masked). Also the per-token concentration: share of neurons needed for 90 % / 99 % of contribution.
Boundary: a 0.5B SiLU model, 4 topics, short texts; masking only saves work if the kept set is known BEFORE computing h, which only the static and prev-token sets are.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

TOPICS = {
    "code": [
        "To reverse a linked list in Python you keep three pointers, previous, current and next, and walk the list once while flipping each node's next reference to point backwards.",
        "A race condition happens when two threads read and write the same variable without a lock, so the final value depends on scheduling; a mutex or an atomic operation fixes it.",
        "In SQL an inner join returns only rows with matching keys in both tables, while a left join keeps every row from the left table and fills missing columns with null values.",
        "The function parses the JSON payload, validates each field against the schema, raises a ValueError for unknown keys and returns a typed dataclass instance to the caller.",
        "Git rebase rewrites commits on top of another branch which keeps history linear, but you should never rebase commits that were already pushed to a shared remote branch.",
        "A hash map stores key value pairs in buckets chosen by a hash function, giving constant time lookups on average and linear time when many keys collide in one bucket.",
    ],
    "cooking": [
        "To make a basic bread dough mix flour, water, salt and yeast, knead it for ten minutes until smooth, let it rise until doubled and bake it in a hot oven until golden.",
        "Sear the steak in a very hot pan for two minutes per side, add butter, garlic and thyme, baste it continuously and let the meat rest before slicing against the grain.",
        "A good tomato sauce starts with onions softened slowly in olive oil, then garlic, crushed tomatoes and a pinch of sugar, simmered gently until thick and finished with basil.",
        "Whisk the egg yolks with sugar until pale, fold in the mascarpone, layer it with coffee soaked biscuits and chill the tiramisu overnight before dusting it with cocoa.",
        "Roast the chicken at high heat with lemon and rosemary, basting it with its own juices, until the skin is crisp and the thickest part of the thigh runs clear.",
        "For a creamy risotto toast the rice in butter, add warm stock one ladle at a time while stirring, and finish with parmesan and a knob of cold butter off the heat.",
    ],
    "finance": [
        "A bond's price falls when interest rates rise because its fixed coupon becomes less attractive than new issues, and the effect is larger for bonds with longer duration.",
        "The company reported quarterly revenue growth of eight percent, but operating margin narrowed as higher wages and freight costs outpaced the price increases it passed on.",
        "Diversification lowers portfolio risk because assets that are not perfectly correlated offset each other's losses, although it cannot remove the risk of the whole market.",
        "Compound interest means you earn returns on past returns, so an investment growing seven percent a year roughly doubles in ten years without any additional deposits.",
        "The central bank raised its policy rate to slow inflation, which increases borrowing costs for households and firms and usually strengthens the currency in the short term.",
        "Free cash flow is operating cash flow minus capital expenditure, and investors watch it because it shows the money available for dividends, buybacks and paying down debt.",
    ],
    "medicine": [
        "Type two diabetes develops when cells respond poorly to insulin and the pancreas cannot compensate, so blood glucose stays high and slowly damages vessels and nerves.",
        "The patient presented with fever, productive cough and chest pain, and the x-ray showed a lobar infiltrate consistent with bacterial pneumonia requiring antibiotic treatment.",
        "Vaccines train the immune system by presenting a harmless antigen, so memory cells can produce antibodies quickly when the real pathogen is encountered later in life.",
        "High blood pressure often has no symptoms, yet over years it strains the heart and arteries and raises the risk of stroke, kidney disease and heart failure considerably.",
        "An MRI scan uses a strong magnetic field and radio waves to image soft tissue in detail, which makes it better than a CT scan for the brain, spinal cord and joints.",
        "Antibiotics kill bacteria or stop their growth but do nothing against viruses, and overusing them selects resistant strains that are much harder to treat afterwards.",
    ],
}
KEEPS = (0.10, 0.25, 0.50)


class Masker:
    """Forward pre-hooks on every ``down_proj``: record contributions, or zero all neurons outside a kept set."""

    def __init__(self, model):
        self.layers = [m for n, m in model.named_modules() if n.endswith("mlp.down_proj")]
        self.norms = [l.weight.detach().float().norm(dim=0) for l in self.layers]          # ||W_down[:, i]|| per neuron
        self.mode, self.keep, self.sets, self.record = "dense", 1.0, None, None
        self.fill, self.record_h = None, None                                              # fill: per-layer mean h used in place of dropped neurons (mean ablation)
        for i, l in enumerate(self.layers):
            l.register_forward_pre_hook(self._hook(i))

    def _hook(self, i):
        def fn(module, args):
            h = args[0]
            c = h.detach().float().abs() * self.norms[i]                                    # (1, T, N)
            if self.record is not None:
                self.record[i].append(c[0])
                self.record_h[i].append(h.detach().float()[0])
            if self.mode == "dense":
                return None
            n = c.shape[-1]
            k = max(1, int(n * self.keep))
            if self.mode in ("oracle", "prev_token"):
                idx = c.topk(k, dim=-1).indices                                             # (1, T, k)
                mask = torch.zeros_like(c).scatter_(-1, idx, 1.0)
                if self.mode == "prev_token":
                    mask = torch.cat([torch.ones_like(mask[:, :1]), mask[:, :-1]], dim=1)   # position t uses t-1's set; the first token is dense
            else:
                mask = self.sets[i].view(1, 1, n)
                if self.fill is not None:
                    return ((h * mask + self.fill[i].view(1, 1, n) * (1 - mask)).to(h.dtype),)
            return (h * mask.to(h.dtype),)
        return fn


def static_set(mean_contrib, keep):
    out = []
    for m in mean_contrib:
        mask = torch.zeros_like(m)
        mask[m.topk(max(1, int(m.numel() * keep))).indices] = 1.0
        out.append(mask)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--out", default="results/sparsity_probe_qwen0.5b.json")
    p.add_argument("--train-per-topic", type=int, default=4)
    args = p.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).eval()
    mk = Masker(model)
    L = len(mk.layers)

    def run(text):
        ids = tok(text, return_tensors="pt").input_ids
        with torch.no_grad():
            return model(ids).logits[0].float()

    # 1. profile contributions on the training texts of each topic
    mean, mean_h = {}, {}
    conc = {0.90: [], 0.99: []}
    for topic, texts in TOPICS.items():
        mk.mode, mk.record, mk.record_h = "dense", [[] for _ in range(L)], [[] for _ in range(L)]
        for t in texts[: args.train_per_topic]:
            run(t)
        per_layer = [torch.cat(r, dim=0) for r in mk.record]                               # (tokens, N)
        mean[topic] = [c.mean(dim=0) for c in per_layer]
        mean_h[topic] = [torch.cat(r, dim=0).mean(dim=0) for r in mk.record_h]
        for c in per_layer:                                                                 # concentration: neurons needed for 90 / 99 % of the contribution mass
            s = c.sort(dim=-1, descending=True).values
            cum = s.cumsum(-1) / s.sum(-1, keepdim=True)
            for q in conc:
                conc[q].append(((cum < q).sum(-1).float() + 1).div(c.shape[-1]).mean().item())
        mk.record = mk.record_h = None
    glob_h = [torch.stack([mean_h[t][i] for t in TOPICS]).mean(0) for i in range(L)]
    glob = [torch.stack([mean[t][i] for t in TOPICS]).mean(0) for i in range(L)]
    topics = list(TOPICS)

    # 2. evaluate on held-out texts
    results = {}
    dense = {}
    for topic in topics:
        mk.mode = "dense"
        dense[topic] = [run(t) for t in TOPICS[topic][args.train_per_topic:]]
    for keep in KEEPS:
        for cond in ("oracle", "prev_token", "topic", "global", "wrong_topic", "random", "topic_mean", "global_mean"):
            agree = kl = n = 0
            for ti, topic in enumerate(topics):
                mk.keep, mk.fill = keep, None
                if cond in ("oracle", "prev_token"):
                    mk.mode = cond
                else:
                    mk.mode = "static"
                    if cond in ("topic", "topic_mean"):
                        mk.sets = static_set(mean[topic], keep)
                        mk.fill = mean_h[topic] if cond == "topic_mean" else None
                    elif cond in ("global", "global_mean"):
                        mk.sets = static_set(glob, keep)
                        mk.fill = glob_h if cond == "global_mean" else None
                    elif cond == "wrong_topic":
                        mk.sets = static_set(mean[topics[(ti + 1) % len(topics)]], keep)
                    else:
                        g = torch.Generator().manual_seed(1)
                        mk.sets = static_set([torch.rand(m.shape, generator=g) for m in glob], keep)
                for text, d in zip(TOPICS[topic][args.train_per_topic:], dense[topic]):
                    m = run(text)
                    agree += (m.argmax(-1) == d.argmax(-1)).sum().item()
                    kl += torch.nn.functional.kl_div(m.log_softmax(-1), d.log_softmax(-1), log_target=True, reduction="sum").item()
                    n += d.shape[0]
            results[f"{keep}|{cond}"] = {"top1_agreement": agree / n, "kl": kl / n, "tokens": n}
            print(f"keep {keep:.2f} {cond:<12} top-1 agreement {agree / n:.3f}  KL {kl / n:.3f}", flush=True)
    mk.mode = "dense"
    # 3. how different are the topic sets? (Jaccard overlap of static sets at 25 %)
    sets = {t: torch.cat(static_set(mean[t], 0.25)) for t in topics}
    jac = {f"{a}~{b}": ((sets[a] * sets[b]).sum() / ((sets[a] + sets[b]) > 0).sum()).item() for i, a in enumerate(topics) for b in topics[i + 1:]}
    summary = {"experiment": "neuron_sparsity_probe", "model": args.model, "layers": L, "neurons_per_layer": int(mk.norms[0].numel()),
               "share_of_neurons_for_90pct_contribution": sum(conc[0.90]) / len(conc[0.90]), "share_of_neurons_for_99pct_contribution": sum(conc[0.99]) / len(conc[0.99]),
               "results": results, "topic_set_jaccard_at_25pct": jac,
               "boundary": "0.5B SiLU model, 4 topics, 2 held-out texts per topic; oracle needs the full computation, only static and prev-token sets are known in advance."}
    print(f"neurons for 90% of contribution: {summary['share_of_neurons_for_90pct_contribution']:.2f}  for 99%: {summary['share_of_neurons_for_99pct_contribution']:.2f}")
    print("topic-set overlap (Jaccard @25%):", {k: round(v, 2) for k, v in jac.items()})
    Path(args.out).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
