"""
prototype/security/multi_probe_bench.py
================================
Multi-Level Adversarial Probing Security Suite.

Trains progressively stronger probing classifiers on hidden states from each
retrofit level to evaluate how much protected information (secret token IDs)
can be recovered from internal representations.

Probe tiers (increasingly powerful adversaries):
  Probe 1: Linear logistic regression
  Probe 2: 2-layer MLP (ReLU)
  Probe 3: Deep 4-layer MLP with residual connections
  Probe 4: Cross-layer concatenated probe (uses all layer outputs)
  Probe 5: Attention-based extractor probe

For each retrofit level, reports probe recovery rate (%) across all tiers.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from nsa.layers import NSACausalLM

# ─────────────────────────────── Probe models ────────────────────────────────

class LinearProbe(nn.Module):
    """Probe 1: Linear logistic regression."""
    def __init__(self, d): super().__init__(); self.fc = nn.Linear(d, 2)
    def forward(self, x): return self.fc(x)


class MLPProbe2(nn.Module):
    """Probe 2: 2-layer MLP."""
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 2))
    def forward(self, x): return self.net(x)


class MLPProbe4(nn.Module):
    """Probe 3: Deep 4-layer MLP with residual connection."""
    def __init__(self, d):
        super().__init__()
        self.l1 = nn.Linear(d, 128); self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 64); self.l4 = nn.Linear(64, 2)
        self.proj = nn.Linear(d, 128)
    def forward(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h)) + self.proj(x)
        h = F.relu(self.l3(h))
        return self.l4(h)


class CrossLayerProbe(nn.Module):
    """Probe 4: Probes concatenated outputs from multiple layers."""
    def __init__(self, d_per_layer, n_layers):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_per_layer * n_layers, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
    def forward(self, x): return self.net(x)


class AttentionProbe(nn.Module):
    """Probe 5: Self-attention based extractor."""
    def __init__(self, d):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, num_heads=4, batch_first=True)
        self.fc = nn.Linear(d, 2)
    def forward(self, x):
        # x: [N, 1, d] — treat each sample as a seq of length 1
        x_3d = x.unsqueeze(1)
        out, _ = self.attn(x_3d, x_3d, x_3d)
        return self.fc(out.squeeze(1))


PROBE_CLASSES = [
    ("Linear", LinearProbe),
    ("MLP-2L", MLPProbe2),
    ("MLP-4L (Residual)", MLPProbe4),
    ("Attention Extractor", AttentionProbe),
]


# ─────────────────────────── Data helpers ────────────────────────────────────

SECRET_IDS = {99, 98, 97, 96}
VOCAB_SIZE = 1000
D_MODEL = 128
SEQ_LEN = 64


def make_data(n=1200):
    torch.manual_seed(42)
    tokens = torch.randint(10, VOCAB_SIZE - 1, (n, SEQ_LEN))
    targets = torch.roll(tokens, -1, 1)
    tokens[:, 10:14] = torch.tensor([99, 98, 97, 96])
    return tokens, targets


def collect_hidden_states(model_fn, tokens, batch_size=32, device="cpu"):
    """Extract flat hidden states [N*T, d_model] and secret-presence labels [N*T]."""
    all_h, all_labels = [], []
    N = tokens.shape[0]
    for start in range(0, N, batch_size):
        b_tok = tokens[start:start + batch_size].to(device)
        with torch.no_grad():
            h = model_fn(b_tok)  # [B, T, d_model]
        all_h.append(h.cpu())
        labels = torch.zeros(b_tok.shape[0], b_tok.shape[1], dtype=torch.long)
        for sid in SECRET_IDS:
            labels |= (b_tok.cpu() == sid).long()
        all_labels.append(labels)
    H = torch.cat(all_h, 0).view(-1, D_MODEL)
    L = torch.cat(all_labels, 0).view(-1)
    return H, L


def train_and_eval_probe(probe: nn.Module, H_tr, y_tr, H_te, y_te,
                          device, epochs=30, lr=5e-4) -> float:
    probe = probe.to(device)
    H_tr, y_tr = H_tr.to(device), y_tr.to(device)
    H_te, y_te = H_te.to(device), y_te.to(device)
    opt = optim.Adam(probe.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(probe(H_tr), y_tr).backward()
        opt.step()
    with torch.no_grad():
        preds = probe(H_te).argmax(1)
        secret_mask = y_te == 1
        if secret_mask.sum() == 0:
            return 0.0
        return (preds[secret_mask] == 1).float().mean().item() * 100.0


# ─────────────────────────── Model builders ──────────────────────────────────

def build_baseline_hidden_fn(device):
    class BLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
            self.pos_emb = nn.Embedding(SEQ_LEN, D_MODEL)
            layer = nn.TransformerEncoderLayer(
                d_model=D_MODEL, nhead=4, dim_feedforward=D_MODEL*4,
                batch_first=True, norm_first=True
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=4)
            self.ln_f = nn.LayerNorm(D_MODEL)

        def forward(self, tokens):
            B, T = tokens.shape
            pos = torch.arange(T, device=tokens.device).unsqueeze(0)
            causal = torch.triu(torch.full((T, T), float("-inf"), device=tokens.device), diagonal=1)
            x = self.tok_emb(tokens) + self.pos_emb(pos)
            return self.ln_f(self.encoder(x, mask=causal, is_causal=True))

    m = BLM().to(device)
    m.eval()
    return m


# ─────────────────────────── Main benchmark ──────────────────────────────────

def run_multi_probe_bench(epochs: int = 3) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 90)
    print("MULTI-PROBE ADVERSARIAL SECURITY BENCHMARK")
    print(f"Device: {device} | Training epochs: {epochs}")
    print("=" * 90 + "\n")

    tokens, targets = make_data()
    n = tokens.shape[0]
    split = int(0.8 * n)
    tok_tr, tok_te = tokens[:split], tokens[split:]

    results = {}

    retrofit_levels = {
        "Level 0: Baseline":      None,   # Use baseline hidden fn
        "Level 3: Dynamic NSA":   "nsa",  # Use NSACausalLM hidden states
    }

    for level_name, level_type in retrofit_levels.items():
        print(f"--> Collecting hidden states for '{level_name}'...")

        if level_type is None:
            model = build_baseline_hidden_fn(device)
            hidden_fn = model
        else:
            nsa_model = NSACausalLM(
                vocab_size=VOCAB_SIZE, d_model=D_MODEL, state_dim=8,
                max_seq_len=SEQ_LEN, gate_mode="soft"
            ).to(device)
            # Train briefly
            opt = optim.AdamW(nsa_model.parameters(), lr=1e-3)
            crit = nn.CrossEntropyLoss()
            ds = TensorDataset(tok_tr, torch.roll(tok_tr, -1, 1))
            loader = DataLoader(ds, batch_size=32, shuffle=True)
            nsa_model.train()
            for _ in range(epochs):
                for b_tok, b_tgt in loader:
                    b_tok, b_tgt = b_tok.to(device), b_tgt.to(device)
                    opt.zero_grad()
                    logits, _, _ = nsa_model(b_tok)
                    crit(logits.view(-1, VOCAB_SIZE), b_tgt.view(-1)).backward()
                    opt.step()
            nsa_model.eval()

            def hidden_fn(tokens):
                with torch.no_grad():
                    _, x, _ = nsa_model(tokens.to(device))
                    return x

        H_all, L_all = collect_hidden_states(hidden_fn, tokens, device=device)
        split_h = int(0.8 * H_all.shape[0])
        H_tr, y_tr = H_all[:split_h], L_all[:split_h]
        H_te, y_te = H_all[split_h:], L_all[split_h:]

        level_results = {}
        for probe_name, ProbeCls in PROBE_CLASSES:
            probe = ProbeCls(D_MODEL)
            recovery = train_and_eval_probe(probe, H_tr, y_tr, H_te, y_te, device)
            level_results[probe_name] = recovery
            print(f"   {probe_name:<30}: {recovery:.2f}% recovery")

        results[level_name] = level_results
        print()

    # Summary table
    probe_names = [p[0] for p in PROBE_CLASSES]
    col_w = 20
    print("=" * 90)
    print("MULTI-PROBE SECURITY BENCHMARK SUMMARY")
    print("=" * 90)
    header = f"{'Retrofit Level':<35}" + "".join(f"{p:<{col_w}}" for p in probe_names)
    print(header)
    print("-" * 90)
    for lvl, lvl_res in results.items():
        row = f"{lvl:<35}" + "".join(f"{lvl_res.get(p, 0):.1f}%{'':<{col_w-7}}" for p in probe_names)
        print(row)
    print("=" * 90 + "\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Probe NSA Security Suite")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    run_multi_probe_bench(epochs=args.epochs)
