"""
prototype/retrofit/open_llm_retrofit.py
==============================
Phase 3 Benchmark: Open-LLM-style retrofit simulation (toy scale).

IMPORTANT
---------
This script is an **explicit simulation** of retrofit plumbing on a small
BaselineCausalLM.  It does **not** load Llama-3-8B / Qwen-2.5-7B weights and
does **not** run AdvGLUE.  Metrics below are measured on the toy model only.

Usage:
    python prototype/retrofit/open_llm_retrofit.py
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from nsa.lora import apply_nsa_lora_retrofit, NSALoRALinear
from nsa.utils import state_labels_to_vectors
from nsa.algebra import StateLabel, build_label_attention_mask
from prototype.pillars.pretrain_lm import BaselineCausalLM, generate_lm_corpus


def _wrap_encoder_linears(model: BaselineCausalLM, r: int = 8) -> int:
    """Wrap MultiheadAttention out_proj and FFN linears with LoRA."""
    replaced = 0
    for layer in model.encoder.layers:
        # out_proj is a Linear
        if isinstance(layer.self_attn.out_proj, nn.Linear):
            layer.self_attn.out_proj = NSALoRALinear(layer.self_attn.out_proj, r=r)
            replaced += 1
        if isinstance(layer.linear1, nn.Linear):
            layer.linear1 = NSALoRALinear(layer.linear1, r=r)
            replaced += 1
        if isinstance(layer.linear2, nn.Linear):
            layer.linear2 = NSALoRALinear(layer.linear2, r=r)
            replaced += 1
    return replaced


def run_open_llm_retrofit_simulation(
    epochs: int = 3,
    device: str = "cpu",
) -> Dict[str, float]:
    print("=" * 80)
    print("  PHASE 3 BENCHMARK: Open-LLM-Style Retrofit SIMULATION (toy scale)")
    print("  NOTE: Not Llama-3-8B / Qwen-2.5-7B. Not AdvGLUE. Toy BaselineCausalLM only.")
    print("=" * 80)

    vocab_size = 128
    d_model = 128
    num_layers = 4
    num_heads = 8
    state_dim = 8
    n_samples = 512
    seq_len = 48

    inputs, targets, state_vectors, state_labels = generate_lm_corpus(
        n_samples=n_samples, seq_len=seq_len, vocab_size=vocab_size, device=device
    )
    split = int(n_samples * 0.8)
    loader = DataLoader(
        TensorDataset(inputs[:split], targets[:split], state_vectors[:split], state_labels[:split]),
        batch_size=32,
        shuffle=True,
    )
    criterion = nn.CrossEntropyLoss()

    print("\n[1/3] Building toy pre-trained-style base model...")
    base_model = BaselineCausalLM(
        vocab_size=vocab_size, d_model=d_model, num_layers=num_layers, num_heads=num_heads, dropout=0.0
    ).to(device)
    opt = optim.AdamW(base_model.parameters(), lr=1e-3)
    base_model.train()
    for _ in range(2):
        for b_in, b_tgt, _, _ in loader:
            opt.zero_grad()
            loss = criterion(base_model(b_in).reshape(-1, vocab_size), b_tgt.reshape(-1))
            loss.backward()
            opt.step()

    base_model.eval()
    with torch.no_grad():
        pre_loss = criterion(base_model(inputs[split:]).reshape(-1, vocab_size), targets[split:].reshape(-1)).item()
        pre_ppl = math.exp(min(pre_loss, 20.0))
    print(f"  Pre-retrofit Val PPL: {pre_ppl:.2f}")

    print("\n[2/3] Applying NSA-LoRA retrofit (freeze base + wrap linears)...")
    # Freeze + state emb via shared helper
    _, stats = apply_nsa_lora_retrofit(base_model, state_dim=state_dim, r=8, add_state_emb=True)
    wrapped = _wrap_encoder_linears(base_model, r=8)
    # Recount after wrapping
    total = sum(p.numel() for p in base_model.parameters())
    trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    pct = (trainable / max(total, 1)) * 100.0
    print(f"  Layers wrapped               : {wrapped}")
    print(f"  Total parameters             : {total:,}")
    print(f"  Trainable parameters         : {trainable:,}")
    print(f"  Trainable ratio              : {pct:.2f}%")
    print(f"  apply_nsa_lora_retrofit note : layers_wrapped={stats.get('layers_wrapped', 0)}")

    # Fine-tune adapters briefly
    train_params = [p for p in base_model.parameters() if p.requires_grad]
    if train_params:
        ft_opt = optim.AdamW(train_params, lr=2e-3)
        base_model.train()
        for _ in range(epochs):
            for b_in, b_tgt, _, _ in loader:
                ft_opt.zero_grad()
                loss = criterion(base_model(b_in).reshape(-1, vocab_size), b_tgt.reshape(-1))
                loss.backward()
                ft_opt.step()

    base_model.eval()
    with torch.no_grad():
        post_loss = criterion(base_model(inputs[split:]).reshape(-1, vocab_size), targets[split:].reshape(-1)).item()
        post_ppl = math.exp(min(post_loss, 20.0))
    retention = (pre_ppl / max(post_ppl, 1e-6)) * 100.0

    print("\n[3/3] Toy policy-mask sanity (hard lattice, not AdvGLUE)...")
    # SYSTEM query vs UNTRUSTED key must be allowed; PUBLIC query vs PRIVATE key blocked
    q = torch.tensor([[StateLabel.SYSTEM.value, StateLabel.PUBLIC.value]])
    k = torch.tensor([[StateLabel.UNTRUSTED.value, StateLabel.PRIVATE.value]])
    mask = build_label_attention_mask(q, k)
    # mask[0,0,0,0] SYSTEM<-UNTRUSTED allowed 0
    # mask[0,0,1,1] PUBLIC<-PRIVATE forbidden < 0
    policy_ok = (mask[0, 0, 0, 0].item() == 0.0) and (mask[0, 0, 1, 1].item() < 0)
    policy_rate = 100.0 if policy_ok else 0.0

    print("\n" + "=" * 80)
    print("  OPEN LLM RETROFIT SIMULATION SUMMARY (TOY SCALE)")
    print("=" * 80)
    print("  SIMULATED = true")
    print("  Target claim architecture   : Llama-3-8B / Qwen-2.5-7B (NOT loaded)")
    print(f"  Toy trainable ratio          : {pct:.2f}%")
    print(f"  Toy task retention (PPL)     : {retention:.2f}%")
    print(f"  Lattice mask sanity          : {policy_rate:.2f}%")
    print(f"  Pre PPL → Post PPL           : {pre_ppl:.2f} → {post_ppl:.2f}")
    print("=" * 80)
    if policy_ok and trainable > 0 and trainable <= total:
        print("  [PASSED] Simulation plumbing OK (not an industrial verification).")
    else:
        print("  [FAILED] Simulation plumbing checks failed.")

    return {
        "simulated": 1.0,
        "pct_trainable": pct,
        "retention_pct": retention,
        "policy_mask_sanity_pct": policy_rate,
        "pre_ppl": pre_ppl,
        "post_ppl": post_ppl,
        "layers_wrapped": float(wrapped),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--hf-model",
        type=str,
        default="",
        help="If set, also run real HF retrofit path (prototype/retrofit/hf_nsa_retrofit.py)",
    )
    parser.add_argument("--skip-toy", action="store_true", help="Skip toy simulation")
    args = parser.parse_args()
    if not args.skip_toy:
        run_open_llm_retrofit_simulation(epochs=args.epochs)
    if args.hf_model:
        from prototype.retrofit.hf_nsa_retrofit import run_hf_retrofit

        run_hf_retrofit(model_id=args.hf_model)
