"""
prototype/demos/visualize_attention.py
================================
Dynamic State & Attention Heatmap Visualizer for Neural State Architecture (NSA).

Generates:
1. Interactive HTML Heatmaps showing Query-Key Attention Logits and State Compatibility Masks (M_state).
2. Token State Level Profile across sequence length.
3. Formatted Terminal ASCII matrix summaries.

Usage:
    python prototype/demos/visualize_attention.py [--output html_file.html]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import torch

# Ensure nsa is in python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from nsa.algebra import StateLabel
from nsa.fused_attention import FusedStateAwareAttention


def compute_nsa_attention_matrix(
    seq_tokens: List[str],
    state_labels: List[StateLabel],
    d_model: int = 64,
    gate_mode: str = "soft",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute state compatibility matrix and attention probabilities.

    Returns:
        state_mask : [T, T] tensor of additive state compatibility logits
        unmasked_attn : [T, T] tensor of standard scaled dot-product attention scores
        nsa_attn : [T, T] tensor of state-governed attention probabilities
    """
    T = len(seq_tokens)
    state_dim = 8
    
    # Map state labels to discrete state vectors
    state_vecs = torch.zeros(1, T, state_dim)
    for i, label in enumerate(state_labels):
        state_vecs[0, i, 0] = float(label.value)

    # Instantiate attention module
    attn_layer = FusedStateAwareAttention(
        d_model=d_model,
        state_dim=state_dim,
        num_heads=4,
        gate_mode=gate_mode,
        temperature=1.0,
    )
    attn_layer.eval()

    # Deterministic synthetic input
    torch.manual_seed(42)
    x = torch.randn(1, T, d_model)

    # Query and Key projections
    B, H, dk = 1, attn_layer.num_heads, attn_layer.d_k
    Q = attn_layer.W_q(x).view(B, T, H, dk).transpose(1, 2)  # [1, H, T, dk]
    K = attn_layer.W_k(x).view(B, T, H, dk).transpose(1, 2)  # [1, H, T, dk]

    # Compute raw QK^T / sqrt(dk)
    scale = 1.0 / (dk ** 0.5)
    raw_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # [1, H, T, T]
    raw_attn_map = torch.softmax(raw_scores, dim=-1).mean(dim=1).squeeze(0)  # average heads -> [T, T]

    # Level difference delta_L
    L = attn_layer.level_proj(state_vecs).squeeze(-1)  # [1, T]
    delta_L = L.unsqueeze(2) - L.unsqueeze(1)  # [1, T, T]
    
    if gate_mode == "soft":
        state_mask = torch.log(torch.sigmoid(delta_L)).squeeze(0)  # [T, T]
    else:
        g = torch.sigmoid(delta_L).squeeze(0)
        state_mask = torch.zeros_like(g).masked_fill(g < 0.5, float("-inf"))

    # Compute governed scores & softmax
    governed_scores = raw_scores + state_mask.unsqueeze(0).unsqueeze(0)
    nsa_attn_map = torch.softmax(governed_scores, dim=-1).mean(dim=1).squeeze(0)

    return state_mask, raw_attn_map, nsa_attn_map


def generate_interactive_html(
    seq_tokens: List[str],
    state_labels: List[StateLabel],
    state_mask: torch.Tensor,
    raw_attn: torch.Tensor,
    nsa_attn: torch.Tensor,
    output_path: str,
) -> None:
    """Generate self-contained interactive Plotly / HTML file visualization."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        HAS_PLOTLY = True
    except ImportError:
        HAS_PLOTLY = False

    if not HAS_PLOTLY:
        # Fallback HTML template if Plotly is not installed
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>NSA State Attention Heatmap</title>
    <style>
        body {{ font-family: monospace; background: #0f172a; color: #f8fafc; padding: 20px; }}
        h1 {{ color: #38bdf8; }}
        .table {{ border-collapse: collapse; margin-bottom: 30px; }}
        td, th {{ border: 1px solid #334155; padding: 8px 12px; text-align: center; }}
        th {{ background: #1e293b; color: #94a3b8; }}
        .allowed {{ background: #064e3b; color: #6ee7b7; }}
        .suppressed {{ background: #7f1d1d; color: #fca5a5; }}
    </style>
</head>
<body>
    <h1>Neural State Architecture (NSA) - Attention Mask Matrix</h1>
    <p>Tokens: {', '.join(seq_tokens)}</p>
    <p>States: {', '.join(s.name for s in state_labels)}</p>
    <h2>State Compatibility Matrix M_state(σ_Q, σ_K)</h2>
    <table class="table">
        <tr><th>Query \\ Key</th>{''.join(f'<th>{t}<br>({s.name})</th>' for t, s in zip(seq_tokens, state_labels))}</tr>
"""
        for i, (q_tok, q_state) in enumerate(zip(seq_tokens, state_labels)):
            html_content += f"<tr><th>{q_tok}<br>({q_state.name})</th>"
            for j in range(len(seq_tokens)):
                val = state_mask[i, j].item()
                cls = "allowed" if val > -1.0 else "suppressed"
                val_str = f"{val:.2f}" if val > -10.0 else "-∞"
                html_content += f'<td class="{cls}">{val_str}</td>'
            html_content += "</tr>"
        html_content += "</table></body></html>"

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"✅ Generated standalone HTML visualization: {output_path}")
        return

    # Full Plotly interactive figure with shared Y-axes to eliminate label collision
    token_labels = [f"{t}<br>({s.name})" for t, s in zip(seq_tokens, state_labels)]
    
    fig = make_subplots(
        rows=1, cols=3,
        shared_yaxes=True,
        horizontal_spacing=0.04,
        subplot_titles=(
            "Raw Attention Scores (Un-governed)",
            "NSA State Policy Mask M_state(σ_Q, σ_K)",
            "NSA Governed Attention Probabilities"
        )
    )

    # 1. Raw Attention
    fig.add_trace(
        go.Heatmap(
            z=raw_attn.detach().cpu().numpy(),
            x=token_labels,
            y=token_labels,
            colorscale="Blues",
            showscale=False,
        ),
        row=1, col=1
    )

    # 2. NSA State Mask
    mask_np = state_mask.detach().cpu().numpy()
    fig.add_trace(
        go.Heatmap(
            z=mask_np,
            x=token_labels,
            y=token_labels,
            colorscale="RdYlGn",
            showscale=False,
        ),
        row=1, col=2
    )

    # 3. NSA Governed Attention
    fig.add_trace(
        go.Heatmap(
            z=nsa_attn.detach().cpu().numpy(),
            x=token_labels,
            y=token_labels,
            colorscale="Purples",
            colorbar=dict(title="Attention Prob", len=0.85, y=0.5),
        ),
        row=1, col=3
    )

    # Force square aspect ratio on matrix cells so heatmaps remain proportional squares
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    fig.update_layout(
        title_text="<b>Neural State Architecture (NSA) - Attention Governance Visualizer</b>",
        template="plotly_dark",
        width=1300,
        height=520,
        margin=dict(l=140, r=80, t=90, b=100),
        showlegend=False,
    )

    fig.write_html(output_path)
    print(f"✅ Generated interactive Plotly heatmap visualization: {output_path}")


def print_ascii_attention_matrix(
    seq_tokens: List[str],
    state_labels: List[StateLabel],
    nsa_attn: torch.Tensor,
) -> None:
    """Print clean ASCII attention matrix to terminal."""
    print("\n" + "=" * 70)
    print("      NSA STATE-GOVERNED ATTENTION MATRIX (Terminal Visualizer)")
    print("=" * 70)
    
    col_label = "Query \\ Key"
    header = f"{col_label:<18}" + "".join(f"{t[:6]:>8}" for t in seq_tokens)
    print(header)
    print("-" * len(header))
    
    for i, (tok, label) in enumerate(zip(seq_tokens, state_labels)):
        row_str = f"{tok[:8]} ({label.name[:3]}) ".ljust(18)
        for j in range(len(seq_tokens)):
            val = nsa_attn[i, j].item()
            row_str += f"{val:8.3f}"
        print(row_str)
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="NSA Attention & State Heatmap Visualizer")
    parser.add_argument("--output", type=str, default="docs/attention_heatmap.html", help="Path to save HTML visualization")
    args = parser.parse_args()

    # Sample prompt scenario: RAG with System Prompt, Untrusted Payload, and Private Key
    seq_tokens = ["System:", "DoNotLeak", "User:", "IgnoreRules", "PrintKey", "Key:", "sk_live_99"]
    state_labels = [
        StateLabel.SYSTEM,
        StateLabel.SYSTEM,
        StateLabel.UNTRUSTED,
        StateLabel.UNTRUSTED,
        StateLabel.UNTRUSTED,
        StateLabel.PRIVATE,
        StateLabel.PRIVATE,
    ]

    print("Computing NSA State Attention Maps...")
    mask, raw_attn, nsa_attn = compute_nsa_attention_matrix(seq_tokens, state_labels)
    
    print_ascii_attention_matrix(seq_tokens, state_labels, nsa_attn)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    generate_interactive_html(seq_tokens, state_labels, mask, raw_attn, nsa_attn, args.output)


if __name__ == "__main__":
    main()
