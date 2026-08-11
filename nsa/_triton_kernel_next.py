"""
nsa.triton_kernel
=================
Optional Triton path for fused state-aware attention.

When the ``triton`` package is installed *and* tensors live on a CUDA device,
``triton_fused_state_attention`` dispatches a real ``@triton.jit`` kernel that
fuses QK^T scaling, NSA state-mask add, causal mask, softmax, and PV.

On CPU (or if Triton is missing / JIT fails), falls back to PyTorch SDPA with
the same mask algebra.

Flags
-----
HAS_TRITON
    ``triton`` import succeeded (package present; does not imply CUDA).
TRITON_KERNEL_DEFINED
    Real ``@triton.jit`` kernel object was defined in this module.
USING_TRITON_KERNEL
    True only for the duration of a successful JIT kernel launch.
    Call ``last_backend()`` for the most recent dispatch: ``triton``|``sdpa``|``manual``.
"""

from __future__ import annotations

from typing import Optional, Tuple
import math
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F

from nsa.algebra import DEFAULT_LATTICE, StateLattice, build_label_attention_mask

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover
    triton = None  # type: ignore
    tl = None  # type: ignore
    HAS_TRITON = False

USING_TRITON_KERNEL = False
_BACKEND_LOCK = threading.Lock()
_LAST_BACKEND = "sdpa"
_KERNEL_DEFINED = False


def last_backend() -> str:
    """Most recent attention backend used by triton_fused_state_attention."""
    with _BACKEND_LOCK:
        return _LAST_BACKEND


def _set_backend(name: str) -> None:
    global _LAST_BACKEND
    with _BACKEND_LOCK:
        _LAST_BACKEND = name


if HAS_TRITON:

    @triton.jit
    def _nsa_attn_fwd_kernel(
        Q, K, V, Out, Mask,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_om, stride_od,
        stride_mb, stride_mh, stride_mm, stride_mn,
        Tq, Tk,
        scale,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Fused NSA attention forward. Grid: (cdiv(Tq, BLOCK_M), B*H)."""
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)

        q_bh = Q + pid_bh * stride_qh
        k_bh = K + pid_bh * stride_kh
        v_bh = V + pid_bh * stride_vh
        o_bh = Out + pid_bh * stride_oh
        m_bh = Mask + pid_bh * stride_mh

        q_ptrs = q_bh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        q_mask = offs_m[:, None] < Tq
        q = tl.load(q_ptrs, mask=q_mask, other=0.0)
        q = q * scale

        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        for start_n in range(0, Tk, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            offs_n = start_n + tl.arange(0, BLOCK_N)

            k_ptrs = k_bh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
            k_mask = offs_n[:, None] < Tk
            k = tl.load(k_ptrs, mask=k_mask, other=0.0)

            scores = tl.dot(q, tl.trans(k))

            m_ptrs = m_bh + offs_m[:, None] * stride_mm + offs_n[None, :] * stride_mn
            m_valid = (offs_m[:, None] < Tq) & (offs_n[None, :] < Tk)
            m_vals = tl.load(m_ptrs, mask=m_valid, other=float("-inf"))
            scores = scores + m_vals
            scores = tl.where(offs_n[None, :] < Tk, scores, float("-inf"))

            m_ij = tl.max(scores, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new[:, None])
            l_ij = tl.sum(p, axis=1)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]

            v_ptrs = v_bh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
            v = tl.load(v_ptrs, mask=k_mask, other=0.0)
            acc = acc + tl.dot(p.to(v.dtype), v)
            m_i = m_new

        acc = acc / l_i[:, None]
        o_ptrs = o_bh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
        tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=q_mask)

    _KERNEL_DEFINED = True

TRITON_KERNEL_DEFINED = _KERNEL_DEFINED


def _pow2_at_least(n: int, lo: int = 16) -> int:
    p = 1
    while p < n:
        p *= 2
    return max(p, lo)


def _build_total_mask(
    q: torch.Tensor,
    q_state: torch.Tensor,
    k_state: torch.Tensor,
    level_proj_weight: Optional[torch.Tensor],
    temp: float,
    alpha: float,
    gate_mode: str,
    use_discrete_levels: bool,
    lattice: StateLattice,
    causal: bool,
) -> torch.Tensor:
    """Build additive [B, 1, Tq, Tk] mask (state + optional causal)."""
    B, H, Tq, d_head = q.shape

    if use_discrete_levels:
        q_labels = q_state[..., 0].round().long().clamp(0, 5)
        k_labels = k_state[..., 0].round().long().clamp(0, 5)
        if gate_mode == "hard":
            state_mask = build_label_attention_mask(
                q_labels, k_labels, lattice=lattice, forbidden_value=float("-inf")
            ).to(dtype=q.dtype)
        elif gate_mode == "off":
            Tk_ = k_labels.shape[-1]
            state_mask = torch.zeros(B, 1, Tq, Tk_, device=q.device, dtype=q.dtype)
        else:
            L_q = q_state[..., 0]
            L_k = k_state[..., 0]
            delta = (L_q.unsqueeze(2) - L_k.unsqueeze(1)) / max(temp, 1e-5)
            state_mask = (alpha * F.logsigmoid(delta)).unsqueeze(1)
    else:
        if level_proj_weight is None:
            raise ValueError("level_proj_weight required when use_discrete_levels=False")
        L_q = F.linear(q_state, level_proj_weight).squeeze(-1)
        L_k = F.linear(k_state, level_proj_weight).squeeze(-1)
        delta = (L_q.unsqueeze(2) - L_k.unsqueeze(1)) / max(temp, 1e-5)
        if gate_mode == "hard":
            g = torch.sigmoid(delta).unsqueeze(1)
            state_mask = torch.zeros_like(g).masked_fill(g < 0.5, float("-inf"))
        elif gate_mode == "off":
            state_mask = torch.zeros(
                B, 1, L_q.shape[1], L_k.shape[1], device=q.device, dtype=q.dtype
            )
        else:
            state_mask = (alpha * F.logsigmoid(delta)).unsqueeze(1)

    Tk = state_mask.shape[-1]
    if causal and Tq == Tk:
        causal_mask = torch.triu(
            torch.full((Tq, Tk), float("-inf"), device=q.device, dtype=q.dtype),
            diagonal=1,
        ).unsqueeze(0).unsqueeze(0)
        return state_mask + causal_mask
    return state_mask


def _launch_triton_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    total_mask: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Launch the JIT kernel. q/k/v: [B,H,T,d], mask: [B,1,Tq,Tk]."""
    global USING_TRITON_KERNEL
    assert HAS_TRITON and TRITON_KERNEL_DEFINED
    B, H, Tq, d_head = q.shape
    Tk = k.shape[2]
    if d_head > 128:
        raise ValueError("Triton NSA kernel supports d_head <= 128 in this build")

    q_ = q.contiguous()
    k_ = k.contiguous()
    v_ = v.contiguous()
    if total_mask.shape[1] == 1:
        mask_ = total_mask.expand(B, H, Tq, Tk).contiguous()
    else:
        mask_ = total_mask.contiguous()

    mask_f = torch.nan_to_num(mask_, neginf=-1e4).to(dtype=torch.float32)
    out = torch.empty_like(q_)

    q_f = q_.reshape(B * H, Tq, d_head)
    k_f = k_.reshape(B * H, Tk, d_head)
    v_f = v_.reshape(B * H, Tk, d_head)
    o_f = out.reshape(B * H, Tq, d_head)
    m_f = mask_f.reshape(B * H, Tq, Tk)

    BLOCK_M = 32 if Tq >= 32 else _pow2_at_least(max(Tq, 1), 16)
    BLOCK_N = 32 if Tk >= 32 else _pow2_at_least(max(Tk, 1), 16)
    BLOCK_D = _pow2_at_least(d_head, 16)

    if d_head != BLOCK_D:
        q_pad = torch.zeros(B * H, Tq, BLOCK_D, device=q.device, dtype=q.dtype)
        k_pad = torch.zeros(B * H, Tk, BLOCK_D, device=q.device, dtype=q.dtype)
        v_pad = torch.zeros(B * H, Tk, BLOCK_D, device=q.device, dtype=q.dtype)
        o_pad = torch.zeros(B * H, Tq, BLOCK_D, device=q.device, dtype=q.dtype)
        q_pad[:, :, :d_head] = q_f
        k_pad[:, :, :d_head] = k_f
        v_pad[:, :, :d_head] = v_f
        q_f, k_f, v_f, o_f = q_pad, k_pad, v_pad, o_pad

    grid = (triton.cdiv(Tq, BLOCK_M), B * H)

    USING_TRITON_KERNEL = True
    try:
        _nsa_attn_fwd_kernel[grid](
            q_f, k_f, v_f, o_f, m_f,
            q_f.stride(0), 0, q_f.stride(1), q_f.stride(2),
            k_f.stride(0), 0, k_f.stride(1), k_f.stride(2),
            v_f.stride(0), 0, v_f.stride(1), v_f.stride(2),
            o_f.stride(0), 0, o_f.stride(1), o_f.stride(2),
            m_f.stride(0), 0, m_f.stride(1), m_f.stride(2),
            Tq, Tk,
            float(scale),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_D,
        )
    finally:
        USING_TRITON_KERNEL = False

    if d_head != BLOCK_D:
        out = o_f[:, :, :d_head].reshape(B, H, Tq, d_head)
    else:
        out = o_f.reshape(B, H, Tq, d_head)
    return out.to(dtype=q.dtype)


def triton_fused_state_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_state: torch.Tensor,
    k_state: torch.Tensor,
    level_proj_weight: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    temp: float = 1.0,
    alpha: float = 1.0,
    gate_mode: str = "hard",
    use_discrete_levels: bool = True,
    lattice: StateLattice = DEFAULT_LATTICE,
    causal: bool = True,
    force_backend: Optional[str] = None,
) -> torch.Tensor:
    """State-aware attention: Triton JIT on CUDA when available, else SDPA."""
    B, H, Tq, d_head = q.shape
    if scale is None:
        scale = 1.0 / math.sqrt(d_head)

    total_mask = _build_total_mask(
        q=q,
        q_state=q_state,
        k_state=k_state,
        level_proj_weight=level_proj_weight,
        temp=temp,
        alpha=alpha,
        gate_mode=gate_mode,
        use_discrete_levels=use_discrete_levels,
        lattice=lattice,
        causal=causal,
    )

    want_triton = (
        force_backend == "triton"
        or (
            force_backend is None
            and HAS_TRITON
            and TRITON_KERNEL_DEFINED
            and q.is_cuda
            and k.is_cuda
            and v.is_cuda
        )
    )

    if want_triton and force_backend not in ("sdpa", "manual"):
        try:
            out = _launch_triton_attn(q, k, v, total_mask, scale)
            _set_backend("triton")
            return out
        except Exception:
            pass

    if force_backend == "manual":
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        scores = scores + total_mask
        attn = torch.nan_to_num(F.softmax(scores, dim=-1), nan=0.0)
        _set_backend("manual")
        return torch.matmul(attn, v)

    try:
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=total_mask, scale=scale, is_causal=False
        )
        _set_backend("sdpa")
        return out
    except Exception:
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        scores = scores + total_mask
        attn = torch.nan_to_num(F.softmax(scores, dim=-1), nan=0.0)
        _set_backend("manual")
        return torch.matmul(attn, v)


class FusedTritonStateAttention(nn.Module):
    """API-compatible fused attention; Triton JIT on CUDA, SDPA otherwise."""

    def __init__(
        self,
        d_model: int = 128,
        state_dim: int = 8,
        num_heads: int = 8,
        temp: float = 1.0,
        alpha: float = 1.0,
        gate_mode: str = "hard",
        use_discrete_levels: bool = True,
        lattice: StateLattice = DEFAULT_LATTICE,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.state_dim = state_dim
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.scale = 1.0 / (self.d_head ** 0.5)
        self.temp = temp
        self.alpha = alpha
        self.gate_mode = gate_mode
        self.use_discrete_levels = use_discrete_levels
        self.lattice = lattice

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.level_proj = nn.Linear(state_dim, 1, bias=False)
        with torch.no_grad():
            self.level_proj.weight.zero_()
            self.level_proj.weight[0, 0] = 1.0

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        q = self.W_q(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)

        if mask is None:
            attn_out = triton_fused_state_attention(
                q=q, k=k, v=v,
                q_state=state, k_state=state,
                level_proj_weight=self.level_proj.weight,
                scale=self.scale, temp=self.temp, alpha=self.alpha,
                gate_mode=self.gate_mode,
                use_discrete_levels=self.use_discrete_levels,
                lattice=self.lattice,
                causal=True,
            )
        else:
            labels = state[..., 0].round().long().clamp(0, 5)
            state_mask = build_label_attention_mask(
                labels, labels, lattice=self.lattice, forbidden_value=float("-inf")
            ).to(dtype=q.dtype)
            causal_addon = torch.zeros_like(mask, dtype=q.dtype).masked_fill(
                mask == 0, float("-inf")
            )
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            scores = scores + state_mask + causal_addon
            attn = torch.nan_to_num(F.softmax(scores, dim=-1), nan=0.0)
            attn_out = torch.matmul(attn, v)
            _set_backend("manual")

        out = self.W_o(attn_out.transpose(1, 2).contiguous().view(B, T, C))
        return out, state
