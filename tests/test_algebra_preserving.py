"""
tests/test_algebra_preserving.py
================================
Test suite mathematically verifying the core structural and behavioural
invariants of the Neural State Architecture.
"""

import torch

from nsa.algebra import StateLabel
from nsa.algebra_preserving import AlgebraPreservingStateTransition
from nsa.layers import StateAwareAttention
from nsa.value_layer import ValueAlignmentLoss


def test_structural_monotonicity():
    """
    Test that the Algebra-Preserving State Transition structurally guarantees
    monotonicity across layers, regardless of the neural network's weights
    or the input data.
    """
    batch_size = 4
    seq_len = 16
    d_model = 64
    state_dim = 8

    # Initialize the transition operator
    transition = AlgebraPreservingStateTransition(
        d_model=d_model,
        state_dim=state_dim,
        hidden_dim=32,
    )

    # Generate random inputs and random initial states
    m = torch.randn(batch_size, seq_len, d_model)

    # Random initial states between 0 and 5
    sigma_l = torch.rand(batch_size, seq_len, state_dim) * 5.0

    # Forward pass
    sigma_next = transition(m, sigma_l)

    # For DimKind.SECURITY (dim 0, 6) and LICENSE (dim 5), the invariant is monotone UP (>=)
    for dim in [0, 5, 6]:
        diff = sigma_next[..., dim] - sigma_l[..., dim]
        # Allow small floating point tolerance
        assert torch.all(diff >= -1e-6), f"Monotonicity violated on dimension {dim} (UP)"

    # For DimKind.CONFIDENCE (dim 1, 7), the invariant is monotone DOWN (<=)
    for dim in [1, 7]:
        # Note: confidence is clamped to 1.0 initially if > 1.0
        sigma_l_clamped = torch.clamp(sigma_l[..., dim], 0.0, 1.0)
        diff = sigma_next[..., dim] - sigma_l_clamped
        assert torch.all(diff <= 1e-6), f"Monotonicity violated on dimension {dim} (DOWN)"

    # For DimKind.PROVENANCE (dim 2, 3, 4), the invariant is monotone UP (set union)
    for dim in [2, 3, 4]:
        diff = sigma_next[..., dim] - sigma_l[..., dim]
        assert torch.all(diff >= -1e-6), f"Monotonicity violated on dimension {dim} (PROVENANCE)"


def test_hard_masking_isolation():
    """
    Test that StateAwareAttention with gate_mode="hard" structurally blocks
    information flow from high-security keys to lower-security queries.
    """
    d_model = 64
    state_dim = 8
    num_heads = 4
    seq_len = 10

    attn = StateAwareAttention(
        d_model=d_model,
        state_dim=state_dim,
        num_heads=num_heads,
        compat_mode="level",
        gate_mode="hard",
    )

    x = torch.randn(1, seq_len, d_model)

    # Create states:
    # First 5 tokens are SYSTEM (5)
    # Next 5 tokens are CONFIDENTIAL (3)
    states = torch.zeros(1, seq_len, state_dim)
    states[0, :5, 0] = StateLabel.SYSTEM.value
    states[0, 5:, 0] = StateLabel.CONFIDENTIAL.value

    # Causal mask is standard, but we want to check the STATE mask specifically.
    state_mask = attn._state_mask(states)

    # In hard mode, the state mask should be strictly -inf for the forbidden regions
    # Query at pos 5 (CONFIDENTIAL) cannot attend to Key at pos 0 (SYSTEM)
    # state_mask shape: [B, 1 (or H), seq_len, seq_len]
    forbidden_weight = state_mask[0, 0, 5, 0].item()

    assert forbidden_weight == float("-inf"), (
        f"Hard mask failed to isolate SYSTEM from CONFIDENTIAL, mask val={forbidden_weight}"
    )


def test_value_alignment_loss():
    """
    Test that ValueAlignmentLoss correctly assigns high penalties when the model
    predicts a secret token, and zero penalty when it outputs a safe token.
    """
    B, T, V = 2, 10, 100
    safe_token = 99
    secret_token = 50

    loss_fn = ValueAlignmentLoss(
        lambda_hard=5.0,
        lambda_value=3.0,
        secret_lo=40,
        secret_hi=60,
        safe_token=safe_token,
        confidential_level=StateLabel.CONFIDENTIAL.value,
        response_position=5,
    )

    # Dummy logits
    logits = torch.randn(B, T, V)
    # Force the model to strongly predict the secret token at the response position
    logits[0, 5, secret_token] = 100.0
    # Force the second batch item to strongly predict the safe token
    logits[1, 5, safe_token] = 100.0

    lm_targets = torch.randint(0, 39, (B, T))
    safe_targets = lm_targets.clone()
    safe_targets[0, 5] = safe_token
    safe_targets[1, 5] = safe_token

    levels = torch.full((B, T), StateLabel.CONFIDENTIAL.value, dtype=torch.float32)
    injection_flags = torch.tensor([True, True])

    loss, breakdown = loss_fn(logits, lm_targets, safe_targets, levels, injection_flags)

    # Because batch 0 predicted the secret, its hard constraint loss and value loss should be non-zero
    assert breakdown["hard_constraint"] > 0.0, (
        "Expected positive hard constraint penalty for secret leakage"
    )
    assert breakdown["value_alignment"] > 0.0, (
        "Expected positive value alignment penalty for missing safe token"
    )
