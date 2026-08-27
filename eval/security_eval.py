"""Compatibility entry point for the legacy NL security evaluation tests.

The NL red-team implementation now lives in ``prototype.security.nl_redteam_suite``.
This module intentionally contains no second implementation; it re-exports the
canonical suite so older tests and external scripts continue to work.
"""

from prototype.security.nl_redteam_suite import (  # noqa: F401
    ADVGLUE_STYLE_VARIANTS,
    ATTACK_CATALOGUE,
    AttackCase,
    build_role_labels,
    main,
    mask_firewall_score,
    run_advglue_label_consistency,
    run_hf_optional,
    run_mask_firewall,
    secret_leak_in_text,
)

__all__ = [
    "ADVGLUE_STYLE_VARIANTS",
    "ATTACK_CATALOGUE",
    "AttackCase",
    "build_role_labels",
    "main",
    "mask_firewall_score",
    "run_advglue_label_consistency",
    "run_hf_optional",
    "run_mask_firewall",
    "secret_leak_in_text",
]
