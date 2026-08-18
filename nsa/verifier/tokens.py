"""
nsa.verifier.tokens
===================
State Control Tokens & Vocabulary Registry for Dynamic State Tracking.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from nsa.algebra import StateLabel


class StateControlTokens:
    """Special tokens used to delineate dynamic state transitions inside text generation."""

    START_SYSTEM = "<|start_system_thought|>"
    END_SYSTEM = "<|end_system_thought|>"
    START_CONFIDENTIAL = "<|start_confidential_thought|>"
    END_CONFIDENTIAL = "<|end_confidential_thought|>"
    START_PRIVATE = "<|start_private_thought|>"
    END_PRIVATE = "<|end_private_thought|>"

    # Token string to corresponding StateLabel transition
    ENTER_MAP: Dict[str, StateLabel] = {
        START_SYSTEM: StateLabel.SYSTEM,
        START_CONFIDENTIAL: StateLabel.CONFIDENTIAL,
        START_PRIVATE: StateLabel.PRIVATE,
    }

    EXIT_MAP: Dict[str, StateLabel] = {
        END_SYSTEM: StateLabel.CONFIDENTIAL,
        END_CONFIDENTIAL: StateLabel.PUBLIC,
        END_PRIVATE: StateLabel.CONFIDENTIAL,
    }

    ALL_TOKENS: List[str] = [
        START_SYSTEM,
        END_SYSTEM,
        START_CONFIDENTIAL,
        END_CONFIDENTIAL,
        START_PRIVATE,
        END_PRIVATE,
    ]

    @classmethod
    def register(cls, tokenizer: Any, model: Optional[Any] = None) -> int:
        """Register state control tokens into a HuggingFace tokenizer.

        Optionally resizes the model's token embeddings to match the new vocab size.
        Returns the number of added tokens.
        """
        if tokenizer is None:
            return 0
        added = tokenizer.add_tokens(cls.ALL_TOKENS)
        if added > 0 and model is not None and hasattr(model, "resize_token_embeddings"):
            model.resize_token_embeddings(len(tokenizer))
        return added

    @classmethod
    def check_transition(cls, token_str: str, current_state: int) -> Tuple[bool, int]:
        """Inspect token text for control tokens and return (state_changed, new_state_val)."""
        for token, target_label in cls.ENTER_MAP.items():
            if token in token_str:
                return True, target_label.value
        for token, target_label in cls.EXIT_MAP.items():
            if token in token_str:
                return True, target_label.value
        return False, current_state
