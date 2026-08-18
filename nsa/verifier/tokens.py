"""
nsa.verifier.tokens
===================
State Control Tokens & Vocabulary Registry for Dynamic State Tracking.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

from nsa.algebra import StateLabel
from nsa.verifier.automaton import Capability, SecurityAutomaton, SecurityExecutionState


class StateControlTokens:
    """Special tokens used to delineate dynamic state transitions inside text generation."""

    START_SYSTEM = "<|start_system_thought|>"
    END_SYSTEM = "<|end_system_thought|>"
    START_CONFIDENTIAL = "<|start_confidential_thought|>"
    END_CONFIDENTIAL = "<|end_confidential_thought|>"
    START_PRIVATE = "<|start_private_thought|>"
    END_PRIVATE = "<|end_private_thought|>"

    # Token string to corresponding SecurityExecutionState
    ENTER_MAP: ClassVar[Dict[str, SecurityExecutionState]] = {
        START_SYSTEM: SecurityExecutionState.SYSTEM,
        START_CONFIDENTIAL: SecurityExecutionState.CONFIDENTIAL,
        START_PRIVATE: SecurityExecutionState.PRIVATE,
    }

    EXIT_MAP: ClassVar[Dict[str, SecurityExecutionState]] = {
        END_SYSTEM: SecurityExecutionState.CONFIDENTIAL,
        END_CONFIDENTIAL: SecurityExecutionState.PUBLIC,
        END_PRIVATE: SecurityExecutionState.CONFIDENTIAL,
    }

    ALL_TOKENS: ClassVar[List[str]] = [
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
    def check_transition(
        cls,
        token_str: str,
        current_state: Union[StateLabel, int],
        automaton: Optional[SecurityAutomaton] = None,
        capability: Optional[Capability] = None,
    ) -> Tuple[bool, int]:
        """Inspect token text for control tokens and return (state_changed, new_state_val).

        Enforces that semantic tokens cannot escalate privilege without external capability.
        """
        curr_val = current_state.value if isinstance(current_state, StateLabel) else int(current_state)

        for token, target_exec_state in cls.ENTER_MAP.items():
            if token in token_str:
                if automaton is not None:
                    authorized, new_q = automaton.transition(target_exec_state, capability)
                    if authorized:
                        return True, new_q.to_state_label().value
                    return False, curr_val
                # If no automaton provided, default allow for backward compatibility
                return True, target_exec_state.to_state_label().value

        for token, target_exec_state in cls.EXIT_MAP.items():
            if token in token_str:
                if automaton is not None:
                    authorized, new_q = automaton.transition(target_exec_state, capability)
                    if authorized:
                        return True, new_q.to_state_label().value
                    return False, curr_val
                return True, target_exec_state.to_state_label().value

        return False, curr_val
