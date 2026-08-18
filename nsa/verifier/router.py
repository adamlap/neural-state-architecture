"""
nsa.verifier.router
===================
StreamRouter: Compartmented Execution & Clearance-Based Multi-Stream Token Dispatch.

Part of the Trusted Computing Base (TCB) governing model-to-sink authorization:
    Model Output State (sigma_t) ==> Permitted Output Sink (Y_sink)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

import torch

from nsa.algebra import StateLabel


class StreamRouter:
    """Routes generated tokens to compartmented output channels based on runtime StateLabel.

    Allows the model to securely generate `SYSTEM` tokens intended exclusively for
    backend tools/APIs while sending `PUBLIC` tokens to the user's interface.
    """

    def __init__(
        self,
        tokenizer: Optional[Any] = None,
        default_sink: Optional[Callable[[str, int], None]] = None,
    ):
        self.tokenizer = tokenizer
        self.default_sink = default_sink
        self._sinks: Dict[int, List[Callable[[str, int], None]]] = {
            label.value: [] for label in StateLabel
        }
        self._buffers: Dict[int, List[int]] = {
            label.value: [] for label in StateLabel
        }
        self._history: List[Tuple[int, int]] = []  # (state_val, token_id) sequence

    def register_sink(
        self,
        state: Union[StateLabel, int],
        sink: Callable[[str, int], None],
    ) -> None:
        """Register a callback sink for a specific StateLabel level."""
        key = state.value if isinstance(state, StateLabel) else int(state)
        if key not in self._sinks:
            self._sinks[key] = []
        self._sinks[key].append(sink)

    def route_token(
        self,
        token: Union[torch.Tensor, int],
        current_state: Union[StateLabel, int],
    ) -> str:
        """Dispatch a single token to the appropriate state stream sink(s)."""
        state_val = (
            current_state.value
            if isinstance(current_state, StateLabel)
            else int(current_state)
        )

        if isinstance(token, torch.Tensor):
            token_id = (
                int(token.squeeze().item())
                if token.numel() == 1
                else int(token[0, 0].item())
            )
        else:
            token_id = int(token)

        self._buffers[state_val].append(token_id)
        self._history.append((state_val, token_id))

        token_str = ""
        if self.tokenizer is not None:
            token_str = self.tokenizer.decode([token_id], skip_special_tokens=False)

        # Dispatch to registered sinks
        sinks = self._sinks.get(state_val, [])
        for sink in sinks:
            sink(token_str, token_id)

        if self.default_sink is not None:
            self.default_sink(token_str, state_val)

        return token_str

    def rollback_tokens(self, drop_count: int) -> None:
        """Remove the last `drop_count` tokens from stream buffers during a rollback."""
        for _ in range(min(drop_count, len(self._history))):
            st_val, tok_id = self._history.pop()
            if self._buffers[st_val] and self._buffers[st_val][-1] == tok_id:
                self._buffers[st_val].pop()

    def get_stream_tokens(self, state: Union[StateLabel, int]) -> List[int]:
        """Return all token IDs routed to a specific security level."""
        state_val = state.value if isinstance(state, StateLabel) else int(state)
        return list(self._buffers.get(state_val, []))

    def get_stream_text(self, state: Union[StateLabel, int]) -> str:
        """Decode and return the aggregated text for a specific security level."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be configured on StreamRouter to decode text.")
        tokens = self.get_stream_tokens(state)
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def clear(self) -> None:
        """Clear all stream buffers."""
        for k in self._buffers:
            self._buffers[k].clear()
        self._history.clear()
