"""
nsa.verifier.router
===================
StreamRouter: Compartmented Execution & Clearance-Based Multi-Stream Token Dispatch.

Part of the Trusted Computing Base (TCB) governing model-to-sink authorization:
    Model Output State (sigma_t) ==> Permitted Output Sink (Y_sink)
    Invariant: Route(x, sink) is permitted iff sigma_x <= Clearance(sink)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from nsa.algebra import StateLabel


class StreamRouter:
    """Security-aware router dispatching tokens to compartmented output sinks based on clearance.

    Enforces that high-security data (e.g. SYSTEM / PRIVATE) cannot be routed to low-clearance
    sinks (e.g. PUBLIC user chat).
    """

    def __init__(
        self,
        tokenizer: Optional[Any] = None,
        default_sink: Optional[Callable[[str, int], None]] = None,
        default_sink_clearance: Union[StateLabel, int] = StateLabel.PUBLIC,
    ):
        self.tokenizer = tokenizer
        self.default_sink = default_sink
        self.default_sink_clearance = (
            default_sink_clearance.value
            if isinstance(default_sink_clearance, StateLabel)
            else int(default_sink_clearance)
        )
        # Sinks format: {state_key: [(callback, max_clearance)]}
        self._sinks: Dict[int, List[Tuple[Callable[[str, int], None], int]]] = {
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
        max_clearance: Optional[Union[StateLabel, int]] = None,
    ) -> None:
        """Register a callback sink for a specific StateLabel level with clearance bounds.

        Args:
            state: Primary StateLabel level to subscribe to.
            sink: Callback function receiving (token_text, token_id).
            max_clearance: Maximum data clearance permitted to enter this sink.
                           Defaults to the channel state itself.
        """
        key = state.value if isinstance(state, StateLabel) else int(state)
        clearance = (
            (max_clearance.value if isinstance(max_clearance, StateLabel) else int(max_clearance))
            if max_clearance is not None
            else key
        )
        if key not in self._sinks:
            self._sinks[key] = []
        self._sinks[key].append((sink, clearance))

    def can_route(self, data_state: Union[StateLabel, int], sink_clearance: Union[StateLabel, int]) -> bool:
        """Predicate checking whether data_state can flow into sink_clearance without leak."""
        st_val = data_state.value if isinstance(data_state, StateLabel) else int(data_state)
        cl_val = sink_clearance.value if isinstance(sink_clearance, StateLabel) else int(sink_clearance)
        return st_val <= cl_val

    def route_token(
        self,
        token: Union[torch.Tensor, int],
        current_state: Union[StateLabel, int],
    ) -> str:
        """Dispatch a single token to authorized state stream sink(s).

        Enforces Clearance(sink) >= sigma_data.
        """
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

        # Dispatch to registered sinks iff clearance is respected
        sinks = self._sinks.get(state_val, [])
        for sink_fn, clearance in sinks:
            if self.can_route(state_val, clearance):
                sink_fn(token_str, token_id)

        # Dispatch to default sink iff default clearance is respected
        if self.default_sink is not None and self.can_route(state_val, self.default_sink_clearance):
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

    def snapshot(self) -> Tuple[List[Tuple[int, int]], Dict[int, List[int]]]:
        """Snapshot internal stream history and buffers for atomic rollback."""
        return (list(self._history), {k: list(v) for k, v in self._buffers.items()})

    def restore(self, snap: Tuple[List[Tuple[int, int]], Dict[int, List[int]]]) -> None:
        """Restore internal stream history and buffers."""
        hist, bufs = snap
        self._history = list(hist)
        self._buffers = {k: list(v) for k, v in bufs.items()}

    def clear(self) -> None:
        """Clear all stream buffers."""
        for k in self._buffers:
            self._buffers[k].clear()
        self._history.clear()
