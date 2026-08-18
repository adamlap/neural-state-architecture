import torch
from torch import nn


class StateEncoderHead(nn.Module):
    """
    Lightweight bidirectional encoder (or MLP head) that maps token representations/embeddings
    to state distributions. Supports asynchronous execution on a dedicated CUDA stream.
    """

    def __init__(self, hidden_size: int, num_states: int, use_bidirectional: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_states = num_states
        self.use_bidirectional = use_bidirectional

        if self.use_bidirectional:
            # Simple Transformer Encoder for bidirectional context
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=4, dim_feedforward=hidden_size * 4, batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        else:
            self.encoder = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, num_states)
        )

        # Dedicated stream for parallel execution if on CUDA
        self.stream = torch.cuda.Stream() if torch.cuda.is_available() else None

    def forward(self, hidden_states: torch.Tensor, async_execution: bool = True) -> torch.Tensor:
        """
        Forward pass.
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            async_execution: Whether to use the dedicated CUDA stream
        Returns:
            Logits of shape [batch_size, seq_len, num_states]
        """
        if async_execution and self.stream is not None:
            with torch.cuda.stream(self.stream):
                return self._forward_impl(hidden_states)
        else:
            return self._forward_impl(hidden_states)

    def _forward_impl(self, hidden_states: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(hidden_states)
        logits = self.classifier(encoded)
        return logits

    def synchronize(self):
        """Wait for the asynchronous execution to finish."""
        if self.stream is not None:
            self.stream.synchronize()
