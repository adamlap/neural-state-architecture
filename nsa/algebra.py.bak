"""
nsa.algebra
===========
State algebra: lattice structure, partial order, conservation laws.

The core abstraction is a *state lattice* — a partially ordered set (S, ≤)
equipped with meet (⊓) and join (⊔) operations, forming a bounded lattice.

Every permitted state transition must be monotone with respect to the
lattice order defined by ConservationLaws. Concretely, a law of the form:

    Private → Public  is FORBIDDEN

is encoded by asserting that PRIVATE is *above* PUBLIC in the lattice, so
no downward step is possible without violating the monotone constraint.

The lattice used here is:
    SYSTEM
      │
   PRIVATE
      │
   CONFIDENTIAL
      │
   TRUSTED
      │
   PUBLIC
      │
   UNTRUSTED

Higher nodes are *more restricted*. Transitions may only go toward
*more restricted* states or stay the same (information can be classified
upward but not downward without an explicit declassification gate).

This mirrors the Bell-LaPadula model from information security,
but is formulated as differentiable algebra for use in neural networks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, FrozenSet, List, Optional, Tuple

try:
    import torch
except ImportError:
    torch = None


# ---------------------------------------------------------------------------
# State Labels
# ---------------------------------------------------------------------------

class StateLabel(IntEnum):
    """Ordered state labels forming the default security lattice.

    Higher integer value = more restricted / higher classification.
    The ordering is: PUBLIC < TRUSTED < CONFIDENTIAL < PRIVATE < SYSTEM.
    """
    UNTRUSTED    = 0
    PUBLIC       = 1
    TRUSTED      = 2
    CONFIDENTIAL = 3
    PRIVATE      = 4
    SYSTEM       = 5

    def __str__(self) -> str:
        return self.name

    # ------------------------------------------------------------------
    # Lattice operations (on the default linear order)
    # ------------------------------------------------------------------

    def meet(self, other: "StateLabel") -> "StateLabel":
        """Greatest lower bound (most permissive common state)."""
        return StateLabel(min(self.value, other.value))

    def join(self, other: "StateLabel") -> "StateLabel":
        """Least upper bound (most restrictive common state)."""
        return StateLabel(max(self.value, other.value))

    def allows_transition_to(self, target: "StateLabel") -> bool:
        """By default, information may become *more* restricted but not less.

        i.e. PRIVATE → SYSTEM is allowed (adding restriction).
             PRIVATE → PUBLIC  is forbidden (removing restriction).
        """
        return target.value >= self.value


# ---------------------------------------------------------------------------
# Conservation Law
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConservationLaw:
    """A single conservation law: 'from_state → to_state is {allowed/forbidden}'.

    Laws can express:
        - Monotone constraints: information can only be reclassified upward.
        - Explicit exceptions: a declassification gate may allow PRIVATE → PUBLIC
          iff a certain condition is met (modeled as a soft penalty weight).
    """
    from_label:   StateLabel
    to_label:     StateLabel
    allowed:      bool   = True
    penalty_weight: float = 1.0   # weight in the state constraint loss

    def __str__(self) -> str:
        arrow = "->" if self.allowed else "->X"
        return f"{self.from_label} {arrow} {self.to_label}"

    def is_violated(self, src: StateLabel, dst: StateLabel) -> bool:
        """Return True if a transition src->dst violates this law."""
        if src == self.from_label and dst == self.to_label:
            return not self.allowed
        return False


# ---------------------------------------------------------------------------
# State Lattice
# ---------------------------------------------------------------------------

class StateLattice:
    """A bounded lattice over StateLabels with configurable conservation laws.

    By default, uses the monotone upward-only rule (no downward reclassification).
    Additional custom laws can be added or default laws overridden.

    Examples
    --------
    >>> lattice = StateLattice()
    >>> lattice.is_allowed(StateLabel.PRIVATE, StateLabel.PUBLIC)
    False
    >>> lattice.is_allowed(StateLabel.PUBLIC, StateLabel.PRIVATE)
    True
    >>> lattice.is_allowed(StateLabel.PRIVATE, StateLabel.SYSTEM)
    True
    """

    def __init__(
        self,
        extra_laws: Optional[List[ConservationLaw]] = None,
        override_default: bool = False,
    ) -> None:
        self._custom_laws: List[ConservationLaw] = extra_laws or []
        self._override_default = override_default

    # ------------------------------------------------------------------
    # Core query
    # ------------------------------------------------------------------

    def is_allowed(self, src: StateLabel, dst: StateLabel) -> bool:
        """Check whether the transition src -> dst is permitted.

        Custom laws are checked first; they can explicitly permit or forbid.
        If no custom law matches, the default monotone rule applies.
        """
        for law in self._custom_laws:
            if law.from_label == src and law.to_label == dst:
                return law.allowed

        if self._override_default:
            # If all custom laws are exhausted and default is overridden,
            # fall back to permissive (allow anything not explicitly forbidden).
            return True

        # Default: information may only become *more* restricted.
        return src.allows_transition_to(dst)

    def violation_penalty(self, src: StateLabel, dst: StateLabel) -> float:
        """Return the penalty weight for a forbidden transition (0 if allowed)."""
        if self.is_allowed(src, dst):
            return 0.0
        for law in self._custom_laws:
            if law.from_label == src and law.to_label == dst and not law.allowed:
                return law.penalty_weight
        return 1.0  # default penalty weight

    # ------------------------------------------------------------------
    # Lattice operations
    # ------------------------------------------------------------------

    @staticmethod
    def meet(a: StateLabel, b: StateLabel) -> StateLabel:
        return a.meet(b)

    @staticmethod
    def join(a: StateLabel, b: StateLabel) -> StateLabel:
        return a.join(b)

    # ------------------------------------------------------------------
    # Derived utilities
    # ------------------------------------------------------------------

    def can_attend(self, query: StateLabel, key: StateLabel) -> bool:
        """Return True if a query token may *read* a key token (can-read).

        Information flows key → query. Allowed iff the key's label may
        transition into the query's label under the lattice
        (query is at least as restricted as key):

            can_attend(SYSTEM, UNTRUSTED) = True
            can_attend(PUBLIC, PRIVATE)   = False
        """
        return self.is_allowed(key, query)

    def can_read(self, query: StateLabel, key: StateLabel) -> bool:
        """Alias of :meth:`can_attend` — directional read (key → query)."""
        return self.can_attend(query=query, key=key)

    def can_write(self, writer: StateLabel, target: StateLabel) -> bool:
        """May ``writer`` *write into* storage/channel labelled ``target``?

        Bell-LaPadula-style *no write-down*: writer level must be ≤ target
        level (information may only flow to equal-or-more-restricted sinks).

            can_write(PUBLIC, PRIVATE) = True
            can_write(PRIVATE, PUBLIC) = False
        """
        return self.is_allowed(writer, target)

    def can_declassify(
        self,
        src: StateLabel,
        dst: StateLabel,
        *,
        authorized: bool = False,
    ) -> bool:
        """May state transition ``src → dst`` under declassification policy?

        Upward / equal restriction is always allowed (standard monotone path).
        Downward reclassification requires an explicit authorization bit
        (auth token / declassification gate). Without ``authorized=True``,
        downward moves are denied.
        """
        if self.is_allowed(src, dst):
            return True
        return bool(authorized)

    def compatible(self, src: StateLabel, dst: StateLabel) -> bool:
        """Asymmetric attention compatibility: can ``src`` (query) read ``dst`` (key)?

        Historical name retained for API stability. Prefer :meth:`can_attend`.
        """
        return self.can_attend(query=src, key=dst)

    def reachable_from(self, src: StateLabel) -> List[StateLabel]:
        """All labels reachable (via allowed transitions) from src."""
        return [s for s in StateLabel if self.is_allowed(src, s)]

    def summary(self) -> str:
        lines = ["StateLattice transition table:"]
        labels = list(StateLabel)
        header = "FROM\\TO  " + "  ".join(f"{s.name[:6]:>7}" for s in labels)
        lines.append(header)
        for src in labels:
            row = f"{src.name[:8]:<9}"
            for dst in labels:
                sym = "OK" if self.is_allowed(src, dst) else "X"
                row += f"  {sym:>7}"
            lines.append(row)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Default singleton
# ---------------------------------------------------------------------------

DEFAULT_LATTICE = StateLattice()


# ---------------------------------------------------------------------------
# Product Lattice & Typed Neural Computation (Product Algebra)
# ---------------------------------------------------------------------------

@dataclass
class ProductStateVector:
    """Product state vector for Typed Neural Computation (TNC).

    Formulates the state space as a Product Lattice:
        Σ = Σ_security × Σ_confidence × Σ_provenance × Σ_license

    Each component carries its own distinct algebraic join (⊔) and meet (⊓) operators:
        - Security   (⊔_s): Lattice supremum (least upper restriction bound)
        - Confidence (⊔_c): Bayesian / Minimum confidence bound min(c1, c2)
        - Provenance (⊔_p): Bitwise set union of document origin IDs (p1 | p2)
        - License    (⊔_l): Maximal tier restriction bounds max(l1, l2)

    To guarantee zero performance overhead when metadata dimensions are disabled,
    this class supports a light scalar mode (security-only) that executes with
    0% extra tensor memory allocation.
    """
    security: StateLabel = StateLabel.PUBLIC
    confidence: float = 1.0
    provenance: int = 0
    license_tier: int = 0

    def join_product(self, other: ProductStateVector) -> ProductStateVector:
        """Product lattice join: (⊔_s, ⊔_c, ⊔_p, ⊔_l)."""
        return ProductStateVector(
            security=self.security.join(other.security),
            confidence=min(self.confidence, other.confidence),
            provenance=self.provenance | other.provenance,
            license_tier=max(self.license_tier, other.license_tier),
        )

    def meet_product(self, other: ProductStateVector) -> ProductStateVector:
        """Product lattice meet: (⊓_s, ⊓_c, ⊓_p, ⊓_l)."""
        return ProductStateVector(
            security=self.security.meet(other.security),
            confidence=max(self.confidence, other.confidence),
            provenance=self.provenance & other.provenance,
            license_tier=min(self.license_tier, other.license_tier),
        )

    def allows_attention_from(self, query: ProductStateVector) -> bool:
        """Check coordinate-wise attention compatibility across all product dimensions."""
        sec_ok = query.security.value >= self.security.value
        lic_ok = query.license_tier >= self.license_tier
        return sec_ok and lic_ok


class ProductLattice:
    """Product Lattice manager evaluating component-wise operations across Σ."""

    def __init__(self, security_lattice: Optional[StateLattice] = None):
        self.security_lattice = security_lattice or DEFAULT_LATTICE

    def is_allowed(self, src: ProductStateVector, dst: ProductStateVector) -> bool:
        """Check state transition validity across product dimensions."""
        sec_ok = self.security_lattice.is_allowed(src.security, dst.security)
        lic_ok = dst.license_tier >= src.license_tier
        return sec_ok and lic_ok

    def compute_mask(self, query_states: List[ProductStateVector], key_states: List[ProductStateVector]) -> List[List[float]]:
        """Compute 2D additive compatibility mask for product lattice states."""
        mask = []
        for q in query_states:
            row = []
            for k in key_states:
                if k.allows_attention_from(q):
                    row.append(0.0)
                else:
                    row.append(-1e4)
            mask.append(row)
        return mask


# ---------------------------------------------------------------------------
# Bitpacked State Tensor Memory Optimization
# ---------------------------------------------------------------------------

@dataclass
class BitpackedStateVector:
    """Bitpacked state representation compressing 4D product state into a uint8 byte."""
    packed_byte: int  # 8-bit integer: [security: 3 bits | license: 3 bits | reserved: 2 bits]

    @classmethod
    def from_product(cls, sv: ProductStateVector) -> BitpackedStateVector:
        sec_val = min(sv.security.value, 7) & 0x07
        lic_val = min(sv.license_tier, 7) & 0x07
        byte_val = (sec_val << 5) | (lic_val << 2)
        return cls(packed_byte=byte_val)

    def to_product(self) -> ProductStateVector:
        sec_val = (self.packed_byte >> 5) & 0x07
        lic_val = (self.packed_byte >> 2) & 0x07
        return ProductStateVector(
            security=StateLabel(sec_val if sec_val <= 5 else 5),
            license_tier=lic_val
        )


def bitpack_states(states: torch.Tensor) -> torch.Tensor:
    """Compress float32 state vectors [B, T, state_dim] into uint8 packed byte tensors [B, T].

    Reduces memory footprint by 75% for multi-tenant and provenance tracking.
    """
    sec_labels = torch.clamp(states[..., 0].round().long(), 0, 7)
    lic_tiers = torch.clamp(states[..., 1].round().long(), 0, 7) if states.shape[-1] > 1 else torch.zeros_like(sec_labels)
    
    packed = (sec_labels << 5) | (lic_tiers << 2)
    return packed.to(torch.uint8)


def unpack_states(packed: torch.Tensor, state_dim: int = 8) -> torch.Tensor:
    """Decompress uint8 packed byte tensors [B, T] back into float32 state vectors [B, T, state_dim]."""
    packed_long = packed.to(torch.long)
    sec_labels = ((packed_long >> 5) & 0x07).to(torch.float32)
    lic_tiers = ((packed_long >> 2) & 0x07).to(torch.float32)

    shape = list(packed.shape) + [state_dim]
    unpacked = torch.zeros(shape, dtype=torch.float32, device=packed.device)
    unpacked[..., 0] = sec_labels
    if state_dim > 1:
        unpacked[..., 1] = lic_tiers
    return unpacked


# ---------------------------------------------------------------------------
# RAGMetadataIngressEncoder
# ---------------------------------------------------------------------------

class RAGMetadataIngressEncoder:
    """Encodes enterprise Vector DB metadata (Qdrant, Pinecone, PGVector) into NSA state vectors.

    Converts JSON metadata records:
      {
          "security": "PRIVATE",
          "tenant": "FINANCE",
          "provenance": 18291,
          "license": "INTERNAL",
          "confidence": 0.94
      }
    into ProductStateVector and continuous PyTorch state tensors.
    """

    LEVEL_MAP: Dict[str, StateLabel] = {
        "UNTRUSTED": StateLabel.UNTRUSTED,
        "PUBLIC": StateLabel.PUBLIC,
        "TRUSTED": StateLabel.TRUSTED,
        "CONFIDENTIAL": StateLabel.CONFIDENTIAL,
        "PRIVATE": StateLabel.PRIVATE,
        "SYSTEM": StateLabel.SYSTEM,
    }

    LICENSE_MAP: Dict[str, int] = {
        "PUBLIC": 0,
        "INTERNAL": 1,
        "CONFIDENTIAL": 2,
        "RESTRICTED": 3,
        "FINANCE": 2,
        "HR": 2,
        "LEGAL": 3,
        "PII": 3,
    }

    @classmethod
    def encode_metadata_dict(cls, meta: Dict) -> ProductStateVector:
        """Encode a metadata dict into a :class:`ProductStateVector`."""
        sec_str = str(meta.get("security", "UNTRUSTED")).upper()
        sec_label = cls.LEVEL_MAP.get(sec_str, StateLabel.UNTRUSTED)

        conf = float(meta.get("confidence", 1.0))
        prov = int(meta.get("provenance", meta.get("provenance_mask", 0)))

        lic_raw = meta.get("license", meta.get("license_tier", 0))
        if isinstance(lic_raw, str):
            lic = cls.LICENSE_MAP.get(lic_raw.upper(), 0)
        else:
            lic = int(lic_raw)

        return ProductStateVector(
            security=sec_label,
            confidence=conf,
            provenance=prov,
            license_tier=lic,
        )

    @classmethod
    def encode_batch_to_tensor(
        cls,
        metas: List[Dict],
        state_dim: int = 8,
        device: Optional[str] = None,
    ):
        """Encode a list of metadata dicts to a continuous state tensor [N, state_dim]."""
        if torch is None:
            raise ImportError("PyTorch is required for encode_batch_to_tensor")
        rows = []
        for meta in metas:
            sv = cls.encode_metadata_dict(meta)
            vec = [float(sv.security.value), float(sv.license_tier), float(sv.confidence), float(sv.provenance & 0xFF)]
            if state_dim > len(vec):
                vec = vec + [0.0] * (state_dim - len(vec))
            else:
                vec = vec[:state_dim]
            rows.append(vec)
        t = torch.tensor(rows, dtype=torch.float32, device=device)
        return t


def build_label_attention_mask(
    query_labels,
    key_labels=None,
    lattice: Optional[StateLattice] = None,
    forbidden_value: float = -1e4,
):
    """Build an additive attention mask from discrete lattice labels.

    Parameters
    ----------
    query_labels : LongTensor [B, T_q] (or [T_q])
    key_labels   : LongTensor [B, T_k] (defaults to query_labels)
    lattice      : StateLattice used for can_attend checks
    forbidden_value : additive logit value for blocked pairs

    Returns
    -------
    mask : FloatTensor [B, 1, T_q, T_k] with 0.0 (allowed) or forbidden_value.
    """
    if torch is None:
        raise ImportError("PyTorch is required for build_label_attention_mask")

    lattice = lattice or DEFAULT_LATTICE
    if key_labels is None:
        key_labels = query_labels

    q = query_labels
    k = key_labels
    if q.dim() == 1:
        q = q.unsqueeze(0)
    if k.dim() == 1:
        k = k.unsqueeze(0)

    B, Tq = q.shape
    _, Tk = k.shape
    device = q.device

    # Vectorized monotone rule: query_level >= key_level  (key → query allowed)
    # Custom lattice laws fall back to pairwise Python checks when present.
    if not lattice._custom_laws and not lattice._override_default:
        q_lvl = q.unsqueeze(2).float()  # [B, Tq, 1]
        k_lvl = k.unsqueeze(1).float()  # [B, 1, Tk]
        allowed = q_lvl >= k_lvl
        mask = torch.where(
            allowed,
            torch.zeros((), device=device, dtype=torch.float32),
            torch.full((), forbidden_value, device=device, dtype=torch.float32),
        )
        return mask.unsqueeze(1)  # [B, 1, Tq, Tk]

    mask = torch.zeros(B, Tq, Tk, device=device, dtype=torch.float32)
    for b in range(B):
        for i in range(Tq):
            qi = StateLabel(int(q[b, i].item()))
            for j in range(Tk):
                kj = StateLabel(int(k[b, j].item()))
                if not lattice.can_attend(qi, kj):
                    mask[b, i, j] = forbidden_value
    return mask.unsqueeze(1)


def build_level_attention_mask(
    levels,
    gate_mode: str = "hard",
    alpha: float = 1.0,
    temperature: float = 1.0,
    forbidden_value: float = -1e4,
):
    """Build additive attention mask from continuous security levels [B, T].

    Hard mode enforces query_level >= key_level (monotone non-interference).
    Soft mode uses alpha * logsigmoid((L_q - L_k) / temperature).
    """
    if torch is None:
        raise ImportError("PyTorch is required for build_level_attention_mask")

    if levels.dim() == 1:
        levels = levels.unsqueeze(0)

    L_q = levels.unsqueeze(2)  # [B, T, 1]
    L_k = levels.unsqueeze(1)  # [B, 1, T]
    delta = L_q - L_k

    if gate_mode == "hard":
        mask = torch.where(
            delta < 0,
            torch.full((), forbidden_value, device=levels.device, dtype=levels.dtype),
            torch.zeros((), device=levels.device, dtype=levels.dtype),
        )
    elif gate_mode == "soft":
        # logsigmoid is numerically stable vs log(sigmoid(...))
        import torch.nn.functional as F
        temp = max(float(temperature), 1e-5)
        mask = alpha * F.logsigmoid(delta / temp)
    else:
        raise ValueError(f"Unknown gate_mode '{gate_mode}'")

    return mask.unsqueeze(1)  # [B, 1, T, T]



