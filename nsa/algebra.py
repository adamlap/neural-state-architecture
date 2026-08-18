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

import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, FrozenSet, Generic, List, Optional, TypeVar

try:
    import torch
except ImportError:
    torch = None


# ---------------------------------------------------------------------------
# State Labels
# ---------------------------------------------------------------------------


class LatticeEnum(IntEnum):
    def __str__(self) -> str:
        return self.name

    def meet(self, other: LatticeEnum) -> LatticeEnum:
        return self.__class__(min(self.value, other.value))

    def join(self, other: LatticeEnum) -> LatticeEnum:
        return self.__class__(max(self.value, other.value))

    def allows_transition_to(self, target: LatticeEnum) -> bool:
        return target.value >= self.value


class ConfidentialityLabel(LatticeEnum):
    """Ordered confidentiality labels forming the primary security lattice."""

    UNTRUSTED = 0
    PUBLIC = 1
    TRUSTED = 2
    CONFIDENTIAL = 3
    PRIVATE = 4
    SYSTEM = 5


class IntegrityLabel(LatticeEnum):
    """Ordered integrity labels (taint). Higher value = more untrusted."""

    TRUSTED = 0
    UNTRUSTED = 1


# Backward compatibility alias
StateLabel = ConfidentialityLabel


# ---------------------------------------------------------------------------
# Conservation Law
# ---------------------------------------------------------------------------

T = TypeVar("T", bound=LatticeEnum)


@dataclass(frozen=True)
class ConservationLaw(Generic[T]):
    """A single conservation law: 'from_state → to_state is {allowed/forbidden}'.

    Laws can express:
        - Monotone constraints: information can only be reclassified upward.
        - Explicit exceptions: a declassification gate may allow PRIVATE → PUBLIC
          iff a certain condition is met (modeled as a soft penalty weight).
    """

    from_label: T
    to_label: T
    allowed: bool = True
    penalty_weight: float = 1.0  # weight in the state constraint loss

    def __str__(self) -> str:
        arrow = "->" if self.allowed else "->X"
        return f"{self.from_label} {arrow} {self.to_label}"

    def is_violated(self, src: T, dst: T) -> bool:
        """Return True if a transition src->dst violates this law."""
        if src == self.from_label and dst == self.to_label:
            return not self.allowed
        return False


# ---------------------------------------------------------------------------
# Typed Declassification Capability
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeclassificationCapability(Generic[T]):
    """A typed cryptographic-style capability primitive authorizing downward state transitions.

    D: (σ, c_D) → σ' where Valid(c_D, σ, σ') = 1
    """

    issuer: str
    purpose: str
    scope: str
    expiry: float
    max_downgrade: T

    def validate(self, src: T, dst: T, current_time: Optional[float] = None) -> bool:
        """Evaluate Valid(c_D, σ, σ') to mathematically permit the downgrade."""
        now = time.time() if current_time is None else current_time
        # Strict expiry invariant: now > expiry => invalid (now <= expiry is valid)
        if now > self.expiry:
            return False
        # The downgrade target (dst) cannot be less restricted than max_downgrade.
        # Assuming lower value = less restricted:
        if dst.value < self.max_downgrade.value:
            return False
        return True


# ---------------------------------------------------------------------------
# Transition Operator V ∈ T_Σ
# ---------------------------------------------------------------------------

if torch is not None:

    def project_transition_matrix(V: torch.Tensor, monotone: bool = True) -> torch.Tensor:
        """Exact algebraic projection P_{T_Sigma}(V) onto legal lower-triangular state transitions.

        Under the multiplication convention sigma' = sigma @ V.T (sigma'_j = sum_i sigma_i * V_{j, i}):
            row index j = destination (dst)
            column index i = source (src)
            Legal transition dst >= src (row >= col) -> LOWER TRIANGULAR.

        Guarantees:
            1. P(V) in T_Sigma (for all dst < src, P(V)[dst, src] == 0.0)
            2. Idempotence: P(P(V)) == P(V)
            3. Non-negative diagonal: P(V)[i, i] >= 0.0
        """
        if not monotone:
            return V
        V_tril = torch.tril(V)
        diag = V_tril.diagonal().clamp(min=0.0)
        return V_tril - torch.diag(V_tril.diagonal()) + torch.diag(diag)

    class TransitionOperator(torch.nn.Module):
        """
        A state transition matrix V ∈ T_Σ restricted by architectural lower-triangular projection.
        Illegal transitions are mathematically unrepresentable by projection.
        """

        def __init__(self, d_state: int, valid_transition_mask: Optional[torch.Tensor] = None):
            super().__init__()
            self.d_state = d_state
            # Initialize close to identity (stay in current state)
            self.weight = torch.nn.Parameter(
                torch.eye(d_state) + torch.randn(d_state, d_state) * 0.01
            )
            if valid_transition_mask is None:
                valid_transition_mask = torch.tril(torch.ones(d_state, d_state))
            self.register_buffer("legal_mask", valid_transition_mask.float())

        def get_projected_weight(self) -> torch.Tensor:
            return project_transition_matrix(self.weight * self.legal_mask)

        def forward(self, sigma_h: torch.Tensor) -> torch.Tensor:
            """Apply V to sigma_h, guaranteeing V ∈ T_Σ: sigma' = sigma @ V.T."""
            constrained_V = self.get_projected_weight()
            return torch.matmul(sigma_h, constrained_V.t())


# ---------------------------------------------------------------------------
# State Lattice
# ---------------------------------------------------------------------------


class StateLattice(Generic[T]):
    """A bounded lattice over generic LatticeEnum with configurable conservation laws."""

    def __init__(
        self,
        extra_laws: Optional[List[ConservationLaw[T]]] = None,
        override_default: bool = False,
    ) -> None:
        self._custom_laws: List[ConservationLaw[T]] = extra_laws or []
        self._override_default = override_default

    # ------------------------------------------------------------------
    # Core query
    # ------------------------------------------------------------------

    def is_allowed(self, src: T, dst: T) -> bool:
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

    def violation_penalty(self, src: T, dst: T) -> float:
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
    def meet(a: T, b: T) -> T:
        return a.meet(b)

    @staticmethod
    def join(a: T, b: T) -> T:
        return a.join(b)

    # ------------------------------------------------------------------
    # Derived utilities
    # ------------------------------------------------------------------

    def can_attend(self, query: T, key: T) -> bool:
        """Return True if a query token may *read* a key token (can-read).

        Information flows key → query. Allowed iff the key's label may
        transition into the query's label under the lattice
        (query is at least as restricted as key):
        """
        return self.is_allowed(key, query)

    def can_read(self, query: T, key: T) -> bool:
        """Alias of :meth:`can_attend` — directional read (key → query)."""
        return self.can_attend(query=query, key=key)

    def can_write(self, writer: T, target: T) -> bool:
        """May ``writer`` *write into* storage/channel labelled ``target``?

        Bell-LaPadula-style *no write-down*: writer level must be ≤ target
        level (information may only flow to equal-or-more-restricted sinks).
        """
        return self.is_allowed(writer, target)

    def can_declassify(
        self,
        src: T,
        dst: T,
        *,
        capability: Optional[DeclassificationCapability[T]] = None,
        current_time: Optional[float] = None,
    ) -> bool:
        """May state transition ``src → dst`` under declassification policy?

        Upward / equal restriction is always allowed (standard monotone path).
        Downward reclassification requires an explicit authorization capability.
        """
        if self.is_allowed(src, dst):
            return True
        if capability is not None and capability.validate(src, dst, current_time=current_time):
            return True
        return False

    def compatible(self, src: T, dst: T) -> bool:
        """Asymmetric attention compatibility: can ``src`` (query) read ``dst`` (key)?

        Historical name retained for API stability. Prefer :meth:`can_attend`.
        """
        return self.can_attend(query=src, key=dst)

    def reachable_from(self, src: T) -> List[T]:
        """All labels reachable (via allowed transitions) from src."""
        return [s for s in src.__class__ if self.is_allowed(src, s)]

    def summary(self, enum_cls) -> str:
        lines = ["StateLattice transition table:"]
        labels = list(enum_cls)
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
# Default singletons
# ---------------------------------------------------------------------------

DEFAULT_LATTICE = StateLattice[ConfidentialityLabel]()
INTEGRITY_LATTICE = StateLattice[IntegrityLabel]()


# ---------------------------------------------------------------------------
# Product Lattice & Typed Neural Computation (Product Algebra)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HardStateVector:
    """Hard trusted policy state: Sigma_h = Sigma_C x Sigma_I x Sigma_A x Sigma_L."""

    confidentiality: ConfidentialityLabel = ConfidentialityLabel.PUBLIC
    integrity: IntegrityLabel = IntegrityLabel.TRUSTED
    authorization: Optional[str] = None
    license_tier: int = 0

    def join(self, other: HardStateVector) -> HardStateVector:
        """Component-wise hard lattice join: (⊔_C, ⊔_I, ⊔_A, ⊔_L)."""
        return HardStateVector(
            confidentiality=self.confidentiality.join(other.confidentiality),
            integrity=self.integrity.join(other.integrity),
            authorization=self.authorization or other.authorization,
            license_tier=max(self.license_tier, other.license_tier),
        )

    def meet(self, other: HardStateVector) -> HardStateVector:
        """Component-wise hard lattice meet: (⊓_C, ⊓_I, ⊓_A, ⊓_L)."""
        return HardStateVector(
            confidentiality=self.confidentiality.meet(other.confidentiality),
            integrity=self.integrity.meet(other.integrity),
            authorization=self.authorization if self.authorization == other.authorization else None,
            license_tier=min(self.license_tier, other.license_tier),
        )


@dataclass(frozen=True)
class SoftStateVector:
    """Soft operational risk state: Sigma_s = Sigma_U x Sigma_R."""

    uncertainty: float = 0.0  # Epistemic/semantic uncertainty (0.0 = certain)
    risk: float = 0.0  # Risk score (0.0 = zero risk)

    def join(self, other: SoftStateVector) -> SoftStateVector:
        """Worst-case operational composition (join takes maximum uncertainty & risk)."""
        return SoftStateVector(
            uncertainty=max(self.uncertainty, other.uncertainty),
            risk=max(self.risk, other.risk),
        )

    def meet(self, other: SoftStateVector) -> SoftStateVector:
        """Best-case operational meet."""
        return SoftStateVector(
            uncertainty=min(self.uncertainty, other.uncertainty),
            risk=min(self.risk, other.risk),
        )


def join_hard_state_tensors(sigma_h1: torch.Tensor, sigma_h2: torch.Tensor) -> torch.Tensor:
    """Authoritative tensor product join on hard state: maximum across coordinates."""
    return torch.maximum(sigma_h1, sigma_h2)


def join_soft_state_tensors(sigma_s1: torch.Tensor, sigma_s2: torch.Tensor) -> torch.Tensor:
    """Authoritative tensor product join on soft state: minimum confidence."""
    return torch.minimum(sigma_s1, sigma_s2)


@dataclass
class ProductStateVector:
    """Product state vector for Typed Neural Computation (TNC).

    Formulates the state space as a Product Lattice:
        Σ = Σ_confidentiality × Σ_integrity × Σ_confidence × Σ_provenance × Σ_license

    Each component carries its own distinct algebraic join (⊔) and meet (⊓) operators.
    """

    confidentiality: ConfidentialityLabel = ConfidentialityLabel.PUBLIC
    integrity: IntegrityLabel = IntegrityLabel.TRUSTED
    confidence: float = 1.0
    provenance: FrozenSet[str] = field(default_factory=frozenset)
    license_tier: int = 0

    @property
    def security(self):
        """Backward compatibility alias."""
        return self.confidentiality

    def join_product(self, other: ProductStateVector) -> ProductStateVector:
        """Product lattice join: (⊔_c, ⊔_i, ⊔_c, ⊔_p, ⊔_l)."""
        return ProductStateVector(
            confidentiality=self.confidentiality.join(other.confidentiality),
            integrity=self.integrity.join(other.integrity),
            confidence=min(self.confidence, other.confidence),
            provenance=self.provenance | other.provenance,
            license_tier=max(self.license_tier, other.license_tier),
        )

    def meet_product(self, other: ProductStateVector) -> ProductStateVector:
        """Product lattice meet: (⊓_c, ⊓_i, ⊓_c, ⊓_p, ⊓_l)."""
        return ProductStateVector(
            confidentiality=self.confidentiality.meet(other.confidentiality),
            integrity=self.integrity.meet(other.integrity),
            confidence=max(self.confidence, other.confidence),
            provenance=self.provenance & other.provenance,
            license_tier=min(self.license_tier, other.license_tier),
        )

    def allows_attention_from(self, query: ProductStateVector) -> bool:
        """Check coordinate-wise attention compatibility across all product dimensions."""
        conf_ok = query.confidentiality.value >= self.confidentiality.value
        int_ok = query.integrity.value >= self.integrity.value
        lic_ok = query.license_tier >= self.license_tier
        return conf_ok and int_ok and lic_ok


class ProductLattice:
    """Product Lattice manager evaluating component-wise operations across Σ."""

    def __init__(
        self,
        conf_lattice: Optional[StateLattice[ConfidentialityLabel]] = None,
        int_lattice: Optional[StateLattice[IntegrityLabel]] = None,
    ):
        self.conf_lattice = conf_lattice or DEFAULT_LATTICE
        self.int_lattice = int_lattice or INTEGRITY_LATTICE

    def is_allowed(self, src: ProductStateVector, dst: ProductStateVector) -> bool:
        """Check state transition validity across product dimensions."""
        conf_ok = self.conf_lattice.is_allowed(src.confidentiality, dst.confidentiality)
        int_ok = self.int_lattice.is_allowed(src.integrity, dst.integrity)
        lic_ok = dst.license_tier >= src.license_tier
        return conf_ok and int_ok and lic_ok

    def compute_mask(
        self, query_states: List[ProductStateVector], key_states: List[ProductStateVector]
    ) -> List[List[float]]:
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

    packed_byte: (
        int  # 8-bit integer: [confidentiality: 3 bits | integrity: 2 bits | license: 3 bits]
    )

    @classmethod
    def from_product(cls, sv: ProductStateVector) -> BitpackedStateVector:
        conf_val = min(sv.confidentiality.value, 7) & 0x07
        int_val = min(sv.integrity.value, 3) & 0x03
        lic_val = min(sv.license_tier, 7) & 0x07
        byte_val = (conf_val << 5) | (int_val << 3) | lic_val
        return cls(packed_byte=byte_val)

    def to_product(self) -> ProductStateVector:
        conf_val = (self.packed_byte >> 5) & 0x07
        int_val = (self.packed_byte >> 3) & 0x03
        lic_val = self.packed_byte & 0x07
        return ProductStateVector(
            confidentiality=ConfidentialityLabel(min(conf_val, 4)),
            integrity=IntegrityLabel(min(int_val, 1)),
            license_tier=lic_val,
        )


def bitpack_states(states: torch.Tensor) -> torch.Tensor:
    """Compress float32 state vectors [B, T, state_dim] into uint8 packed byte tensors [B, T].

    Reduces memory footprint by 75% for multi-tenant and provenance tracking.
    """
    conf_labels = torch.clamp(states[..., 0].round().long(), 0, 7)
    int_labels = (
        torch.clamp(states[..., 1].round().long(), 0, 3)
        if states.shape[-1] > 1
        else torch.zeros_like(conf_labels)
    )
    lic_tiers = (
        torch.clamp(states[..., 2].round().long(), 0, 7)
        if states.shape[-1] > 2
        else torch.zeros_like(conf_labels)
    )

    packed = (conf_labels << 5) | (int_labels << 3) | lic_tiers
    return packed.to(torch.uint8)


def unpack_states(packed: torch.Tensor, state_dim: int = 8) -> torch.Tensor:
    """Decompress uint8 packed byte tensors [B, T] back into float32 state vectors [B, T, state_dim]."""
    packed_long = packed.to(torch.long)
    conf_labels = ((packed_long >> 5) & 0x07).to(torch.float32)
    int_labels = ((packed_long >> 3) & 0x03).to(torch.float32)
    lic_tiers = (packed_long & 0x07).to(torch.float32)

    shape = list(packed.shape) + [state_dim]
    unpacked = torch.zeros(shape, dtype=torch.float32, device=packed.device)
    unpacked[..., 0] = conf_labels
    if state_dim > 1:
        unpacked[..., 1] = int_labels
    if state_dim > 2:
        unpacked[..., 2] = lic_tiers
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

    LEVEL_MAP: Dict[str, ConfidentialityLabel] = {
        "UNTRUSTED": ConfidentialityLabel.PUBLIC,  # Map old untrusted to public
        "PUBLIC": ConfidentialityLabel.PUBLIC,
        "TRUSTED": ConfidentialityLabel.TRUSTED,
        "CONFIDENTIAL": ConfidentialityLabel.CONFIDENTIAL,
        "PRIVATE": ConfidentialityLabel.PRIVATE,
        "SYSTEM": ConfidentialityLabel.SYSTEM,
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
        conf_str = str(meta.get("confidentiality", meta.get("security", "PUBLIC"))).upper()
        conf_label = cls.LEVEL_MAP.get(conf_str, ConfidentialityLabel.PUBLIC)

        int_str = str(meta.get("integrity", "TRUSTED")).upper()
        int_label = IntegrityLabel.TRUSTED if int_str == "TRUSTED" else IntegrityLabel.UNTRUSTED

        conf = float(meta.get("confidence", 1.0))

        prov_raw = meta.get("provenance", [])
        if isinstance(prov_raw, (int, str)):
            prov = frozenset([str(prov_raw)])
        else:
            prov = frozenset(str(p) for p in prov_raw)

        lic_raw = meta.get("license", meta.get("license_tier", 0))
        if isinstance(lic_raw, str):
            lic = cls.LICENSE_MAP.get(lic_raw.upper(), 0)
        else:
            lic = int(lic_raw)

        return ProductStateVector(
            confidentiality=conf_label,
            integrity=int_label,
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
            prov_hash = hash(sv.provenance) & 0xFF if sv.provenance else 0
            vec = [
                float(sv.confidentiality.value),
                float(sv.integrity.value),
                float(sv.license_tier),
                float(sv.confidence),
                float(prov_hash),
            ]
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
