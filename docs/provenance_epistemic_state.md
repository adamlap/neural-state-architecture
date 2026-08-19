# NSA Provenance and Epistemic State

Provenance records the lineage of claims and observations so that confidence is not detached from evidence.

A record is:

$$
P=(id,type,parents,evidence,producer)
$$

Evidence is:

$$
E=(id,source,kind,reliability,time)
$$

The first implementation is immutable and append-oriented.

## Design principles

- provenance is not truth;
- source identity is not reliability;
- model confidence must not become evidence by itself;
- derived claims retain links to parent claims;
- provenance cannot silently be overwritten;
- provenance must remain separate from authority.

This enables future epistemic operations such as confidence calibration, contradiction detection, source weighting and evidence-aware state transitions.

## Planned integration

The next layer should connect provenance to the canonical state and self-state:

$$
Evidence \rightarrow Claim \rightarrow Confidence \rightarrow SelfState
$$

while preserving the distinction:

$$
\text{epistemic confidence} \neq \text{authorization}
$$
