# NSA Holographic State Field (HSF) — v0.1

## Research question

Can many associations be represented as overlapping patterns in one fixed,
multidimensional tensor, with retrieval performed by correlation and
reconstruction rather than scanning an explicit key/value memory table?

## Representation

The field is a multi-channel image:

\[
F \in \mathbb{R}^{H \times W \times C}
\]

A key deterministically generates a normalized spatial pattern
\(K(k) \in \mathbb{R}^{H \times W}\). A value generates a normalized channel
code \(V(v) \in \mathbb{R}^{C}\). Writing superposes their outer product:

\[
F \leftarrow F + \alpha\, K(k) \otimes V(v)
\]

Retrieval correlates the query key with the field:

\[
\hat{V}(q) = \sum_{x=1}^{H}\sum_{y=1}^{W} K(q)_{xy} F_{xy:}
\]

The reconstructed vector is decoded against a fixed value vocabulary using
dot-product similarity.

This is a software analogue of distributed associative memory, not a claim
that a literal optical hologram or biological process has been recreated. It
is a matched-filter model with expected cross-talk as memories overlap.

## Why this first experiment?

It isolates the core hypothesis from language-model extraction, tokenisation,
transformers, and hardware. If the field cannot retrieve synthetic facts under
controlled interference, connecting it to an LLM would obscure the result.

The field stores no per-record key/value table. Key patterns are regenerated
from the key string and a seed; the value codebook is fixed at construction.
The tensor size therefore stays constant as writes accumulate. This trades
unbounded record storage for finite representational capacity and interference.

## Run

Install research dependencies:

~~~bash
pip install -e ".[dev,research]"
pytest -q tests/test_holographic_state_field.py
python -m experiments.holographic_state_field.benchmark --out results/hsf_v1.json
~~~

Default sweep: 10 seeds × 7 memory loads (8 through 512 associations). It
reports retrieval accuracy, minimum per-seed accuracy, fixed field bytes, an
explicit vector-slot baseline, and their ratio. Byte counts describe NumPy
payloads, not Python object overhead or process RSS.

## Limitations

- Passing tests establish implementation properties, not a breakthrough.
- The exact dictionary baseline is expected to retrieve perfectly; HSF is
  evaluated on the cost/accuracy trade-off as interference increases.
- The fixed vocabulary makes decoding easier than open-ended semantic recall.
- Key patterns are deterministic random projections; no learned semantic
  geometry exists yet.
- Repeated writes reinforce; uniform decay models forgetting. There is no
  selective unbinding or automatic supersession in v0.1.
- This does not replace transformer attention or measure language quality,
  GPU memory, latency, or energy.

## Next falsification steps

1. Sweep field dimensions and vocabulary sizes to map capacity curves.
2. Add key similarity and correlated-value adversarial tests.
3. Compare with random-projection and holographic reduced representation
   baselines (circular convolution/correlation).
4. Test update/supersession and selective forgetting without an external index.
5. Only then connect a compact field to NSA memory/cognition interfaces.


## Comparative research track

The comparative implementation adds two reference associative memories:
- HRR: circular-convolution binding and correlation-based unbinding.
- HDC/VSA: bipolar elementwise binding and additive bundling.
- An exact dictionary provides a perfect-recall quality control.

Run the controlled comparison:

~~~bash
pytest -q tests/test_holographic_state_field.py tests/test_holographic_associative_baselines.py
python -m experiments.holographic_state_field.comparative_benchmark --out results/hsf_comparison_v1.json
~~~

The comparison defaults to a 32 × 32 × 64 HSF tensor and flat HRR/HDC
vectors with the same flattened dimension. HRR/HDC regenerate deterministic
key/value codes, so only their persistent superposition vector is counted.
HSF includes its fixed channel codebook. The comparison protocol documents
the remaining representation and accounting caveats, as well as the next
correlated-key experiment:

- [HSF comparative experiment protocol](../../docs/HSF_COMPARATIVE_EXPERIMENT_PROTOCOL.md)


## Correlated-key interference experiment

This next-stage experiment generates families of similar bipolar key patterns
with a controlled expected within-family cosine similarity. It compares HSF,
HRR, and HDC over several similarity levels, memory loads, and random seeds.

~~~bash
pytest -q tests/test_holographic_correlated_keys.py
python -m experiments.holographic_state_field.correlated_key_benchmark \
  --out results/hsf_correlated_keys_v1.json
~~~

HSF's cue dimension is the spatial plane (H×W); HRR/HDC use the flattened
H×W×C dimension. The benchmark reports realised cosine for each method.
This deliberately tests each representation's response to the same target
similarity, while acknowledging that their key dimensions and computation
costs differ. See the comparative protocol for interpretation.
