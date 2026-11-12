# HSF Comparative Experiment Protocol — v1.0

## Objective

Test whether HSF offers a useful associative-recall trade-off compared with
established vector-symbolic approaches. This stage tests the memory primitive,
not a cortex, synapse model, or cognition.

## Methods

- **HSF:** spatial key pattern outer-product-bound with a channel value code;
  associations are superposed into a fixed H × W × C tensor and recalled by
  matched-filter correlation.
- **HRR:** holographic reduced representations using circular convolution
  for binding, additive superposition, and correlation/unbinding.
- **HDC:** bipolar hyperdimensional vectors using elementwise multiplication
  for binding, additive bundling, and unbinding.
- **Exact dictionary:** perfect-recall control with explicit overwrite
  semantics; not a distributed representation.

All methods receive the same ordered synthetic key/value records, vocabulary,
seeds, and memory counts. The default task uses eight closed-set value labels.

## What this can establish

The experiment can compare synthetic recall accuracy under interference and
allocated numeric payloads under stated dimensions. It cannot establish
semantic generalisation, biological fidelity, learning from sensory input,
language quality, transformer-attention replacement, energy efficiency, or
general memory-capacity advantage.

## Fairness and accounting

The benchmark records NumPy numeric payload bytes. It excludes Python object
overhead, strings, allocator fragmentation, and process RSS. The explicit
vector-slot estimate assumes two float32 vectors of dimension D per record;
it is a transparent reference calculation, not a measured dictionary size.

HSF's tensor factorisation and flat HRR/HDC vectors have different inductive
biases and operation costs. Equal flattened dimensions make payload sizes
visible but do not make algorithms mathematically identical. The exact
dictionary is a quality upper bound and should later be measured using a
real serialised and in-memory implementation before deployment claims.

## Stage 1: independent random keys

Run:

~~~bash
pytest -q tests/test_holographic_state_field.py tests/test_holographic_associative_baselines.py
python -m experiments.holographic_state_field.comparative_benchmark --out results/hsf_comparison_v1.json
~~~

Record mean and minimum accuracy over seeds, accuracy as association count
rises, and persistent numeric payload. Do not select a winner from one seed
or one memory count.

## Stage 2: correlated keys (implemented; execution pending)

Independent hash-derived patterns are the easiest case. A more revealing test
must create controlled key similarity:
1. Generate one random bipolar prototype per family.
2. Generate a per-item random bipolar residual.
3. Form each key from a tunable mixture of prototype and residual, then
   normalise it.
4. Sweep prototype contribution from zero to high similarity while holding
   association count and value distribution fixed.
5. Query with exact and noisy cues.

The implementation is in
`experiments/holographic_state_field/correlated_key_benchmark.py`. It uses
four families and sweeps target within-family cosine values 0, 0.25, 0.5,
0.75, and 0.9 across multiple seeds and memory counts. Each family has a
shared bipolar prototype; individual keys flip coordinates with probability
(1 - sqrt(rho))/2, giving expected pairwise cosine rho. Value labels cycle
within each family so the experiment does not accidentally reduce to one
label per family.

Use the same family/item identities, value assignments, and target similarity
levels across methods. HSF keys have dimension H×W, while HRR/HDC keys have
dimension H×W×C. Method-specific vectors are generated from the same
family/item seeds and the realised within-family cosine is recorded for each
method. This matches the target similarity distribution, not the absolute
dimensional noise or computational cost; interpret results accordingly.
Do not simulate correlation by merely giving keys similar-looking strings.
The benchmark keeps keys distinct for rho<1.

Run:

~~~bash
pytest -q tests/test_holographic_correlated_keys.py
python -m experiments.holographic_state_field.correlated_key_benchmark \
  --out results/hsf_correlated_keys_v1.json
~~~

## Stage 4: noise and conflicting updates

After Stage 2, flip a controlled fraction of cue bits or perturb cue vectors.
Test repeated writes of the same association separately from conflicting
writes to one key. Define overwrite, reinforcement, and supersession as
distinct policies.

## Stage 4a: conflicting updates and error-correcting writes (implemented and run)

`experiments/holographic_state_field/supersession_benchmark.py` writes `n` keys, rewrites half of them with a
different value, and scores the latest value (updated accuracy), the old value (stale rate) and the keys that
were not rewritten (untouched accuracy). Equal float payload (16×16×16 HSF vs flat D=4096 HRR/HDC), 5 seeds.

Policies: each method's additive write; an index-free *cleanup* update for every method (retrieve, clean up to
the nearest vocabulary code, subtract it, add the new value, threshold calibrated per method); and for HSF the
delta-rule write `HolographicStateField.write_pattern_delta` (`F += K ⊗ (V − F·K)`), single pass and with the
final records replayed three times.

~~~bash
pytest -q tests/test_holographic_supersession.py
python -m experiments.holographic_state_field.supersession_benchmark --out results/hsf_supersession_v1.json
~~~

Result (`results/hsf_supersession_v1.json`; updated accuracy / stale rate / untouched accuracy, means of 5 seeds):

| rho | n | HSF additive | HSF cleanup | HSF delta | HSF delta + replay | HDC cleanup | HRR cleanup |
|---|---|---|---|---|---|---|---|
| 0.0 | 64 | 0.44 / 0.51 / 1.00 | 0.97 / 0.03 / 1.00 | 1.00 / 0.00 / 1.00 | 1.00 / 0.00 / 1.00 | 1.00 / 0.00 / 1.00 | 1.00 / 0.00 / 1.00 |
| 0.0 | 256 | 0.44 / 0.43 / 0.73 | 0.70 / 0.13 / 0.71 | 0.88 / 0.04 / 0.54 | 0.93 / 0.01 / 0.96 | 0.96 / 0.02 / 0.98 | 0.95 / 0.03 / 0.97 |
| 0.5 | 64 | 0.21 / 0.19 / 0.21 | 0.39 / 0.21 / 0.37 | 0.46 / 0.11 / 0.54 | **1.00 / 0.00 / 0.99** | 0.55 / 0.18 / 0.54 | 0.49 / 0.19 / 0.51 |
| 0.5 | 128 | 0.17 / 0.17 / 0.17 | 0.32 / 0.23 / 0.30 | 0.46 / 0.18 / 0.46 | **0.94 / 0.02 / 0.96** | 0.42 / 0.26 / 0.47 | 0.48 / 0.23 / 0.47 |
| 0.9 | 64 | 0.16 / 0.09 / 0.16 | 0.24 / 0.10 / 0.17 | 0.17 / 0.09 / 0.16 | 0.21 / 0.07 / 0.32 | 0.38 / 0.16 / 0.26 | 0.39 / 0.16 / 0.31 |

Reading:

* Additive superposition cannot supersede: the old and new traces tie, so the latest value is returned about half the
  time for every method. Expected, now measured.
* **Index-free supersession is not an HSF-specific advantage.** A calibrated cleanup-subtract update gives HDC and HRR the
  same supersession on independent keys, and they keep more of the untouched keys at high load (n=256: 0.98 vs 0.54 for
  the single-pass delta write, whose later writes perturb earlier ones as the load approaches the key dimension H×W).
* **The one clear HSF-specific result is decorrelation under replay.** At within-family cosine 0.5 the replayed delta rule
  recovers 0.94–1.00 on both updated and untouched keys where every index-free policy of every method is at 0.3–0.55. A
  flat bound vector cannot do this: for elementwise binding the residual write `m + k∘(v − k∘m)` equals `k∘v`, erasing all
  other associations. Replay needs the records again, so this is a property of *field + a replay buffer* (consolidation),
  not of the field alone.
* At cosine 0.9 nothing works, including three replay passes (Kaczmarz converges slowly on near-collinear cues).

Gate assessment: of the four gates below, the third ("useful update/forgetting property unavailable in simpler baselines")
is met only in the qualified form *iterative least-squares fitting of correlated cues under replay*. The first two are not met.
Next falsification steps: (1) replay-pass sweep and a recursive-least-squares write (exact, but it costs an (H·W)² matrix that
must be counted in the budget); (2) the same replay budget given to HDC/HRR as record re-insertion; (3) only if (1)–(2) hold,
connect the replay buffer to NSA's consolidation path, where records are available during "sleep".

## Decision gates

Continue developing HSF only if it demonstrates at least one repeatable:
- accuracy/capacity advantage at a comparable numeric budget;
- robustness advantage under correlated or noisy cues;
- useful update/forgetting property unavailable in simpler baselines; or
- architectural property valuable to NSA even without raw benchmark wins.

Otherwise retain the best established baseline and redirect effort. Do not add
neural-field dynamics, self-organising plasticity, or LLM integration until
these gates are evaluated.

## Reproducibility

Keep seeds, task generation, dimensions, software versions, and raw JSON
artifacts with every run. Never replace raw runs with only an averaged chart.
A failed or inconclusive result is a valid research outcome.


## Stage 2b: fixed-budget HSF factorisation (implemented; CI execution pending)

The allocation sweep holds H×W×C constant and varies the split between
spatial cue dimension H×W and value code dimension C. Flat HRR/HDC baselines
use the full flattened dimension for both key and value codes. Raw runs and
aggregates record these dimensions explicitly. This does not equalise compute
or the dimensions of the individual codes; it makes the HSF allocation
trade-off measurable rather than hiding it behind equal total element counts.

Run:

~~~bash
pytest -q tests/test_holographic_allocation_benchmark.py
python -m experiments.holographic_state_field.allocation_benchmark \
  --out results/hsf_allocation_v1.json
~~~

## Stage 3: unsupervised competitive Hebbian plasticity (implemented; CI execution pending)

A small competitive map is implemented independently of HSF and NSA runtime.
It uses winner/neighbour local updates toward each input followed by weight
normalisation. It receives vectors only; family labels are withheld from
learning and used only after training to score cluster-to-family mapping.
The benchmark compares the fixed random initial codebook with the learned
map on held-out examples from the same synthetic family-generating process.

Run:

~~~bash
pytest -q tests/test_holographic_plasticity.py
python -m experiments.holographic_state_field.plasticity_benchmark \
  --out results/hsf_plasticity_v1.json
~~~

Record held-out accuracy, training purity, occupied units, update count, and
prototype bytes. A positive result would show that this particular local
plasticity rule organises this synthetic family task; it would not establish
semantic generalisation or biological fidelity. A negative result is also
useful. Do not combine the map with HSF until the standalone result and
ablation controls justify doing so.


## CI execution policy

HSF tests are part of the normal `tests/` suite and therefore run through the
existing `.github/workflows/test.yml` workflow. The HSF research sweeps are
opt-in on manual workflow dispatch using `run_hsf_research_benchmarks`.
This avoids a separate push-triggered workflow and keeps long-running
scientific runs out of the automatic PR gate. Raw JSON outputs are uploaded
as a single `hsf-research-results` artifact.
