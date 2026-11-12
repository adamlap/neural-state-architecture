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

## Stage 4b: falsification of 4a, whitened writes and pattern separation (implemented and run)

`experiments/holographic_state_field/consolidation_benchmark.py`, same protocol as 4a (overall accuracy, 5 seeds,
16×16×16 field, flat D=4096 unless stated). Raw data: `results/hsf_consolidation_v1.json`.

| cosine | n | HSF delta+replay | HDC replay | HRR replay | HSF whitened (69,632 floats) | HDC replay, 69,632 floats | HSF separated+replay | HDC separated+replay |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 64 | 0.99 | 0.98 | 0.97 | 1.00 | 0.98 | 1.00 | 1.00 |
| 0.5 | 256 | 0.70 | 0.84 | 0.87 | 1.00 | 0.97 | 0.75 | 0.94 |
| 0.9 | 64 | 0.27 | 0.34 | 0.29 | **1.00** | 0.46 | 0.72 | 0.79 |
| 0.9 | 256 | 0.20 | 0.24 | 0.21 | **1.00** | 0.36 | 0.34 | 0.61 |

1. **The Stage 4a "HSF-specific" result is falsified.** Given the same three replay passes (error-driven re-insertion), HDC and
   HRR recover correlated keys as well as the replayed delta rule at cosine 0.5 and better at high load. The earlier gap came
   from giving replay to HSF only.
2. **The whitened write is the first result the flat memories do not match.** Keeping the inverse cue covariance `P` and writing
   the residual along `P·k` gives exact recall and in-place supersession for *every* similarity level tested, in one pass with no
   replay, up to n = H·W. At the same float budget (field + P = 69,632) HDC with replay reaches 0.36–0.46 at cosine 0.9.
   Two limits, both measured: (a) capacity is hard: past n = H·W accuracy falls to 0.73 (n=384) and 0.59 (n=512), although it
   stays similarity-independent (0.74 at cosine 0.9, where budget-matched HDC is at 0.14); (b) **at n = H·W the budget equals an
   explicit table** of n (key, value) vector pairs (n·(H·W + C) floats), which also recalls perfectly. Within capacity the
   whitened field is therefore not a compression win; what it offers is fixed size, no key storage or scan, exact supersession
   and similarity-independent degradation past capacity.
3. **Pattern separation helps every method, HDC most.** A fixed random expansion with 5 % k-winners-take-all lowers
   within-family cosine from 0.89 to 0.58 (0.50 → 0.27) and lifts replayed accuracy at cosine 0.9 from ~0.3 to 0.72–0.91 at
   n ≤ 64. It is not HSF-specific.

Gate assessment after 4b: capacity gate not met; robustness gate met **only** by the whitened write and only by paying
(H·W)² floats; update gate met in the same form. Plain HSF, replayed or not, has no advantage over HDC/HRR.

Brain-inspired reading (analogy, not a biological claim): the three ingredients that worked are the ones a hippocampal-cortical
account names - sparse pattern separation (dentate gyrus), error-driven plasticity, and decorrelation of cues (what `P` computes;
in cortex attributed to lateral inhibition / anti-Hebbian learning). The open research question is whether decorrelation can be
had *cheaply*: `P` is a full matrix here. Candidates, in order: (i) whiten in the sparse separated space, where the cue covariance
is close to diagonal, so a diagonal or banded `P` may suffice; (ii) a low-rank `P` tracking only the family prototypes;
(iii) a local anti-Hebbian rule that learns `P` online. Each must beat budget-matched HDC with separation and replay, which is now
the baseline to beat.

## Stage 4c: cheap decorrelation - separation + homeostatic writes + replay (implemented and run)

Candidate (i) from 4b, in its simplest form: keep one usage count per key unit (H·W extra floats, counted) and scale each write
by `1 / (floor + usage)`, so units that are active for many cues learn less and a write lands on the units that distinguish the
cue. This is a diagonal stand-in for `P` (analogy: homeostatic / activity-dependent plasticity). Because HSF's limit is its key
dimension, the field is also reshaped to 16×32×8 (512 key units, same 4,096 field floats).

Overall accuracy mean / worst seed, 5 seeds, `results/hsf_consolidation_wide_v1.json`:

| cosine | n | HSF sep+homeo+replay (4,608 floats) | HSF sep+delta+replay (4,096) | HDC sep+replay (4,096) | HDC replay (4,096) |
|---|---|---|---|---|---|
| 0.9 | 32 | **0.99 / 0.94** | 0.94 / 0.91 | 0.91 / 0.81 | 0.29 / 0.25 |
| 0.9 | 64 | **0.97 / 0.86** | 0.85 / 0.77 | 0.79 / 0.69 | 0.34 / 0.17 |
| 0.9 | 128 | **0.91 / 0.76** | 0.68 / 0.56 | 0.71 / 0.62 | 0.24 / 0.18 |
| 0.9 | 256 | 0.69 / 0.55 | 0.45 / 0.38 | 0.61 / 0.57 | 0.24 / 0.20 |
| 0.5 | 256 | 0.97 / 0.86 | 0.93 / 0.82 | 0.94 / 0.93 | 0.84 / 0.80 |

This is the first configuration in which an HSF variant beats the strongest baseline at a near-equal budget: at cosine 0.9 it
leads HDC with separation and replay by 0.08–0.20 for n ≤ 256. Caveats that must travel with the number: 12.5 % more floats
(the usage vector); one shape, one sparsity (5 %), one floor, none tuned for either side; C = 8 equals the vocabulary size, so
this shape has no room for a larger vocabulary; at cosine 0.5 HDC is as good and more stable across seeds (worst seed 0.93 vs
0.86); replay re-presents all records; the homeostatic rule has no obvious counterpart for elementwise binding, so HDC did not
get one. On the 16×16×16 shape the same HSF variant only ties HDC (0.69 vs 0.71 at n = 128) and loses at n = 256.

Gate assessment after 4c: the robustness gate is met for highly similar keys by *pattern separation + homeostatic delta writes +
replay on a wide key plane*, pending replication (more seeds, sparsity/shape sweeps for both methods, larger vocabularies).
Next: (1) that replication; (2) replay from a bounded buffer instead of all records; (3) only then an NSA integration test in
which the key is a fact key and the value a vocabulary-free code.

## Stage 4d: replication with the baseline tuned (implemented and run)

`replication_sweep.py` gives **both** methods their own tuning: separator sparsity in {2, 5, 10, 20 %} for each, and for HSF
the field shape in {16×16×16, 16×32×8} at a fixed 4,096 field floats (32×32×4 is excluded because it has fewer channels than the
8-value vocabulary). The best configuration per method is chosen on selection seeds 1-5 and then evaluated on 10 *different*
seeds, so the reported numbers are not selected on their own data. Raw: `results/hsf_replication_v1.json`.

| cosine | n | HSF best config | HSF mean (worst) | HDC mean (worst) | paired seeds HSF / tie / HDC |
|---|---|---|---|---|---|
| 0.9 | 64 | 16×32×8, 2 % | 0.99 (0.95) | 0.79 (0.64) | 10 / 0 / 0 |
| 0.9 | 128 | 16×32×8, 2 % | 0.93 (0.83) | 0.72 (0.63) | 10 / 0 / 0 |
| 0.5 | 64 | 16×16×16, 2 % | 1.00 (1.00) | 1.00 (1.00) | 0 / 10 / 0 |
| 0.5 | 128 | 16×16×16, 2 % | 1.00 (0.99) | 0.97 (0.92) | 9 / 1 / 0 |

The Stage 4c advantage **replicates** at cosine 0.9 (HSF better on all 10 held-out seeds, +0.20-0.21 mean) and the cosine-0.5
disadvantage seen in 4c disappears once sparsity is tuned for both (sparser, 2 %, is best for both methods). It still is not a
statement about the real world: HSF pays H·W extra floats, HDC has no homeostatic counterpart, only n ≤ 128 and one key family
structure were swept, replay re-presents every record, and the task is a closed 8-value vocabulary.

Gate assessment after 4d: the robustness gate is met for correlated keys, with replication, for this synthetic task.
Next: bounded-buffer replay; a larger vocabulary (which needs more channels, shifting the shape trade-off); then the NSA
integration test.

## Stage 4e: bounded replay buffer (implemented and run)

Real consolidation does not replay everything. `run_one(..., replay_fraction=f)` replays a random subset of the records (chosen
without regard to which were updated). Cosine 0.9, 16×32×8, 2 % sparsity (the 4d configuration), the 10 held-out seeds of 4d,
overall accuracy mean (`results/hsf_bounded_replay_v1.json`):

| n | replay fraction | HSF sep+homeo+replay | HDC sep+replay |
|---|---|---|---|
| 64 | 1/16 | 0.62 | 0.37 |
| 64 | 1/4 | 0.75 | 0.43 |
| 64 | 1/2 | 0.84 | 0.59 |
| 64 | all | 0.99 | 0.79 |
| 128 | 1/16 | 0.63 | 0.35 |
| 128 | 1/4 | 0.66 | 0.46 |
| 128 | 1/2 | 0.74 | 0.57 |
| 128 | all | 0.93 | 0.72 |

HSF's lead survives a bounded buffer at every size tested (+0.17 to +0.28), but both methods degrade monotonically as the buffer
shrinks: with 1/4 of the records replayed, HSF at n=128 is no better (0.66) than HDC with a full buffer (0.72). The practical reading
is that most of the accuracy still comes from replay volume; the field's advantage is that each replayed record is worth more.
Buffer selection here is random; prioritised replay (replay the records with the largest recent residual, as hippocampal replay is
biased toward surprising events) is the obvious next test and is not yet implemented.

## Stage 4f: prioritised replay (implemented and run)

`run_one(..., replay_policy="priority")`: every replay pass re-ranks all records by current prediction error (HSF: distance between
the target value code and what the field returns; HDC: wrong recalls first, then smallest margin) and replays the worst
`fraction × n`. Cosine 0.9, 16×32×8, 2 % sparsity, 10 held-out seeds, overall accuracy mean (`results/hsf_priority_replay_v1.json`):

| n | buffer | HSF random | HSF priority | HDC random | HDC priority |
|---|---|---|---|---|---|
| 64 | 1/16 | 0.62 | 0.78 | 0.37 | 0.47 |
| 64 | 1/4 | 0.75 | 0.86 | 0.43 | 0.55 |
| 64 | 1/2 | 0.84 | 0.98 | 0.59 | 0.76 |
| 128 | 1/16 | 0.63 | 0.62 | 0.35 | 0.40 |
| 128 | 1/8 | 0.67 | 0.60 | 0.36 | 0.39 |
| 128 | 1/4 | 0.66 | 0.73 | 0.46 | 0.45 |
| 128 | 1/2 | 0.74 | 0.89 | 0.57 | 0.56 |

Prioritising by error helps both methods at n = 64 (HSF +0.10 to +0.14, HDC +0.10 to +0.17) and helps HSF at n = 128 once the buffer
is a quarter of the records or more (+0.07, +0.15). With a very small buffer at n = 128 it does not help and for HSF is slightly worse
(0.60 vs 0.67 at 1/8): the worst-error records are the hardest, mutually interfering ones, and replaying only those can over-fit them.
HDC gains nothing from prioritisation at n = 128 above 1/8. HSF's lead over HDC persists under both policies (priority at 1/2, n = 128:
0.89 vs 0.56).

## Stage 4g: larger value vocabularies (implemented and run)

HSF needs C >= V channels for V values, which comes out of the key plane (capacity); a flat HDC vector pays no such split.
`vocabulary_sweep.py` compares equal float budgets (4,096 and 16,384), HSF with C in {V, 2V} and a square-ish key plane, sparsity
chosen per method on selection seeds, cosine 0.9, 10 disjoint evaluation seeds (`results/hsf_vocabulary_v1.json`). Mean accuracy
(paired seed wins HSF/HDC):

| V | budget | n | HSF (shape) | HDC | wins |
|---|---|---|---|---|---|
| 8 | 4,096 | 64 | 0.99 (16×32×8) | 0.79 | 10 / 0 |
| 8 | 16,384 | 64 | 1.00 (32×32×16) | 0.88 | 9 / 0 |
| 32 | 4,096 | 32 | 0.83 (8×16×32) | 0.93 | 2 / 8 |
| 32 | 4,096 | 64 | 0.65 | 0.79 | 0 / 10 |
| 32 | 16,384 | 64 | **1.00** (16×32×32) | 0.95 | 8 / 0 |
| 128 | 4,096 | 32 | 0.38 (4×8×128) | 0.93 | 0 / 10 |
| 128 | 16,384 | 32 | 0.85 (8×16×128) | 0.98 | 0 / 9 |
| 128 | 16,384 | 64 | 0.67 | 0.95 | 0 / 10 |

Reading: the HSF advantage is conditional on the vocabulary being small relative to the budget. It holds for V = 8 at both budgets,
holds for V = 32 only when the budget is 4× larger, and reverses at V = 128 at both budgets (the key plane shrinks to 32-128 cells, so
capacity falls below the record count). The cause is structural, not a tuning artefact: a V-sized one-hot-like value code spends
V of the H·W·C floats per cell on the value side. Two fixes are untested: (i) a *distributed value code* (random codes of dimension
C << V decoded by cleanup against a codebook, as HRR/HDC already do) so C stops scaling with V; (ii) hierarchical values (a small
field per attribute). Neither belongs in an NSA integration until measured, and an NSA memory whose values are open-ended strings
would need (i) at minimum.

Gate assessment after 4f/4g: HSF's robustness advantage for correlated keys is real but confined to small closed vocabularies, so for
NSA's open-ended fact values it is **not yet usable**; the next experiment is the distributed value code at V = 128-1024.

## Stage 4h: distributed (short) value codes at large vocabularies (implemented and run)

The Stage 4g failure was caused by forcing C >= V. The field's value codes are random C-dimensional vectors for any vocabulary size and
decoding is nearest-code against the vocabulary, so C can stay short. `vocabulary_sweep.py --distributed` lets HSF choose C in
{16, 32, 64} (shape and nothing else selected on selection seeds 1-5) for V in {128, 512, 1024}, against HDC (D = budget), sparsity
2 % for both, cosine 0.9, n = 64, 10 disjoint evaluation seeds (`results/hsf_distributed_vocabulary_v1.json`):

| V | budget | HSF (shape) | HDC | paired seeds HSF / HDC |
|---|---|---|---|---|
| 128 | 4,096 | 0.88 (16×16×16) | 0.76 | 10 / 0 |
| 128 | 16,384 | 1.00 (32×32×16) | 0.95 | 9 / 0 |
| 512 | 4,096 | 0.85 (16×16×16) | 0.69 | 10 / 0 |
| 512 | 16,384 | 1.00 (32×32×16) | 0.94 | 9 / 0 |
| 1024 | 4,096 | 0.81 (16×16×16) | 0.62 | 10 / 0 |
| 1024 | 16,384 | 0.99 (32×32×16) | 0.92 | 10 / 0 |

Short codes remove the vocabulary penalty: C = 16 was selected every time and accuracy degrades only slowly with V (0.88 → 0.81 at
4,096 floats, 1.00 → 0.99 at 16,384). Only about 64 distinct values occur per episode while the decoder must discriminate among all V,
so the decoding task is the harder one.

Caveats that must travel with this result:

1. **Decoding needs a codebook of candidate values.** An HSF recall returns a C-dimensional vector that is matched against a list of known
   value codes. For open-ended facts that list is a set of the value strings themselves (or their hashes to strings). Both HSF and HDC
   regenerate codes deterministically from a seed and neither counts that value store in its float budget; an NSA deployment must keep
   it. What the field then replaces is the key → value association table, not the values.
2. Decoding cost is O(V·C) for HSF against O(V·D) for flat HDC at the same budget (HSF is cheaper here).
3. Same synthetic correlated-key task as before, replay of every record, H·W extra usage floats for HSF.

Gate assessment after 4h: the robustness advantage for correlated keys now holds for large closed vocabularies with short codes, so the
representation question no longer blocks an NSA integration test. Remaining blockers, in order: (1) real keys (token overlap rather than
synthetic cosines: do NSA's "entity attribute" keys behave like correlated families after pattern separation?); (2) an open-vocabulary value
store; (3) the replay buffer must be the NSA episodic log; (4) a comparison against NSA's current exact `TemporalMemoryStore` (the dictionary
that has perfect recall), where the case for HSF can only be fixed size, graceful degradation and in-field supersession.

## Stage 5a: NSA's real fact keys (implemented and run)

`nsa_keys_benchmark.py` uses the normalised "entity attribute" keys NSA's extractor produces (10 entities × 8 attributes, e.g.
"admin billing email"). A cue is the normalised sum of random codes of the key's tokens, so correlation comes from shared words.
Values are opaque strings with a hashed short code and an open-vocabulary store of the strings seen. Half the keys are rewritten, the
records replayed, 10 held-out seeds (`results/hsf_nsa_keys_v1.json`). Accuracy mean (worst seed):

| keys | HSF additive | HSF sep+delta | HSF sep+homeo+replay | HDC sep+replay | exact dictionary |
|---|---|---|---|---|---|
| 40, 4,096 floats | 0.17 | 0.81 | **1.00 (1.00)** | 1.00 (0.97) | 1.00 |
| 80, 4,096 floats | 0.07 | 0.53 | **1.00 (1.00)** | 0.96 (0.91) | 1.00 |
| 80, 16,384 floats | 0.08 | 0.72 | 1.00 (1.00) | 1.00 (1.00) | 1.00 |

Findings:

1. **Real keys are not the hard regime.** Mean pairwise key cosine is 0.10-0.11 (not 0.9), and separation barely changes it (0.10 → 0.10).
   The correlated-family advantage measured in Stage 4 does not describe NSA's keys.
2. **The HSF advantage over HDC mostly vanishes**: 1.00 vs 1.00, and 1.00 vs 0.96 at 80 keys in the smallest budget (worst seed 1.00 vs 0.91).
3. **What matters for real keys is the error-driven write plus replay**, not separation: the additive write fails (0.07-0.17, old and new values
   superpose), a single-pass delta write degrades with load (0.53 at 80 keys), replay restores it to 1.00.
4. The exact dictionary is perfect; the approximate memories reach the same accuracy in 4,096-4,352 floats against a 10,880-21,760-float
   reference for key and value codes (a real dictionary also stores the strings). The compression is real but is not a recall advantage.

Gate assessment after 5a: for NSA's actual keys the case for HSF over the exact store and over HDC is **fixed size and in-memory
supersession**, not accuracy. That is a modest case; it becomes decisive only when the memory must be *inside* a network (no external
dictionary available, differentiable, constant per-step cost), which is the Path B question, not an NSA-memory question.

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
