# Organic track: continuous streams, trailing field, selective learning

Working hypothesis (from the project owner): the ingredients that make biological cognition different from token-based models are
(1) *continuous, multi-channel sensory streams* instead of discrete tokens, (2) a *trailing field* - recent input persisting as a decaying
trace so that later events can bind to earlier ones, and (3) memory that is *selective and lossy* - only filtered patterns are kept and
learned from, and behaviour is the criterion, not exact recall. Consciousness is not claimed or tested here; this track asks which of the
mechanisms measurably help an agent learn from a stream, and which do not.

## Experiment 1: the "dog test" (`experiments/organic/dog_test.py`)

World: three 8-dim sensory channels with smooth noise; cue prototypes across channels; some cues are followed after a 3-step delay by a
rewarding (+1) or aversive (-1) outcome, most are decoys followed by nothing. The agent predicts the outcome at the end of each cue.

Organic agent: a trailing field (leaky trace, λ = 0.8) whose state at cue end is still present when the outcome arrives; a fixed sparse
random expansion (5 % winners-take-all) of the field state; **one** fixed memory vector over that code; delta-rule writes only when the
prediction error exceeds a threshold (surprise gate); optional replay of the 32 most surprising events during idle time.
Baselines at the same float budget: k-nearest-neighbour over stored (field state, outcome) entries with FIFO, random-reservoir and
surprise-gated retention; an unbounded kNN (perfect recall, reference); no memory. Scores: sign accuracy on held-out noisy variants,
decoys predicted ~0, graded response on morphs between a rewarding and an aversive cue, accuracy after 400 decoy-only events (retention).
Raw data: `results/organic_dog_test_v1.json`, `results/organic_dog_test_hard_v1.json` (6 seeds, 800 training events).

| setting | agent | floats | valenced acc. | decoy acc. | morph corr. | after 400 distractors |
|---|---|---|---|---|---|---|
| 24+48 cues, ~1.8k floats | organic (with replay) | 1,824 | 0.99 | 1.00 | 0.98 | 0.98 |
| | organic, no replay | 1,024 | **1.00** | 1.00 | 0.99 | 0.99 |
| | kNN FIFO | 1,800 | 0.67 | 0.92 | 0.91 | **0.00** |
| | kNN random | 1,800 | 0.70 | 0.93 | 0.92 | 0.63 |
| | kNN surprise-gated | 1,800 | 0.99 | 0.92 | 0.92 | 0.98 |
| | kNN unbounded | - | 1.00 | 1.00 | 0.92 | 1.00 |
| 24+48 cues, ~0.7k floats | organic, no replay | **256** | 0.92 | 0.95 | 0.98 | 0.88 |
| | organic (with replay) | 656 | 0.89 | 0.88 | 0.97 | 0.71 |
| | kNN surprise-gated | 650 | 0.76 | 0.75 | 0.93 | 0.78 |
| | kNN FIFO / random | 650 | 0.42 / 0.48 | 0.90 / 0.87 | 0.06 / 0.22 | 0.00 / 0.44 |

What the data support:

1. **Forgetting is a design choice, and unselective retention is the failure.** A FIFO buffer forgets every valenced cue after a stretch
   of uneventful events (0.00); random retention degrades; keeping what is surprising retains it (0.98). Selectivity is the mechanism,
   whether the store is a vector or a list.
2. **The fixed-size distributed memory is far more float-efficient for this behaviour.** With 256 floats it reaches 0.92 where a
   surprise-gated entry store needs roughly 1,800-3,000 floats for 0.99-1.00 (and gets 0.76 at 650). The mechanism is that the delta rule
   stores *what the memory does not already predict* and overlapping traces share the load.
3. **Overlap generalises.** The distributed memory's response between a rewarding and an aversive cue is monotone (r ≈ 0.97-0.99) where kNN
   is less so (0.06-0.94 depending on budget). The overlap that counted as "interference" in the exact-recall benchmarks is the
   generalisation property here.
4. **Negative results:** replay did **not** help once the world was harder (no-replay is equal or better and costs fewer floats; with a
   buffer at 700 floats it is worse, 0.71 vs 0.88 after distractors). The surprise gate was inert in the easy world because decoys stop
   producing error once learned (write-all equals gated). The unbounded kNN is perfect and an honest upper bound; the point is the cost.

Limits: a toy world with synthetic prototypes and one outcome per event; the trailing field is a single leaky trace with a hand-set
constant; thresholds and constants were not tuned for either side; results are not language and not a model of awareness.

## Where the "trailing field" matters
**Correction to Experiment 1:** v1 captured the cue-end key and handed it to the learning step after the delay, so the trailing field did
not mediate the binding there; the claim in the paragraph below was only demonstrated by v2. A multi-timescale version (several λ), prediction of the *next field state* rather than the
outcome, and delays longer than the trace are the next measurements.

## Experiment 2: does the trailing field really bind the delayed outcome? (`experiments/organic/dog_test_v2.py`)

Two agents differ only in the key used when the outcome arrives: `stored` (the cue-end key kept in an eligibility buffer, v1's behaviour) and
`trace` (true trace conditioning: the live field state at outcome time, the cue surviving only as a decaying trace). Field time constants:
`mid` (λ = .8), `slow` (.95), `multi` (concatenation of .5/.8/.95, each normalised), plus `slower` (.99) in the long-delay probe.
8 seeds (probe: 6), 400 events (probe: 300), valenced-cue accuracy (decoy accuracy stayed ≥ 0.98 throughout), `results/organic_dog_test_v2.json`,
`results/organic_dog_test_v2_long_delay.json`.

| key | delay 3 | 8 | 15 | 25 | 40 | 60 | 100 |
|---|---|---|---|---|---|---|---|
| stored (any field) | 1.00 | 1.00 | 1.00 | 1.00 | - | - | - |
| trace, mid (.8) | 1.00 | 1.00 | 0.80 | **0.17** | - | - | - |
| trace, slow (.95) | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 | 0.62 | 0.18 |
| trace, slower (.99) | - | - | - | 1.00 | 1.00 | 1.00 | 0.99 |
| trace, multi (.5/.8/.95) | 1.00 | 0.98 | 0.95 | 0.77 (0.69) | 0.49 | 0.28 | 0.12 |
| trace, (.8/.95/.99) | - | - | - | 0.98 | 0.96 | 0.90 | 0.59 |

1. **The trace binds only while it lasts.** A fast trace gives full accuracy at delay 3-8 and collapses at 15-25 (0.80 → 0.17); a trace whose
   time constant is matched to the delay holds (λ = .95 to delay 40, λ = .99 past 100). The mechanism works and has a measurable horizon of about
   1-2 trace time constants; the explicit eligibility buffer has no such limit but stores a key per pending event.
2. **Concatenating timescales hurts** (multi is worse than slow at every long delay): the fast components are noise at outcome time and dilute
   the key. A multi-timescale field needs *selection or learned weighting* of timescales (e.g. weight each trace by how well it predicts the
   outcome), not an equal-weight sum. Not yet implemented.
3. **Self-detected events** (next-input prediction error as a z-score above its running mean) found every cue onset with no false alarms
   (recall 1.00, precision 1.00 over 8 seeds). That is a ceiling result: the cues are large relative to the noise, so it shows the mechanism
   works, not that it is robust. A noise/amplitude sweep is the next test.

## Wrapping a language model (design, not implemented)
Earlier NSA results are a constraint here: showing serialised state *text* to an LLM did not help (v1.2, v4.2); what helped was state that
**controls** something. So the organic module should not be pasted into the prompt. Three coupling modes, in order of what the evidence
supports: (1) gating - the field's surprise/novelty decides *whether* to call the LLM, retrieve, or abstain (cf. v7); (2) selective context -
the field decides what is written to and retrieved from NSA's memory instead of a transcript (cf. v8); (3) latent injection - the field state
is projected into the LLM's embedding space as soft-prompt vectors (needs a model whose weights we can run and a trained projection).

## Branch note: using the field to keep a large LLM mostly out of memory (not started)
The same machinery could, in principle, predict which parts of a large model the current input needs so the rest stays on disk: a small
predictive field over the input/context that selects experts (mixture-of-experts routing) or attention heads and MLP neurons ("contextual
sparsity"). NSA's current residency layer only prefetches layers in sequence and measured neutral on a dense model, so the promising targets
are MoE models and learned contextual sparsity, evaluated by memory resident, latency and answer quality against llama.cpp-style mmap.
It needs a trainable predictor and a model with sparse structure; treat as a separate branch.
