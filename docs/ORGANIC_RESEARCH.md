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

## Experiment 2: does the trailing field really bind the delayed outcome? (`dog_test_v2.py`, `dog_test_v3.py`)

**Methodological correction.** The first version of this experiment re-created the field for every event, which removed all contamination from
earlier events and made slow traces look perfect (and v1 handed the cue-end key to the learning step, so no trace was involved at all). The
numbers below are from a **continuous stream** in which the field persists across events. Earlier figures in this repository history
(slow trace perfect at delay 25-100) were artefacts and are superseded.

Two agents differ only in the key used when the outcome arrives: `stored` (the cue-end key kept in an eligibility buffer) and `trace` (true
trace conditioning: the live field state at outcome time). 8 seeds (probe 6), 300-400 events, valenced-cue accuracy; decoy accuracy stayed
≥ 0.97 except where noted. `results/organic_dog_test_v2.json`, `results/organic_dog_test_v2_long_delay.json`.

| key | delay 3 | 15 | 25 | 40 | 60 | 100 |
|---|---|---|---|---|---|---|
| stored (any field) | 1.00 | 1.00 | 1.00 | - | - | - |
| trace, fast (λ .8) | 1.00 | 0.68 | 0.18 | 0.15 | 0.11 | 0.19 |
| trace, slow (λ .95) | 1.00 | 1.00 | 1.00 | 1.00 | 0.69 | 0.26 |
| trace, very slow (λ .99) | **0.67** | 0.96 | 0.97 | 1.00 | 1.00 | 1.00 |
| trace, equal-weight sum (.5/.8/.95) | 1.00 | 0.90 | 0.75 | 0.48 | 0.32 | 0.10 |
| trace, equal-weight sum (.8/.95/.99) | 1.00 | 0.99 | 0.94 | 0.91 | 0.81 | 0.62 |

1. **A trace binds only while it lasts and only while it is clean.** A trace must be slow enough to still hold the cue at outcome time and fast
   enough that earlier events have decayed: λ .8 fails beyond delay ~10, λ .99 fails at delay 3 (0.67, contaminated by previous events), and each
   timescale has a window. The eligibility buffer has no such window but costs a stored key per pending event.
2. **Mixed delays** (cue types with outcome delays 3/15/40/80, `results/organic_dog_test_v3.json`, 6 seeds), valenced accuracy overall and by delay:

| learner | overall | d=3 | d=15 | d=40 | d=80 |
|---|---|---|---|---|---|
| single λ .8 | 0.71 | 1.00 | 1.00 | 0.22 | 0.03 |
| single λ .95 | 0.89 | 1.00 | 1.00 | 1.00 | 0.36 |
| single λ .99 | 0.94 | 0.96 | 0.93 | 0.86 | 1.00 |
| equal-weight concatenation | 0.90 | 1.00 | 1.00 | 0.89 | 0.53 |
| **joint codes** (one sparse code per timescale, one delta-rule vector) | **0.95** | 1.00 | 1.00 | 1.00 | 0.69 |
| expert mixture (weights exp(-βerr)) | 0.90 | 1.00 | 1.00 | 0.97 | 0.42 |

   Giving each timescale its own sparse code and letting a single delta rule assign the weights is the best of the combinations at short and
   medium delays and ties the best single trace overall (0.95 vs 0.94); it does not recover the longest delay (0.69 vs 1.00 for λ .99).
   The online expert-weighting scheme was no better than equal-weight concatenation. No combination dominates; the differences between the
   top three are small for 6 seeds.
3. **Self-detected events** (`dog_test_v3.py`, 4 seeds): a learned next-input predictor and a plain z-score of the input norm detect cue
   onsets equally well. Both are perfect at cue amplitude ≥ 1 with noise ≤ 0.5 and at amplitude 0.5 with noise 0.2, and both collapse when the
   signal-to-noise ratio falls to ~1 or below (recall 0.03-0.23 at amplitude 0.25-0.5 with noise ≥ 0.5, and at amplitude 1.0 with noise 0.8;
   precision stays 0.6-0.95). **The predictive detector gives no advantage over the trivial baseline in this world**, so "surprise from the
   stream" is so far not shown to be a better event signal than energy.

## Experiment 3: the first LLM wrapper (v0) - a familiarity gate (`experiments/organic/novelty_gate.py`)

The wrapper does not feed the field to the LLM; it *gates* it, following the earlier NSA finding that state shown to a model does nothing while
state that controls something does. `NoveltyGate` is a fixed-size (256×256 = 65,536 floats) autoassociative linear memory trained online with
the delta rule on hashed character-trigram cues; an event's novelty is `‖x − W x‖` (what the memory cannot reconstruct), after which the memory learns
it. It decides which events deserve an LLM call or a memory write. No LLM is called in this benchmark; it scores how well each detector ranks events
by importance on a synthetic conversation stream (facts, reminders - same value, sometimes with a typo or other phrasing - updates, chatter;
important = first mention of a key or an update to a different value; 10 seeds; `results/organic_novelty_gate_v1.json`).

| stream | detector | memory floats | ROC AUC (worst seed) | recall at a 20 % call budget | false-call rate |
|---|---|---|---|---|---|
| 300 lines | **field gate** | 65,536 | **0.863** (0.843) | 0.42 | 0.09 |
| | exact token-set novelty | 232 | 0.840 | 0.32 | 0.14 |
| | nearest-neighbour over all stored lines | 76,800 | 0.826 | 0.43 | 0.09 |
| | random | 0 | 0.502 | 0.20 | 0.20 |
| 2000 lines | field gate | 65,536 | 0.868 (0.856) | 0.56 | 0.12 |
| | exact token-set novelty | 804 | 0.868 | 0.42 | 0.15 |
| | nearest-neighbour over all stored lines | 512,000 | 0.880 | 0.58 | 0.12 |
| | random | 0 | 0.493 | 0.19 | 0.20 |

Reading, with limits stated:

1. **The gate works and its cost is constant.** It ranks important events far above chance (AUC 0.86-0.87) at a fixed 65,536 floats while the
   nearest-neighbour detector grows from 76,800 to 512,000 floats for the same quality (AUC 0.88 at 2000 lines, 0.826 at 300).
2. **It is not better than simple baselines.** At 2000 lines the exact token-set detector ties it on AUC with a few hundred stored tokens, and the
   nearest-neighbour detector is slightly better. The field's measurable edge is at the call budget (recall 0.56 vs 0.42 for the token set), where
   typo'd reminders are familiar to a distributed memory but "new tokens" to the set.
3. **Sensitivity:** the learning rate matters (AUC at 300 lines: 0.81 / 0.86 / 0.88 for η = .25 / .5 / 1.0; at 2000 lines 0.85 / 0.86 / 0.86). Not tuned for any detector.
4. The ceiling here is about 0.88 for every detector: some "important" events (a first mention that looks like earlier text, an update whose phrasing
   repeats earlier updates) are lexically indistinguishable from reminders; telling them apart needs the key-value structure, i.e. the memory
   itself, not a lexical familiarity score.

What this is *not*: an end-to-end test with an LLM in the loop, nor a compute saving yet. Next: (1) merge this branch's field code with `main`
and implement the gate inside `NSARuntime` (decide *observe-only / write to memory / call the model* from the novelty score); (2) run the existing
benchmarks (v7/v9 style streams) with a real model and report calls saved against answer accuracy at equal budget; (3) replace the lexical
cue with embeddings of continuous sensory input for the organic world.

## Wrapping a language model (design)
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
