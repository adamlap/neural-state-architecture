# HSF Research Notebook: Brainstorming, Hypotheses, and Experimental Roadmap

**Project:** NSA Holographic State Field (HSF)  
**Status:** Living research document — hypothesis generation, not validated conclusions  
**Version:** 0.1  
**Current implementation:** \`experiments/holographic_state_field/\`  
**Research branch:** \`research/holographic-state-field\`

---

## 1. Purpose and research posture

This document is the durable idea bank and decision framework for exploring distributed state fields as a possible alternative or complement to conventional memory and attention mechanisms.

It is deliberately broader than the current implementation. It should preserve promising directions, competing explanations, negative results, unresolved questions, and experiments that are not yet worth implementing.

The goal is not to prove that a holographic field is superior. The goal is to discover which representational and dynamical principles, if any, provide measurable advantages under fair comparisons.

### Guardrails

1. **No architecture loyalty.** HSF is a candidate family, not a foregone conclusion.
2. **Falsifiability first.** Each experiment must have a result that would make us abandon or substantially revise its hypothesis.
3. **Baselines are first-class.** Compare against simple exact stores, vector stores, established associative-memory methods, and relevant neural architectures.
4. **Separate claims.** Storage capacity, retrieval accuracy, semantic generalisation, temporal reasoning, compute efficiency, and language-model capability are different claims.
5. **Preserve provenance.** Every result records code revision, configuration, seeds, raw observations, and analysis method.
6. **Keep NSA stable.** Research prototypes remain isolated until there is replicated evidence and a clear integration contract.
7. **Prefer the smallest decisive experiment.** Do not introduce an LLM, GPU, or complex training loop when a synthetic test can answer the question.

---

## 2. The central question

Can a bounded, distributed, continuously modifiable state field encode relationships and context in overlapping patterns, then recover useful information through local interaction, correlation, or dynamics — without retaining a conventional addressable slot for every memory?

This contains several independent hypotheses:

- **H1 — Distributed storage:** multiple associations can coexist in a shared field without explicit per-item slots.
- **H2 — Associative reconstruction:** a partial cue can recover a stored association from the field.
- **H3 — Graceful capacity:** degradation with load is gradual and predictable rather than catastrophic.
- **H4 — Structured generalisation:** related cues or concepts can share useful structure instead of behaving like unrelated random identifiers.
- **H5 — Dynamic cognition:** field evolution can maintain and update a compact working state over time.
- **H6 — Useful efficiency:** at a matched quality level, the representation can reduce memory movement, lookup cost, or compute relative to credible alternatives.
- **H7 — Compositionality:** multiple relations can be bound, queried, and recombined without enumerating every combination.
- **H8 — Selective plasticity:** new information can update, reinforce, suppress, or expire traces without damaging unrelated memories.

A successful result for H1 does not establish H4–H8. Each must be tested separately.

---

## 3. What we have now: HSF v0.1

The current prototype represents the field as:

\[
F \in \mathbb{R}^{H \times W \times C}
\]

A key generates a normalised bipolar spatial pattern \(K(k)\), and a value maps to a normalised channel code \(V(v)\). A write superposes an outer product:

\[
F \leftarrow F + \alpha K(k) \otimes V(v)
\]

A query correlates its key pattern with the field:

\[
\hat V(q) = \sum_{x,y} K(q)_{xy} F_{xy:}
\]

The reconstructed vector is decoded against a fixed vocabulary.

### What this proves — and does not prove

The implementation is a useful minimal test of fixed-size superposition and matched-filter retrieval. It is a software analogue of distributed associative memory, not an optical hologram and not yet a learned neural field.

It currently uses:

- deterministic pseudo-random bipolar key patterns;
- a fixed value vocabulary and codebook;
- additive, order-independent superposition;
- correlation-based retrieval;
- uniform decay and repeated-write reinforcement.

It does not yet provide:

- semantic geometry between keys;
- open-ended value encoding;
- learned representations;
- selective unbinding or reliable deletion;
- multi-hop relational reasoning;
- continuous-time field dynamics;
- a fair comparison with established associative-memory baselines;
- evidence of replacing transformer attention.

**Important benchmark correction:** the current explicit vector-slot byte baseline is a storage reference, not a complete competitive baseline. It should be supplemented with exact dictionaries, compact key/value storage, classical vector-symbolic methods, and matched-capacity alternatives. Report both raw payload bytes and actual process memory when practical.

---

## 4. Research directions and candidate mechanisms

These directions are intentionally competing. Several may turn out to be incompatible.

### A. Holographic and vector-symbolic representations

**Idea:** represent concepts and relations as distributed vectors, with algebraic operations for binding, bundling, permutation, and unbinding.

Candidate approaches:

- Holographic Reduced Representations (HRR): circular convolution for binding and correlation for unbinding.
- Binary Spatter Codes and related hyperdimensional computing (HDC) methods: high-dimensional bipolar or binary vectors with majority bundling.
- Fourier-domain binding: convolution becomes element-wise multiplication in the frequency domain.
- Complex-valued representations: phase can encode additional relational or temporal structure.
- Tensor-product binding: explicit role/filler composition as a high-capacity but potentially expensive control.

**Questions to test:**

- Does 2-D spatial structure add anything over a flat vector of equal element count?
- Is circular convolution more robust than the current outer-product binding?
- Does complex phase improve capacity or compositional retrieval?
- Can a query recover a relation when one or more components are missing?
- How does interference change with correlated rather than random keys?

**Small decisive experiment:** implement HRR and HDC baselines on identical synthetic key/value episodes, seeds, dimensions, vocabulary, and load curves. Compare accuracy, memory bytes, operation count, and retrieval latency.

### B. Sparse distributed memory and local associative access

**Idea:** memory is distributed over many locations, but a query activates a subset of nearby locations. The answer is reconstructed from their combined activity.

Candidate approaches:

- Kanerva-style Sparse Distributed Memory.
- Address patterns with Hamming or cosine neighbourhoods.
- Local activation and weighted voting.
- Multi-radius retrieval: coarse neighbourhood first, finer correction second.
- Sparse activation masks to avoid touching the entire field.

**Questions to test:**

- Can sparse access retain accuracy while reading a small fraction of the field?
- Is the best retrieval neighbourhood fixed, learned, or confidence-adaptive?
- Does sparsity reduce real memory bandwidth, or merely move the cost elsewhere?
- Can local regions specialise without becoming ordinary memory slots?

**Small decisive experiment:** compare dense correlation with top-k/local retrieval at matched accuracy and report actual elements read, wall-clock latency, and memory traffic where measurable.

### C. Attractor networks and energy-based retrieval

**Idea:** a query does not simply perform one correlation. It initializes a state, then recurrent dynamics move it toward a stored pattern or stable attractor.

Candidate approaches:

- Classical Hopfield networks.
- Modern continuous Hopfield networks.
- Dense associative memories with alternative energy functions.
- Recurrent cleanup networks that denoise an approximate retrieval.
- Hybrid: HSF produces an initial estimate; an attractor process resolves ambiguity.

**Questions to test:**

- Does iterative cleanup improve recall under interference?
- What is the cost in iterations and energy?
- Are spurious attractors introduced?
- Can the system recover from partial, corrupted, or noisy cues?
- Is there a measurable basin of attraction around each memory?

**Small decisive experiment:** inject controlled corruption into queries and compare one-shot correlation with iterative retrieval. Measure accuracy, convergence, iterations, false attractors, and runtime.

### D. Continuous neural fields and spatial dynamics

**Idea:** the field is not just a static tensor. It evolves through local interactions, diffusion, inhibition, excitation, and boundary conditions.

Candidate mechanisms:

- Neural-field equations with local excitation and lateral inhibition.
- Diffusion and reaction-diffusion dynamics.
- Wave propagation and interference.
- Oscillatory or phase-coupled fields.
- Local normalisation and competitive inhibition.
- Multiple coupled fields operating at different time scales.

**Questions to test:**

- Can local dynamics separate overlapping traces?
- Can a cue launch a wave that settles around a matching memory?
- Does inhibition reduce cross-talk without erasing weak memories?
- Can local rules produce stable organisation from initially unstructured patterns?
- Are the dynamics stable under continuous writes and reads?

**Small decisive experiment:** a 2-D toy field with a handful of stored patterns, local excitation/inhibition, and controlled noise. Compare static correlation against dynamic settling. Track energy, convergence, recall and stability.

### E. Self-organising and learned representations

**Idea:** patterns should not remain arbitrary hashes. The representation may learn to place related information in compatible regions or directions.

Candidate approaches:

- Hebbian updates and Oja-style normalisation.
- Self-organising maps or competitive learning.
- Contrastive learning for key embeddings.
- Predictive coding: reduce error between predicted and observed state.
- Online clustering with distributed prototypes.
- Learned encoders with fixed-size field storage.
- Local plasticity rules with bounded update magnitude.

**Questions to test:**

- Can similar concepts share structure and support generalisation?
- Does learning improve retrieval without increasing destructive interference?
- Can the system adapt online without catastrophic forgetting?
- Is learning local and incremental, or does it require global retraining?
- Does semantic similarity improve retrieval, or cause false associations?

**Small decisive experiment:** replace random key patterns with a frozen semantic encoder first, then with a small trainable encoder. Use held-out paraphrases, near-neighbour distractors, and unseen combinations. Keep the storage field and retrieval method fixed so representation quality is isolated.

### F. Multi-scale and hierarchical fields

**Idea:** one field may be insufficient. Use several coupled resolutions or subfields.

Candidate structures:

- Coarse global field plus fine local fields.
- Short-, medium-, and long-timescale traces.
- Separate entity, relation, and event subfields.
- Spatial pyramids or wavelet-like decomposition.
- Routing from a coarse field to a small number of detailed regions.
- Shared global context plus specialised local modules.

**Questions to test:**

- Does hierarchy improve capacity per byte?
- Can coarse retrieval route to a fine representation reliably?
- Does it avoid interference between unrelated domains?
- Is routing cost lower than scanning a flat field?
- How should information move between timescales?

**Small decisive experiment:** compare one large field with two- and three-level fields at equal total bytes. Measure accuracy, routing errors, read operations and update cost.

### G. Temporal memory, event streams and state evolution

**Idea:** memories are not static facts. They have timestamps, persistence, confidence, causal relationships, and changing validity.

Candidate mechanisms:

- Time-dependent decay and reinforcement.
- Temporal phase coding.
- Event traces and sequence binding.
- Predictive state fields.
- Separate episodic traces and consolidated semantic traces.
- Event-triggered writes rather than fixed-rate updates.
- Explicit supersession and provenance channels.

**Questions to test:**

- Can the field distinguish current from historical values?
- Can it preserve sequence order and temporal relations?
- Can a new event update a belief without erasing its history?
- Can important traces persist while irrelevant ones decay?
- Can it maintain a compact state over long distraction sequences?

**Small decisive experiment:** fact updates, supersession, historical queries, delayed recall and distractor sequences. Compare HSF against append-only memory, temporal key/value stores, and NSA's existing memory interfaces.

### H. Working memory and attention alternatives

**Idea:** use the field as a continuously updated workspace where inputs, goals, context, and candidate outputs interact.

Possible architectures:

1. **Read/write field:** incoming tokens or observations write traces; queries read the field.
2. **Recurrent workspace:** the field is updated each step and retains only task-relevant state.
3. **Multi-head fields:** independent fields specialise in entities, relations, goals, uncertainty, and temporal state.
4. **Competitive workspace:** candidate activations compete through inhibition and confidence.
5. **Predictive workspace:** the field predicts the next state and only writes prediction errors.
6. **Hybrid attention-field:** conventional attention handles local token processing while the field maintains persistent global context.
7. **Field as an attention cache alternative:** compare retrieval against KV-cache/context at matched quality and compute.

**Questions to test:**

- Can the field retain information after the source tokens leave the context window?
- Can it combine information from distant observations?
- Can it support multiple simultaneous goals without interference?
- Does it reduce repeated token processing?
- Can it be updated incrementally with lower cost than recomputing attention?
- Does it help a frozen language model answer better, or can it independently perform the relevant operation?

**Small decisive experiment:** start with a frozen model and a deterministic fact/update task. Compare bounded transcript, full transcript, external key/value memory, HSF read/write memory, and hybrid. Only move toward language modelling after retrieval and update semantics are robust.

### I. Multiple interacting fields and modular cognition

**Idea:** cognition may be better represented as several specialised but coupled fields rather than one universal memory surface.

Potential fields:

- Working context.
- Long-term semantic memory.
- Episodic/event memory.
- Predictive state.
- Goal and task state.
- Uncertainty/confidence.
- Safety and constraints.

Interaction mechanisms:

- gated message passing;
- shared global context channel;
- competitive or cooperative activation;
- asynchronous updates;
- event-driven synchronisation;
- learned routing.

**Questions to test:**

- Does modularity reduce interference?
- What should be shared and what should remain isolated?
- Can one field update without corrupting another?
- Is there a stable, interpretable interface between fields?
- Does modularity create a routing bottleneck?

**Small decisive experiment:** compare a monolithic field with two specialised fields under cross-domain interference and task switching, holding total memory and compute constant.

### J. Information-theoretic and capacity-driven design

**Idea:** stop treating tensor dimensions as arbitrary hyperparameters. Derive capacity and failure behaviour from information theory and empirical scaling.

Measure:

- bits per stored association;
- mutual information between key, stored value and reconstructed value;
- signal-to-interference ratio;
- retrieval error as a function of load;
- capacity at specified error thresholds;
- robustness to noise and correlated patterns;
- storage and compute cost per successful retrieval.

**Questions to test:**

- Is capacity predictable from dimensionality and code properties?
- Which field dimensions contribute useful independent capacity?
- Does normalisation improve signal-to-interference ratio?
- Can adaptive strength allocation improve total information retained?
- What is the Pareto frontier between accuracy, memory and compute?

**Small decisive experiment:** sweep dimensions, load, vocabulary size, key correlation and noise. Fit scaling curves and validate predictions on held-out dimensions/seeds.

---

## 5. Candidate architecture families to compare

Do not assume these are sequential versions of one design. They are separate candidates.

| Family | Representation | Retrieval | Main risk |
|---|---|---|---|
| HSF outer-product | Spatial key × channel value | Correlation | Cross-talk |
| HRR | Dense vectors | Correlation/unbinding | Binding noise |
| HDC/VSA | High-dimensional symbols | Similarity/cleanup | Weak semantic geometry |
| Sparse distributed memory | Distributed locations | Local activation | Address/read cost |
| Hopfield/attractor | Stored patterns in weights | Iterative settling | Spurious attractors |
| Neural field | Continuous activity surface | Local dynamics | Stability and compute |
| Learned field | Encoded continuous tensor | Learned or associative read | Training complexity |
| Hierarchical field | Coupled multi-scale fields | Coarse-to-fine | Routing errors |
| Hybrid attention-field | Tokens plus persistent field | Attention + field read | Integration overhead |

A result may justify combining families. A combination should only be introduced after its component mechanisms have been measured independently.

---

## 6. Experimental programme: staged gates

### Stage 0 — Reproducibility and baseline correctness

- Freeze HSF v0.1 and preserve its raw outputs.
- Verify tests and benchmark on a clean environment.
- Add exact dictionary and explicit vector-slot baselines.
- Confirm memory accounting and operation counts.
- Add a configuration manifest and per-seed results.
- Check duplicate keys, conflicting writes, empty fields and unseen queries.

**Exit condition:** independent reruns produce the same deterministic results and all baselines are clearly specified.

### Stage 1 — Capacity and interference

- Sweep field height, width and channels independently.
- Sweep number of memories and vocabulary size.
- Test multiple seeds and key/value distributions.
- Add noise to writes and queries.
- Include correlated keys and correlated value codes.
- Report mean, minimum, spread and confidence intervals.

**Exit condition:** obtain a reliable capacity curve and identify where HSF fails.

### Stage 2 — Established associative-memory baselines

- Implement HRR and HDC/VSA controls.
- Add an attractor/cleanup baseline.
- Add sparse retrieval where applicable.
- Match total representational bytes and task difficulty.
- Compare retrieval accuracy, memory, latency and operation count.

**Exit condition:** identify whether the current HSF binding has any defensible advantage.

### Stage 3 — Compositional and semantic retrieval

- Bind subject–relation–object triples.
- Query by subject/relation, relation/object, or partial cue.
- Test multi-hop queries.
- Test semantically similar keys and paraphrased cues.
- Hold out combinations during training where learning is used.
- Measure false positives as well as correct retrieval.

**Exit condition:** show generalisation beyond exact deterministic key regeneration.

### Stage 4 — Plasticity and temporal behaviour

- Add controlled reinforcement.
- Compare uniform, temporal and importance-weighted decay.
- Test supersession and historical recall.
- Test selective forgetting and collateral damage.
- Test online updates during retrieval.
- Test long sequences with distraction and task switching.

**Exit condition:** demonstrate useful updates without uncontrolled interference or hidden external indexes.

### Stage 5 — Dynamic fields and working memory

- Introduce local dynamics in a separate implementation.
- Measure convergence and stability.
- Compare static and dynamic retrieval.
- Test multi-timescale state.
- Test a persistent workspace on a frozen model.
- Compare against bounded context, full context, KV-cache and external memory.

**Exit condition:** show a reproducible advantage at matched task quality and resource budget.

### Stage 6 — NSA integration assessment

Only consider integration after independent validation.

- Define a narrow read/write interface.
- Keep the field optional and feature-gated.
- Preserve provenance, uncertainty and authority boundaries.
- Add checkpointing and recovery.
- Test adversarial and poisoned-memory cases.
- Compare end-to-end behaviour against existing NSA memory/runtime.

**Exit condition:** demonstrate that integration improves a real NSA workload without weakening reliability or governance.

---

## 7. Benchmark design and metrics

Every benchmark should report more than a single accuracy score.

### Retrieval quality

- Exact-match accuracy.
- Top-k accuracy where meaningful.
- False-positive and false-negative rates.
- Confidence calibration.
- Query robustness under corruption.
- Historical/current-value correctness.
- Multi-hop and compositional correctness.

### Capacity and information

- Number of associations.
- Effective capacity at 95%, 90%, and 50% accuracy.
- Bytes per association at each threshold.
- Signal-to-interference ratio.
- Mutual information where estimable.
- Performance under correlated patterns.

### Resource use

- Field and codebook payload bytes.
- Actual process RSS where practical.
- Read/write operations.
- Elements touched per query.
- FLOPs or a clearly stated operation-count proxy.
- Latency distributions, not just averages.
- Energy only when measured on suitable hardware.

### Dynamics

- Convergence steps.
- Stability under continuous writes.
- Recovery after perturbation.
- Forgetting curves.
- Collateral damage from selective forgetting.
- Update/read interference.

### Experimental discipline

- Multiple independent seeds.
- Fixed and published configurations.
- Raw per-episode records.
- Mean, standard deviation and confidence intervals.
- Held-out conditions.
- Ablations for each novel mechanism.
- No tuning on the final evaluation set.

A benchmark should never label a run PASS simply because a chosen threshold was met. Report measurements first; thresholds are preregistered decision aids, not substitutes for analysis.

---

## 8. Baselines and controls we should not skip

At minimum:

1. **Exact dictionary:** upper bound for exact key lookup.
2. **Explicit key/value vectors:** a transparent distributed-storage reference.
3. **Nearest-neighbour vector store:** a practical similarity retrieval baseline.
4. **HRR:** established holographic binding and unbinding.
5. **HDC/VSA:** high-dimensional bundling and similarity.
6. **Hopfield-style associative memory:** attractor-based retrieval.
7. **Sparse distributed memory:** local distributed retrieval.
8. **Bounded transcript:** realistic context-limited control.
9. **Full transcript:** upper-bound context control where feasible.
10. **Hybrid memory:** conventional retrieval plus field, to test whether the field adds value.

Not every experiment needs every baseline. Include the baselines that answer its particular question, and explain exclusions.

---

## 9. High-value experiments to prioritise

These are proposed priorities, not claims that the outcomes are known.

| Priority | Experiment | Why it matters | Effort |
|---|---|---|---|
| P0 | Reproduce v0.1 and add exact controls | Establish trustworthy starting point | Low |
| P0 | HSF vs HRR vs HDC at matched dimensions | Tests whether the current representation is competitive | Medium |
| P0 | Correlated-key interference sweep | Exposes an unrealistic assumption in random-key tests | Low |
| P1 | Subject–relation–object binding | Tests compositionality rather than simple lookup | Medium |
| P1 | Query corruption and cleanup dynamics | Tests robustness and attractor potential | Medium |
| P1 | Selective decay and supersession | Tests memory maintenance | Medium |
| P2 | Semantic key embeddings | Tests generalisation beyond hashes | Medium |
| P2 | Multi-scale field | Tests hierarchical capacity and routing | Medium |
| P2 | Neural-field local dynamics | Tests whether evolution improves retrieval | High |
| P3 | Frozen-LLM working-memory integration | Tests practical usefulness | High |
| P3 | Attention/KV-cache comparison | Tests the larger architectural ambition | Very high |

**Recommended immediate sequence:** P0 reproduction, HRR/HDC comparisons, and correlated-key tests. This is the shortest path to finding out whether the current core mechanism deserves more investment.

---

## 10. Failure modes and explicit kill criteria

We should be willing to stop or redesign a direction.

- If HSF is consistently dominated by HRR/HDC at matched memory and compute, do not keep expanding the outer-product design without a specific new hypothesis.
- If accuracy collapses under modest correlated-key interference, random-key capacity figures are not evidence of useful semantic memory.
- If dynamic settling improves accuracy only by spending substantially more compute than a direct lookup, it needs a compelling robustness or hardware argument.
- If semantic embeddings improve retrieval but cause unacceptable false associations, add explicit disambiguation or abandon that encoding for the task.
- If selective forgetting cannot avoid collateral damage, do not claim memory management is solved.
- If a hybrid field adds no value over a conventional vector store, do not integrate it into NSA.
- If a result depends on a hand-tuned threshold, one seed, or a post-hoc metric, treat it as exploratory only.
- If the field requires an external per-item index to perform the task, account for that index as part of the architecture and memory budget.

A negative result is useful when it narrows the design space.

---

## 11. Idea registry — add new directions here

Use one entry per distinct idea. Keep entries short until an experiment is approved.

### Idea template

**ID:** HSF-I-XXX  
**Name:**  
**Mechanism:**  
**Hypothesis:** If we do X, then Y should improve because Z.  
**Competing explanation:**  
**Minimal experiment:**  
**Control/baseline:**  
**Primary metric:**  
**Failure condition:**  
**Estimated effort:** Low / Medium / High  
**Status:** Captured / Designed / Running / Analysed / Accepted / Rejected / Parked  
**Related ideas:**  
**Notes/results:**  

### Initial idea backlog

| ID | Idea | Hypothesis to test | Status |
|---|---|---|---|
| HSF-I-001 | HRR circular-convolution binding | Algebraic binding may reduce cross-talk vs outer products | Captured |
| HSF-I-002 | HDC majority bundling | Binary high-dimensional codes may improve storage efficiency | Captured |
| HSF-I-003 | Correlated key patterns | Realistic similarity may expose capacity limits | Captured |
| HSF-I-004 | Attractor cleanup | Iterative settling may recover corrupted cues | Captured |
| HSF-I-005 | Sparse local retrieval | Reading only active regions may reduce query cost | Captured |
| HSF-I-006 | Learned semantic keys | Similar concepts may share useful field structure | Captured |
| HSF-I-007 | Hebbian plasticity | Local updates may organise representations online | Captured |
| HSF-I-008 | Selective inhibition | Competition may suppress interference between traces | Captured |
| HSF-I-009 | Multi-scale fields | Coarse-to-fine routing may improve capacity per byte | Captured |
| HSF-I-010 | Temporal phase coding | Phase may encode order and temporal context | Captured |
| HSF-I-011 | Predictive error writes | Writing only unexpected information may reduce redundancy | Captured |
| HSF-I-012 | Field as working memory | Persistent field state may replace repeated transcript processing | Captured |
| HSF-I-013 | Modular interacting fields | Specialised fields may reduce cross-domain interference | Captured |
| HSF-I-014 | Confidence-aware retrieval | Uncertainty could trigger additional retrieval or abstention | Captured |
| HSF-I-015 | Selective unbinding/forgetting | Local inverse or cancellation operations may remove one association | Captured |
| HSF-I-016 | Event-driven field updates | Updating on meaningful changes may be cheaper than fixed-rate updates | Captured |
| HSF-I-017 | Energy-based capacity analysis | Measured signal/interference may predict failure thresholds | Captured |
| HSF-I-018 | Field plus transformer hybrid | A persistent field may complement local attention before replacing anything | Captured |

---

## 12. Experiment record template

Create one record for every run, including failed and partial runs.

- **Experiment ID:**
- **Date/time (UTC):**
- **Git commit:**
- **Hypothesis:**
- **Implementation changes:**
- **Baseline(s):**
- **Configuration:**
- **Seeds:**
- **Dataset/task generator:**
- **Primary metric:**
- **Secondary metrics:**
- **Raw result path:**
- **Analysis path:**
- **Unexpected observations:**
- **Threats to validity:**
- **Conclusion:** Supported / Not supported / Inconclusive
- **Next decision:** Continue / Modify / Replicate / Park / Reject

Never overwrite raw results. Give each run a unique identifier and preserve the exact configuration used.

---

## 13. Proposed repository layout

Keep research code separate from production NSA modules.

~~~text
experiments/
  holographic_state_field/
    README.md
    field.py
    benchmark.py
    baselines/
      hrr.py
      hdc.py
      hopfield.py
      sparse_memory.py
    dynamics/
      neural_field.py
      attractor.py
    tasks/
      associative_recall.py
      relational_binding.py
      temporal_memory.py
      working_memory.py
    analysis/
      capacity.py
      interference.py
      statistics.py
    configs/
    results/
      README.md
    records/
      README.md
tests/
  test_holographic_state_field.py
  test_hsf_baselines.py
  test_hsf_dynamics.py
docs/
  HSF_RESEARCH_BRAINSTORM.md
~~~

Do not create empty modules just to match this proposed layout. Add a directory when the first experiment needs it.

---

## 14. Decision log

| Date | Decision | Reason | Revisit when |
|---|---|---|---|
| 2026-11-10 | Keep HSF isolated under experiments | Avoid destabilising NSA runtime before evidence | Independent validation passes |
| 2026-11-10 | Start with synthetic associative recall | Isolate representation from LLM extraction and language quality | Basic retrieval and baselines are understood |
| 2026-11-10 | Treat attention replacement as a long-term hypothesis | Memory retrieval alone does not establish attention capability | Working-memory experiments show a measurable advantage |

Add decisions here when a result changes our direction. Record why an approach was rejected as carefully as why it was selected.

---

## 15. Current recommendation

The most valuable next step is not to add more features to HSF v0.1. It is to determine what kind of associative memory it actually is, and whether its binding operation is a good one.

Proceed in this order:

1. Reproduce the existing capacity sweep and verify the tests.
2. Add exact dictionary and explicit vector-slot controls.
3. Implement HRR and HDC/VSA baselines with matched dimensions and task generators.
4. Stress-test correlated keys, noisy cues, repeated writes and conflicting values.
5. Publish the capacity/accuracy/compute trade-off before selecting a more complex direction.

Only after this should we invest in learned semantic fields, attractor dynamics, or continuous neural-field behaviour.

**Long-term ambition:** investigate a persistent, distributed, self-organising workspace that can encode, retrieve, update and combine information over time — and rigorously test whether it can complement or replace specific functions of conventional attention and memory.

**Research principle:** build the smallest experiment that could prove us wrong, then let the evidence decide what we build next.
