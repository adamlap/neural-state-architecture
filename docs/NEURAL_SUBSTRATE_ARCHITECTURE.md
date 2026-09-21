# NSA Neural Substrate Architecture

## Purpose

The long-term goal of the Neural State Architecture (NSA) is not merely to wrap an LLM or add a cache around inference.

The goal is a governed neural substrate in which **cognitive state, token flow, computation, neural residency, security, and physical resource movement are coordinated through a common state-transition architecture**.

The central idea is:

> Selective computation can become selective memory, while NSA governs the state and flow of information, computation, and neural resources.

## 1. The original observation

Sparse/MoE models can contain a very large population of specialists while only a small subset is selected for a particular token.

Instead of keeping every specialist in fast memory:

    VRAM = currently useful neural resources
    RAM  = warm / likely resources
    NVMe = cold / rarely needed resources

For dense models, the same principle can be applied to layers or other parameter regions.

The resulting model is a **dynamically materialized computational substrate**.

## 2. Target architecture

    +-----------------------+
    |     NEURAL STATE      |
    | context, goals,       |
    | history, predictions, |
    | authority, provenance |
    +-----------+-----------+
                |
         governed transition
                |
       +--------+--------+--------+
       |                 |        |
       v                 v        v
   TOKEN FLOW       COMPUTATION  MEMORY
       |                 |        |
   routing           experts    residency
       |                 |        |
       +-----------------+--------+
                         |
                    execution
                         |
                    new state

The three flows are related but must not be conflated.

A residency predictor may determine what is likely to be useful; it must not thereby gain authority to execute an otherwise prohibited operation.

## 3. Token flow

NSA should make token movement observable and governable.

A future token context can contain:

    cognitive_state
    task_state
    provenance
    authority
    policy_context
    priority
    routing constraints
    predicted computation
    residency requirements

This lets NSA reason about transitions such as:

    input -> interpretation -> retrieval/tool use -> reasoning -> output

Why: data movement is part of system behavior. Transfers across trust or authority boundaries should be explicit rather than invisible implementation details.

## 4. Computation flow

For an MoE model:

    router
      |
      +-- E12
      +-- E42
      +-- E63
      |
    execution

For dense models, regions can be layers, attention blocks, MLP blocks, or parameter groups.

NSA should observe:

- selected region
- current state
- previous region
- transition probability
- execution latency
- resource requirements
- resulting state

Why: the system can learn current-state -> likely-next-computation relationships instead of treating every inference independently.

## 5. Memory flow

Neural resources have explicit physical residency:

    NVMe -> RAM -> VRAM -> ACTIVE
      ^       ^      |
      |       |      |
      +-------+------+
             eviction

Regions should carry:

- tier
- residency state
- precision
- compression
- size
- predicted usefulness
- dependencies
- transfer cost
- latency
- prefetch status
- execution history

Why: memory placement becomes a first-class part of neural execution.

## 6. Selective computation + selective storage

The key research principle is:

> The sparse computation graph can become a sparse memory graph.

If a model has 100 experts and only 4 are selected, the physical working set does not necessarily need to contain all 100.

Conceptually:

    VRAM: current experts
    RAM:  recent + predicted experts
    NVMe: remaining experts

The exact distribution is dynamic.

For MoE models this can potentially decouple **total parameter count** from **fast-memory requirement**.

For dense models, the same concept applies at layer/block granularity, although every layer may still participate in computation.

## 7. Predictive residency

Reactive caching waits until a resource is requested.

NSA should instead learn transitions:

    E17 -> E42 : 0.81
    E17 -> E63 : 0.62
    E17 -> E08 : 0.11

While E17 executes, the system can prepare E42 and E63.

The feedback loop is:

    prediction
       |
    prefetch
       |
    actual routing
       |
    execution
       |
    prediction verified
       |
    update predictor
       |
    better prediction
       |
    better residency

This makes memory management a learned component of the cognitive substrate.

## 8. Compression as a residency dimension

Location alone is not enough. Cold resources can use more aggressive representations.

For example:

    HOT   -> FP16/BF16
    WARM  -> INT8
    COLD  -> INT4
    VERY COLD -> stronger compression

A resource could transition:

    INT4 NVMe -> INT4 RAM -> INT8 RAM -> FP16 VRAM

This creates **adaptive precision residency**.

The residency planner therefore chooses both:

    where should this region live?
    how should it be represented?

## 9. Parallel materialization

The GPU should ideally compute while the storage pipeline prepares future resources.

Target overlap:

    GPU COMPUTE       ========================>
    NVMe I/O              ======================>
    DECOMPRESSION             ==================>
    CPU -> GPU TRANSFER          ===============>

For multiple experts, independent I/O and decompression jobs can run concurrently.

Multiple NVMe devices can also become independent storage lanes.

The scheduler should eventually consider:

- I/O bandwidth
- decompression throughput
- PCIe bandwidth
- VRAM capacity
- CPU capacity
- transfer latency
- expert dependencies

Why: latency should be hidden behind useful computation whenever hardware permits.

## 10. Semantic bundles

Individual expert prediction is not always the best abstraction.

NSA can learn bundles such as:

    coding state
      -> E12 E18 E41 E63

    planning state
      -> E04 E12 E55

Then the bundle itself can become a prefetch unit.

This changes the prediction problem from:

    expert -> expert

toward:

    cognitive state -> computation bundle -> residency bundle

This is a bridge between cognitive state and physical neural resources.

## 11. Delta compression

Related regions may share parameter structure.

A possible future representation is:

    shared base
       |
       +-- E12 delta
       +-- E18 delta
       +-- E41 delta

This can reduce storage but introduces reconstruction cost, so it must be experimentally validated.

## 12. Quantum ideas

Quantum superposition should not be treated as a magical way to store arbitrary classical model weights in less memory.

A qubit superposition does not allow arbitrary classical information to be read out simultaneously. Therefore quantum memory is not currently a practical replacement for model storage.

A future quantum computer could, however, potentially help with optimization problems such as:

- which regions should remain resident
- which regions should be prefetched
- which should be evicted
- how to allocate constrained memory

For the near term, a useful classical analogue is a probability distribution over future computation:

    P(E12) = 0.41
    P(E37) = 0.28
    P(E51) = 0.19
    P(E82) = 0.07

NSA should retain uncertainty instead of prematurely committing to one future.

The residency planner can then choose the set of regions that maximizes expected utility.

## 13. Residency optimization

A future planner can optimize:

    expected computation benefit
  + latency reduction
  + prediction confidence
  + dependency coverage

subject to:

    VRAM capacity
    RAM capacity
    I/O bandwidth
    CPU capacity
    transfer latency
    policy/security constraints

A simplified utility function is:

    utility(region)
      = P(use) * value(use)
        - memory_cost
        - transfer_cost
        - decompression_cost

The current implementation uses heuristic approximations. Learned or optimization-based schedulers are future work.

## 14. Unified state and flow governance

A token can trigger:

    token
      |
      +--> cognitive-state update
      +--> routing decision
      +--> computation prediction
      +--> residency prediction
      +--> security/policy evaluation
      |
      v
    execution
      |
      v
    new state
      |
      +--> next token
      +--> next computation
      +--> next residency plan

The fundamental loop becomes:

    state -> token -> computation -> memory -> result -> new state

This is the central unification.

## 15. Governance and authority

Optimization must never become authority.

The architecture should maintain:

    NSA GOVERNANCE
       |
       +--> TOKENS
       +--> COMPUTE
       +--> RESIDENCY

The security automaton and policy layer determine what transitions are permitted.

The cognitive/residency system determines what would be useful.

A predictor can say "E42 is likely to be useful"; it cannot thereby authorize an otherwise prohibited operation.

Likewise, a memory manager may move parameters between NVMe, RAM, and VRAM without changing the authority model.

## 16. Observable physical flow

Physical movement should be represented as typed state transitions.

Example:

    REGION_42
       |
      NVMe
       | prefetch
       v
      RAM
       | materialize
       v
      VRAM
       | execute
       v
     ACTIVE
       | evict
       v
      NVMe

Each transition can generate a ResidencyEvent containing region, action, source, destination, bytes, latency, and reason.

This means NSA can learn:

> I predicted region 42, materialized it, used it, and the prediction was correct.

That closes the loop between prediction and physical execution.

## 17. Compression, parallelism, prediction and governance are one system

These should not become unrelated features.

The target relationship is:

    cognitive state
          |
          v
    future prediction
          |
      +---+---+----------------+
      |       |                |
      v       v                v
   routing  residency       policy
      |       |                |
      v       v                |
   compute  prefetch           |
      |       |                |
      +---+---+----------------+
          |
      compression
          |
      parallel materialization
          |
       execution
          |
      new state

Each mechanism operates on the same underlying state-transition substrate while retaining explicit authority boundaries.

## 18. Current implementation

The current NSA residency work provides the foundation:

- neural regions
- VRAM/RAM/NVMe tiers
- residency states
- byte-bounded caches
- residency policy
- transition prediction
- online prediction
- active residency controller
- asynchronous prefetch scheduling
- residency telemetry
- Accelerate integration
- safetensors page-cache prefetching
- selective Transformers storage
- local model registry
- benchmark and experiment harnesses

The current Transformers backend primarily demonstrates layer-level selective residency.

The Accelerate integration deliberately distinguishes logical prefetch intent from actual parameter materialization.

That distinction must remain explicit.

## 19. Evolution roadmap

### Phase 1 — Selective residency

Measure:

- memory usage
- transfer latency
- generation latency
- residency transitions
- prefetch hit rate

### Phase 2 — Predictive residency

Improve:

- transition prediction
- cognitive-state prediction
- lookahead
- prefetch scheduling
- eviction policy

### Phase 3 — Adaptive precision

Add:

- FP16/BF16
- INT8
- INT4
- region-specific precision

### Phase 4 — Parallel materialization

Add:

- asynchronous I/O
- decompression workers
- transfer pipelines
- multiple storage lanes
- GPU/CPU overlap

### Phase 5 — MoE specialist residency

Use real sparse models and map:

    expert -> neural region

Then connect router decisions directly to the residency planner.

This is the strongest experiment for validating the original selective-computation/selective-storage hypothesis.

### Phase 6 — Semantic bundles

Learn:

    state -> expert bundle
    state -> residency bundle

### Phase 7 — Unified flow governance

Connect:

    token state
    computation state
    residency state
    security state

into the canonical NSA state-transition system.

### Phase 8 — Self-optimizing substrate

The system learns:

    what to compute
    what to keep
    what to compress
    what to prefetch
    what to evict

while remaining constrained by explicit governance and security invariants.

## 20. What we must measure

### Computation

- active parameters
- selected experts
- tokens/sec
- time to first token
- per-token latency

### Memory

- peak VRAM
- peak RAM
- NVMe footprint
- resident bytes
- bytes transferred

### Residency

- prefetch hit rate
- prefetch miss rate
- eviction rate
- prediction accuracy
- prediction lead time

### Storage

- compression ratio
- decompression throughput
- I/O bandwidth
- reconstruction latency

### Quality

- output quality
- task accuracy
- perplexity where appropriate
- quantization/compression degradation

### Governance

- permitted transitions
- rejected transitions
- policy violations
- provenance integrity
- observational equivalence where applicable

The important research question is not simply "does it use less memory?"

It is:

> How much computation and model capacity can we expose per unit of physical fast memory without unacceptable latency, quality, or governance degradation?

## 21. Long-term vision: a virtual neural machine

The complete model exists as a logical computational substrate. Only a dynamically selected working set is physically materialized.

    LOGICAL MODEL
         |
    neural state + model storage
         |
    prediction
         |
    materialization
         |
    +----+----+----+
    |    |    |    |
   VRAM RAM  NVMe compressed storage
    |    |    |
    +----+----+
         |
     COMPUTATION
         |
      NEW STATE
         |
         +----> next transition

The managed unit need not be a conventional memory page. It can be:

- an expert
- a layer
- an attention block
- a parameter group
- a semantic bundle
- a compressed representation
- a computation-graph fragment

The system decides what should be materialized based on **state and predicted computation**.

## 22. Design principles

1. **State is first-class.** Cognitive state participates in prediction and transitions.
2. **Computation is selective.** Execute only necessary regions when model architecture permits it.
3. **Storage is selective.** Keep only necessary neural resources in scarce fast memory.
4. **Residency is predictive.** Future computation influences present memory placement.
5. **Compression is adaptive.** Cold resources can use more aggressive representations.
6. **Data flow is observable.** Token and resource movement produce explicit transitions.
7. **Optimization does not imply authority.** Predictors and schedulers cannot bypass policy.
8. **Physical execution is measurable.** Logical intent and actual materialization remain distinct.
9. **Parallelism should hide latency.** I/O, decompression, transfer and computation should overlap.
10. **Research claims require measurement.** Reproducible experiments and telemetry are mandatory.

## 23. Core research question

> Can a large neural system be treated as a dynamically materialized computational substrate whose active physical state is governed by cognitive state and predicted computation, rather than requiring its entire logical parameter space to remain resident?

The strongest MoE formulation is:

    SELECTIVE COMPUTATION
          +
    SELECTIVE RESIDENCY
          +
    PREDICTIVE PREFETCH
          +
    ADAPTIVE COMPRESSION
          +
    PARALLEL MATERIALIZATION
          +
    STATE/TOKEN GOVERNANCE
          =
    DYNAMIC NEURAL SUBSTRATE

NSA's role is to provide the state, governance, prediction, and transition machinery required to make that substrate observable, controllable, secure, and experimentally testable.
