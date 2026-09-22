# Changelog

Format loosely follows [Keep a Changelog](https://keepachangelog.com/). Versions
before 0.6.0 were not documented here; see git history.

## [0.6.0] - 2026-09-22

### Breaking

- `CapabilityConstraintEvaluator` now defaults to `strict=True`: a capability
  constraint key it doesn't recognise (e.g. a typo like `max_rsik`) is now
  rejected instead of silently ignored. Pass `strict=False` to restore the
  old tolerant behaviour.
- `NSAPolicy.from_mapping()` now raises `ValueError` on a malformed policy
  document (a bare string where a list was required, an unknown top-level
  key now *warns*, a `prohibited` rule missing `category`, etc.) instead of
  silently loading a policy that is quietly weaker than the document says.
- `ImmutableSafetyKernel.evaluate_transition()` no longer accepts
  `valid_capability_supplied`. Nothing in the codebase ever passed it, and
  it let a caller assert "a capability was already verified upstream" with
  no actual verification. Use a real `supplied_capability` instead.
- The `ml`, `ml-residency` and `research` optional dependency groups now
  require `torch>=2.5.0` (previously `>=2.0.0`, sometimes with `<2.5.0`, an
  invalid combination against `transformers>=5`, which silently disables
  its PyTorch backend below torch 2.5).
- `nsa.server.proxy` (the OpenAI/Ollama-compatible HTTP server) no longer
  sends `Access-Control-Allow-Origin: *`. It now only ever echoes a loopback
  origin or one listed in `NSA_CORS_ORIGINS`. Set `NSA_API_TOKEN` to require
  `Authorization: Bearer <token>` on every request; unset keeps the
  previous open-by-default behaviour. See `docs/ollama_policy_server.md`.
- `PyTorchTransformersBackend`'s `trust_remote_code` now defaults to
  `False` (previously always `True`, executing code shipped in a model
  repository unconditionally).

### Added

- `nsa.semantic_classifier.ZeroShotSemanticClassifier`: a pretrained
  zero-shot `PolicyClassifier` (paraphrase backstop to
  `KeywordClassifier`'s exact substring matching), plus `HybridClassifier`
  to combine classifiers and `PolicyCompiler.compile_semantic(policy)` to
  wire them together. Requires the `ml` extra. See `docs/policy_interface.md`.
- `nsa.residency`: a neural virtual-memory subsystem (region-level weight
  placement, prediction and NVMe/page-cache prefetch for
  Accelerate-disk-offloaded models). Experimental; see
  `docs/NEURAL_RESIDENCY.md` and `docs/ACTIVE_RESIDENCY.md` for what is and
  isn't demonstrated on real hardware.
- `nsa.residency.page_cache.PageCacheProbe`: real OS page-cache residency
  checks via `mincore(2)`, used to avoid re-reading already-cached weight
  bytes during prefetch.
- Renewable capabilities: a token minted with a `max_calls` constraint is
  no longer burned on first use; it stays valid (subject to expiry and its
  own rate limit) across repeated calls. A token without `max_calls` is
  unaffected and stays one-shot.
- `NSA_API_TOKEN`, `NSA_CORS_ORIGINS`, `NSA_MAX_BODY_BYTES` environment
  variables for `nsa.server.proxy`.
- `CCECheckpointManager`/`nsa.runtime.cce_checkpoint.validate_checkpoint_id`:
  checkpoint ids are validated against a bounded charset before being used
  in a file path.
- `CapabilityAuthority.ephemeral()`: a fresh per-process random secret,
  used as the library default instead of the public, hardcoded demo
  secret. Constructing `CapabilityAuthority()` directly now warns.

### Fixed

- `nsa.algebra` (and therefore `import nsa`) no longer imports PyTorch at
  module load time; tensor helpers load it lazily on first use.
- A capability token covering more than one required capability was
  consumed/reserved once per requirement instead of once, so it always
  failed on the second requirement. Fixed in both the sync and async
  transaction engines.
- A cancelled async effect (`asyncio.CancelledError`) used to leave its
  capability reservation permanently held. Reservations are now released
  on any `BaseException`, not just `Exception`.
- `state_digest()`/`dumps_state()` depended on Python's per-process hash
  seed whenever semantic state contained a `set`, so the same canonical
  state could hash differently across restarts, breaking journal/
  checkpoint integrity verification. Sets are now serialised in sorted
  order.
- `ContinuousCognitiveEngine.tick()` could silently discard a `set_state()`
  call (e.g. a checkpoint restore) that landed while a transition was in
  flight. `stop(timeout=...)` that timed out while the loop thread was
  still running could let `start()` spawn a second loop thread.
- `StateCheckpointStore.save()` and `CCECheckpointManager`'s atomic writer
  used a shared, non-unique temp file name and could leave orphaned temp
  files behind on failure.
- `TwoPhaseExecutor.commit()`'s `return` inside a `finally` block silently
  discarded any exception raised by `abort()`, including
  `KeyboardInterrupt`/`SystemExit`.
- `PolicyEngine.evaluate()` returned on the first matched category, so an
  earlier `escalate` category could mask a later `deny`. The most
  restrictive outcome across every matched category now wins.
  `unknown_policy` is now actually applied to categories the policy
  doesn't describe (previously dead configuration).
- `KeywordClassifier` now normalises Unicode width/case/invisible format
  characters and whitespace before matching, closing a trivial evasion
  (full-width letters, a zero-width joiner mid-word, a pattern split
  across a newline instead of a space).
- `/api/cce/checkpoint` accepted a client-supplied `checkpoint_id`
  containing path separators, allowing a write outside the checkpoint
  directory.
- Residency: prefetch scoring could never reach its own threshold in the
  common case (no semantic-tag evidence was scored as "irrelevant" rather
  than "unknown"); the prefetcher only understood one of several real
  Accelerate offload index formats; the residency cache double-counted a
  region moved between tiers and could silently evict its own contents on
  an oversize entry; `ActiveResidencyController.prefetch_async()` could
  deadlock when a prefetch future completed instantly.
- Residency prefetch overhead: measured and fixed across three iterations
  (naive re-read, mincore-per-call, mincore-with-cached-mmap-and-
  scheduling-precheck). See `docs/ACTIVE_RESIDENCY.md` for the numbers.
