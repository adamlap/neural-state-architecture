# Mainline Implementation Audit — 2026-08-24

This audit compares `PLAN.md` completion markers with executable implementation and evidence status.

- **Phase 11:** canonical typed state is implemented and tested, but `docs/PLAN_PHASE11_STATUS.md` explicitly lists legacy StateVector/MultiStateVector/quad-tuple compatibility and native/retrofit adapters as remaining work. Treat Phase 11 as foundation-complete, not fully complete.
- **Phases 20–24:** executable implementations exist, but the evidence manifest classifies the combined governance claim as `UNIT-TESTED`, not empirical or robust validation. A completed implementation is not the same as a validated safety claim.
- **Phase 25:** auditing/recovery foundations exist; detection delay, distribution shift, anomaly detection, automated recovery policy and recovery proofs remain open.
- **Phase 26:** evidence/epistemic tooling and formal reachability components exist, but the full machine-checkable TCB/model-checking/non-interference/capability-proof checklist remains open.
- **Phases 27–30:** adversarial infrastructure and integrations exist, but broad attack coverage, joint capability/safety evaluation, production kernel optimization and ecosystem integration remain open.
- **CCE:** continuous maintenance, live model heartbeat, authority-preservation checks and live evaluation infrastructure exist. Multi-seed, multi-model, long-duration, matched-compute and asynchronous deployment evidence remains open.

## Critical implementation issue

`ToolGovernor.register_tool()` accepted `required_authorizations` and a `FlowGraph`, but `execute()` previously checked neither. This meant the public action-governance API advertised authorization/flow enforcement that was not actually enforced at the tool boundary.

The correctness PR makes both checks executable and adds regression tests proving that a tool handler is not called when either requirement fails.

## Evidence rule

The evidence manifest is the source of epistemic status; `PLAN.md` tracks engineering scope. Safety claims must not be promoted from implementation status to empirical or formal validation without the corresponding evidence.
