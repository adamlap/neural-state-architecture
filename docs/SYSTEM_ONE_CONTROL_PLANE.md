# Continuous System 1 Control Plane

NSA now exposes System 1 as a reusable, generation-free control plane rather
than only as an action-pruning helper.

## Architecture

\`\`\`
environment / observations
          |
          v
   CanonicalState <-----------------------------+
          |                                    |
          v                                    |
   SystemOneController                         |
          |                                    |
    +-----+-----------+------------+-----------+
    |                 |            |
 tool routing    model routing  escalation
    |                 |            |
 memory policy    salience      System 2
    |                 |
    +---------> state signals
\`\`\`

System 1 may run continuously without invoking an LLM generation call. Its
outputs are typed decisions and soft control signals.

## Backend abstraction

The public contract is \`SystemOneBackend\`. The repository provides:

- \`DeterministicSystemOneBackend\`: zero-dependency fast path.
- \`CallableSystemOneBackend\`: adapter for Jev-like hosted decision services
  and other custom providers.
- \`FrozenCausalLMLogitBackend\`: optional ML backend that scores typed choices
  directly from a frozen Hugging Face causal LM's output logits.

The frozen LM backend never calls \`generate()\`, never trains, and never
requires gradients. It scores the normalized sequence log probability of each
choice and converts those scores into a typed probability distribution.

## Continuous cognition

\`NSARuntime\` can enable System 1 independently of generation:

\`\`\`python
runtime = NSARuntime(
    backend=backend,
    config=RuntimeConfig(system_one_enabled=True, continuous_enabled=True),
)

runtime.continuous_start()
\`\`\`

Every CCE heartbeat can evaluate:

- escalation to System 2;
- memory retention policy;
- salience;
- optional tool routing;
- optional model routing.

The resulting signal is observable through \`last_system_one_tick\` and the
explicit \`system_one_tick()\` API.

This is an engineering mechanism for persistent cognitive processing. It does
not make a scientific claim about subjective consciousness.

## Tool and model routing

\`\`\`python
decision = runtime.system_one_select_tool()
model = runtime.system_one_route_model(
    ["local-fast", "local-reasoning", "cloud-reasoning"]
)
\`\`\`

Tool routing only considers tools whose capability is already authorized by
the canonical hard state. A System 1 choice does not grant a capability.

Canonical CCE can also accept \`system_one=SystemOneController(...)\`. When no
custom selector is supplied, System 1 chooses among action candidates before
the existing policy, capability, safety, transition-validation and effect
boundaries.

## Open-weight models

A frozen causal LM can become System 1 without fine-tuning:

\`\`\`
state + typed question
        |
        v
frozen open-weight transformer
        |
        v
output logits
        |
        v
choice sequence log-probabilities
        |
        v
typed System 1 decision
\`\`\`

This makes NSA model-agnostic: Jev, a hosted service, a deterministic scorer,
or an open-weight local model can implement the same control-plane contract.

## Design boundary

System 1 is intelligence, not authority.

It may recommend:

- which tool to use;
- which model to use;
- whether System 2 is needed;
- what should remain in working memory;
- what deserves attention;
- which action candidate is best.

It may not:

- grant capabilities;
- mutate hard authority;
- bypass policy;
- bypass safety;
- execute effects directly;
- commit an unauthorized state transition.

Those remain responsibilities of the canonical CCE and trusted runtime.
