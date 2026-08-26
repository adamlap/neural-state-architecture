# Ollama / CCE server with an NSA safety policy

The repository now exposes the practical safety path through `make`.

## Start Ollama without a policy

```bash
make serve-ollama
```

This uses the default `qwen2.5:3b` model on port `8000`.

## Start the continuous CCE server

```bash
make serve-cce
```

CCE remains enabled by default and advances its state on wall-clock time.

## Select a safety policy

The `POLICY` variable is optional:

```bash
make serve-ollama POLICY=examples/policies/safe_assistant.json
```

CCE can use exactly the same policy:

```bash
make serve-cce POLICY=examples/policies/safe_assistant.json
```

Equivalent environment-variable form:

```bash
NSA_POLICY=examples/policies/safe_assistant.json make serve-ollama
```

The model and port are also configurable:

```bash
make serve-cce \
  MODEL=qwen2.5:3b \
  POLICY=examples/policies/safe_assistant.json \
  PORT=8000
```

The preferred variable names are `NSA_MODEL` and `NSA_PORT`; `MODEL`/`PORT` can
be adopted by local wrappers, but the canonical interface is:

```bash
make serve-cce NSA_MODEL=qwen2.5:3b NSA_PORT=8000 POLICY=examples/policies/safe_assistant.json
```

## What the policy does

The launcher evaluates the latest user message **before inference**. A denied
or escalated request therefore does not reach Ollama. Allowed requests are sent
to the existing NSA runtime/CCE stack, and the generated output is evaluated
again before it is returned.

The response includes an `nsa_policy` audit object containing the typed policy
decision summary and whether enforcement occurred at the request or output
boundary.

## Policy format

Rules may contain explicit `patterns` for the deterministic reference
classifier:

```json
{
  "name": "my-policy",
  "prohibited": [
    {
      "category": "example_category",
      "mode": "deny",
      "patterns": ["example phrase"],
      "reason": "Configured reason"
    }
  ],
  "unknown_policy": "escalate",
  "default_uncertainty": "escalate"
}
```

`patterns` are intentionally a reference mechanism, **not a claim of semantic
understanding**. For production use, the same `PolicyEngine` can be supplied a
trained semantic classifier. The hard capability/runtime boundary remains
separate from the classifier.

## Important security boundary

This integration currently provides **request and output policy enforcement**.
It does not make a keyword classifier intrinsically intelligent, and it does
not by itself prove that an LLM cannot internally represent prohibited
knowledge. Strong guarantees require the classifier, trusted runtime,
capability boundary, and model integration to be included explicitly in the
threat model.

## Verified 2026-08-26

`make serve-cce`/`make serve-ollama` (via `GNUmakefile` + `scripts/policy_server.py`)
were exercised end-to-end against real Ollama models (qwen2.5:0.5b, 1.5b, 3b),
both with and without a policy (JSON and YAML), confirming `NSA_MODEL`,
`NSA_PORT` and `POLICY`/`SAFETY_POLICY`/`NSA_POLICY` all work as documented
above. Two bugs found and fixed in the process:

- `nsa/server/proxy.py`'s HTTP handler built the `/api/chat` and
  `/v1/chat/completions` JSON responses from an explicit allow-list of keys
  that did not include `nsa_policy` -- the audit object this document
  describes was computed correctly internally but never reached the client.
- `scripts/policy_server.py`'s request-blocked early return used a different
  response shape (`nsa.policy` = a bare decision summary) than the
  full-response path (`nsa_policy` = `{request, output, enforcement}`), so a
  denied request was indistinguishable from "no policy configured" to a
  client only checking for `nsa_policy`.

Both are fixed; `tests/test_policy_server.py` has regression coverage for the
request-blocked and allowed shapes.
