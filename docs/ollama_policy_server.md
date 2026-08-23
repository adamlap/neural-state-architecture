# Ollama / CCE server with an NSA safety policy

Start the existing server without a policy:

```bash
make serve-ollama
```

Start the continuous CCE server:

```bash
make serve-cce
```

Apply a policy to either:

```bash
make serve-ollama POLICY=examples/policies/safe_assistant.json
make serve-cce POLICY=examples/policies/safe_assistant.json
```

Model and port are configurable:

```bash
make serve-cce NSA_MODEL=qwen2.5:3b NSA_PORT=8000 POLICY=examples/policies/safe_assistant.json
```

The launcher evaluates the latest user message before inference and evaluates generated output again before returning it. The response contains an `nsa_policy` audit object.

`patterns` in the reference JSON policies are deliberately a deterministic demo mechanism, not a claim of semantic understanding. Replace the classifier with a trusted semantic/normative component for production deployments.

The policy boundary does not by itself prove that an LLM cannot internally represent prohibited knowledge. Strong structural guarantees require the classifier, trusted runtime, capability boundary and model integration to be included in the security threat model.
