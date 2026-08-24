# Policy-aware Make entrypoint. GNU Make prefers GNUmakefile automatically.
# The existing Makefile remains the canonical command suite; this thin layer
# adds optional SAFETY_POLICY/POLICY forwarding for inference server targets.
include Makefile

SAFETY_POLICY ?= $(POLICY)
NSA_POLICY ?= $(SAFETY_POLICY)
NSA_MODEL ?= qwen2.5:3b
NSA_PORT ?= 8000
NSA_BACKEND_URL ?=
CCE ?= 1

.PHONY: serve-cce serve-ollama serve-lmstudio policy-validate policy-inspect

define RUN_NSA_SERVER
	@if [ "$(UV_EXISTS)" = "yes" ]; then \
		PYTHONPATH=. $(UV) run python scripts/policy_server.py $(1) --model $(NSA_MODEL) --port $(NSA_PORT) $(if $(NSA_BACKEND_URL),--backend-url $(NSA_BACKEND_URL),) $(if $(NSA_POLICY),--policy $(NSA_POLICY),) $(if $(filter 0 false no,$(CCE)),--no-cce,); \
	else \
		PYTHONPATH=. $(PYTHON) scripts/policy_server.py $(1) --model $(NSA_MODEL) --port $(NSA_PORT) $(if $(NSA_BACKEND_URL),--backend-url $(NSA_BACKEND_URL),) $(if $(NSA_POLICY),--policy $(NSA_POLICY),) $(if $(filter 0 false no,$(CCE)),--no-cce,); \
	fi
endef

serve-cce: ## Launch Ollama-backed CCE server with optional SAFETY_POLICY=policies/strict.yaml
	$(call RUN_NSA_SERVER,--backend ollama)

serve-ollama: ## Launch Ollama-backed NSA server with optional SAFETY_POLICY=policies/strict.yaml
	$(call RUN_NSA_SERVER,--backend ollama)

serve-lmstudio: ## Launch LM Studio-backed NSA server with optional SAFETY_POLICY=policies/strict.yaml
	$(call RUN_NSA_SERVER,--backend lmstudio)

policy-validate: ## Validate a policy: make policy-validate SAFETY_POLICY=policies/strict.yaml
	@test -n "$(SAFETY_POLICY)" || (echo "SAFETY_POLICY is required" && exit 2)
	@if [ "$(UV_EXISTS)" = "yes" ]; then PYTHONPATH=. $(UV) run python -m nsa.policy_cli validate $(SAFETY_POLICY); else PYTHONPATH=. $(PYTHON) -m nsa.policy_cli validate $(SAFETY_POLICY); fi

policy-inspect: ## Inspect a policy: make policy-inspect SAFETY_POLICY=policies/strict.yaml
	@test -n "$(SAFETY_POLICY)" || (echo "SAFETY_POLICY is required" && exit 2)
	@if [ "$(UV_EXISTS)" = "yes" ]; then PYTHONPATH=. $(UV) run python -m nsa.policy_cli inspect $(SAFETY_POLICY); else PYTHONPATH=. $(PYTHON) -m nsa.policy_cli inspect $(SAFETY_POLICY); fi
