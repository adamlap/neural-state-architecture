# Policy-aware Make entrypoint. GNU Make prefers GNUmakefile automatically.
# The existing Makefile remains the canonical command suite; this thin layer
# adds optional POLICY/NSA_POLICY forwarding for inference server targets.
include Makefile

NSA_POLICY ?= $(POLICY)
NSA_MODEL ?= qwen2.5:3b
NSA_PORT ?= 8000
NSA_BACKEND_URL ?=

.PHONY: serve-cce serve-ollama serve-lmstudio

# Preserve the repository's uv-first execution convention.
define RUN_NSA_SERVER
	@if [ "$(UV_EXISTS)" = "yes" ]; then \
		PYTHONPATH=. $(UV) run python scripts/policy_server.py $(1) --model $(NSA_MODEL) --port $(NSA_PORT) $(if $(NSA_BACKEND_URL),--backend-url $(NSA_BACKEND_URL),) $(if $(NSA_POLICY),--policy $(NSA_POLICY),); \
	else \
		PYTHONPATH=. $(PYTHON) scripts/policy_server.py $(1) --model $(NSA_MODEL) --port $(NSA_PORT) $(if $(NSA_BACKEND_URL),--backend-url $(NSA_BACKEND_URL),) $(if $(NSA_POLICY),--policy $(NSA_POLICY),); \
	fi
endef

serve-cce: ## Launch Ollama-backed CCE server with optional NSA safety policy
	$(call RUN_NSA_SERVER,--backend ollama)

serve-ollama: ## Launch Ollama-backed NSA server with optional safety policy
	$(call RUN_NSA_SERVER,--backend ollama)

serve-lmstudio: ## Launch LM Studio-backed NSA server with optional safety policy
	$(call RUN_NSA_SERVER,--backend lmstudio)
