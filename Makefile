.DEFAULT_GOAL := help
PYTHON ?= python3
UV := $(shell command -v uv 2>/dev/null || echo "")
export PYTHONPATH := .

SAFETY_POLICY ?= $(POLICY)
NSA_POLICY ?= $(SAFETY_POLICY)
NSA_MODEL ?= qwen2.5:3b
NSA_PORT ?= 8000
NSA_BACKEND_URL ?=
CCE ?= 1

OLLAMA_MODEL ?= qwen2.5:3b
OUT ?= results/nsa64/ollama-$(subst :,-,$(OLLAMA_MODEL))
TRIALS ?= 20
DEV_SEEDS ?= 7 17 37 73 137
HELDOUT_SEEDS ?= 101 211 307 401 509
HYPOTHESES ?= 2 4 8 16
NOISE ?= 0.0 0.05 0.10 0.20 0.30

define RUN_NSA_SERVER
	@if [ "$(UV_EXISTS)" = "yes" ]; then \
		PYTHONPATH=. $(UV) run python scripts/policy_server.py $(1) --model $(NSA_MODEL) --port $(NSA_PORT) $(if $(NSA_BACKEND_URL),--backend-url $(NSA_BACKEND_URL),) $(if $(NSA_POLICY),--policy $(NSA_POLICY),) $(if $(filter 0 false no,$(CCE)),--no-cce,); \
	else \
		PYTHONPATH=. $(PYTHON) scripts/policy_server.py $(1) --model $(NSA_MODEL) --port $(NSA_PORT) $(if $(NSA_BACKEND_URL),--backend-url $(NSA_BACKEND_URL),) $(if $(NSA_POLICY),--policy $(NSA_POLICY),) $(if $(filter 0 false no,$(CCE)),--no-cce,); \
	fi
endef

# Ollama is frequently installed user-locally (no sudo) rather than on PATH.
OLLAMA_BIN := $(shell command -v ollama 2>/dev/null || { [ -x "$$HOME/.local/ollama/bin/ollama" ] && echo "$$HOME/.local/ollama/bin/ollama"; })
OLLAMA_HOST ?= 127.0.0.1:11434

.PHONY: help install install-dev test test-core test-cce build clean \
        demo serve serve-cce serve-ollama serve-lmstudio serve-cce-policy \
		serve-ollama-policy serve-lmstudio-policy chat-ollama \
        benchmark benchmark-nsa63 benchmark-nsa64 benchmark-ollama \
		benchmark-nsa64-ollama benchmark-nsa64-ollama-smoke \
        benchmark-live evidence research

help: ## Show the supported developer commands
	@printf '\nNSA developer commands\n\n'
	@printf '  make install       Install the runtime package\n'
	@printf '  make install-dev   Install runtime + development dependencies\n'
	@printf '  make test          Run the complete regression suite\n'
	@printf '  make demo          Run the deterministic runtime demo\n'
	@printf '  make serve         Start the NSA Ollama-compatible server\n'
	@printf '  make benchmark     Run the primary NSA 6.4 research benchmark\n'
	@printf '  make research      Validate evidence and run the primary benchmark\n'
	@printf '  make build         Build wheel and source distribution\n\n'

install: ## Install the NSA runtime package
	$(PYTHON) -m pip install -e .

install-dev: ## Install runtime and development dependencies
	$(PYTHON) -m pip install -e '.[dev]'

test: ## Run the complete regression suite
	$(PYTHON) -m pytest -q

test-core: ## Run public runtime/API regression tests
	$(PYTHON) -m pytest -q tests/test_agent.py

test-cce: ## Run CCE/runtime regression tests
	$(PYTHON) -m pytest -q tests -k 'cce or runtime'

build: ## Build wheel and source distribution
	$(PYTHON) -m pip install --upgrade build
	$(PYTHON) -m build

demo: ## Run the deterministic state-aware runtime demo
	$(PYTHON) examples/quickstart.py

# =========================================
# -------------- Serving ------------------
# =========================================
serve: serve-ollama ## Start the default local Ollama server

serve-cce: serve-ollama ## Backwards-compatible CCE server alias

serve-ollama: ## Start the existing OpenAI/Ollama-compatible NSA server
	$(PYTHON) -m nsa.server.proxy --backend ollama --model $${NSA_MODEL:-qwen2.5:3b} --port $${NSA_PORT:-8000}

serve-lmstudio: ## Start the OpenAI/LMStudio-compatible NSA server
	$(PYTHON) -m nsa.server.proxy --backend lmstudio --model $${NSA_MODEL:-qwen2.5:3b} --port $${NSA_PORT:-8000}

serve-cce-policy: ## Launch Ollama-backed CCE server with optional SAFETY_POLICY=policies/strict.yaml
	$(call RUN_NSA_SERVER,--backend ollama)

serve-ollama-policy: ## Launch Ollama-backed NSA server with optional SAFETY_POLICY=policies/strict.yaml
	$(call RUN_NSA_SERVER,--backend ollama)

serve-lmstudio-policy: ## Launch LM Studio-backed NSA server with optional SAFETY_POLICY=policies/strict.yaml
	$(call RUN_NSA_SERVER,--backend lmstudio)

chat-ollama: ## Start the existing interactive Ollama cognitive demo
	$(PYTHON) experiments/nsa62/interactive_chat.py --backend ollama --model $${NSA_MODEL:-qwen2.5:3b}

# =========================================
# -------------- Benchmarks ---------------
# =========================================
benchmark: benchmark-nsa64 ## Primary research benchmark

benchmark-nsa64: ## Run NSA 6.4 replication/falsification benchmark
	$(PYTHON) experiments/nsa64/falsification_suite.py --trials $${NSA_TRIALS:-20}

benchmark-nsa63: ## Run NSA 6.3 scientific validation suite
	$(PYTHON) experiments/nsa63/scientific_validation_suite.py --backend mock --trials $${NSA_TRIALS:-40}
benchmark-ollama: ## Run canonical live NSA 6.3 Ollama benchmark
	$(PYTHON) experiments/nsa63/scientific_validation_suite.py --backend ollama --model $${NSA_MODEL:-qwen2.5:3b} --trials $${NSA_TRIALS:-20}
benchmark-live: benchmark-ollama ## Backwards-compatible live benchmark alias

benchmark-nsa64-ollama: ## Run the full NSA 6.4 live replication matrix against local Ollama
	@[ -n "$(OLLAMA_BIN)" ] || { echo "ERROR: Ollama is not installed or not on PATH (checked \$$PATH and ~/.local/ollama/bin/ollama)."; exit 1; }
	@curl -sf "http://$(OLLAMA_HOST)/api/tags" >/dev/null 2>&1 || { echo "ERROR: Ollama server not reachable at $(OLLAMA_HOST). Start it with: $(OLLAMA_BIN) serve &"; exit 1; }
	@$(OLLAMA_BIN) list | grep -q "^$(OLLAMA_MODEL)" || { echo "ERROR: $(OLLAMA_MODEL) is not installed. Run: $(OLLAMA_BIN) pull $(OLLAMA_MODEL)"; exit 1; }
	@mkdir -p "$(OUT)"
	@if [ -n "$(UV)" ]; then \
		PYTHONPATH=. $(UV) run python experiments/nsa64/replication_matrix.py \
			--backend ollama --models $(OLLAMA_MODEL) \
			--dev-seeds $(DEV_SEEDS) --heldout-seeds $(HELDOUT_SEEDS) \
			--hypotheses $(HYPOTHESES) --noise $(NOISE) --trials $(TRIALS) --out "$(OUT)"; \
	else \
		PYTHONPATH=. $(PYTHON) experiments/nsa64/replication_matrix.py \
			--backend ollama --models $(OLLAMA_MODEL) \
			--dev-seeds $(DEV_SEEDS) --heldout-seeds $(HELDOUT_SEEDS) \
			--hypotheses $(HYPOTHESES) --noise $(NOISE) --trials $(TRIALS) --out "$(OUT)"; \
	fi

benchmark-nsa64-ollama-smoke: ## Run a small live Ollama smoke test before the full matrix
	$(MAKE) -f Makefile.nsa64 benchmark-nsa64-ollama \
		TRIALS=2 DEV_SEEDS="7" HELDOUT_SEEDS="101" HYPOTHESES="2" NOISE="0.0 0.3" \
		OUT=results/nsa64/ollama-smoke



# =========================================
# ---------------- Other ------------------
# =========================================
evidence: ## Validate the machine-readable evidence manifest
	$(PYTHON) evidence/validate_evidence.py

research: evidence benchmark ## Validate evidence and run the primary benchmark

policy-validate: ## Validate a policy: make policy-validate SAFETY_POLICY=policies/strict.yaml
	@test -n "$(SAFETY_POLICY)" || (echo "SAFETY_POLICY is required" && exit 2)
	@if [ "$(UV_EXISTS)" = "yes" ]; then PYTHONPATH=. $(UV) run python -m nsa.policy_cli validate $(SAFETY_POLICY); else PYTHONPATH=. $(PYTHON) -m nsa.policy_cli validate $(SAFETY_POLICY); fi

policy-inspect: ## Inspect a policy: make policy-inspect SAFETY_POLICY=policies/strict.yaml
	@test -n "$(SAFETY_POLICY)" || (echo "SAFETY_POLICY is required" && exit 2)
	@if [ "$(UV_EXISTS)" = "yes" ]; then PYTHONPATH=. $(UV) run python -m nsa.policy_cli inspect $(SAFETY_POLICY); else PYTHONPATH=. $(PYTHON) -m nsa.policy_cli inspect $(SAFETY_POLICY); fi


clean: ## Remove generated Python/build artifacts
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	find . -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
	rm -rf build dist *.egg-info
