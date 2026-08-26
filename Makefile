.DEFAULT_GOAL := help
PYTHON ?= python3
export PYTHONPATH := .

.PHONY: help install install-dev test test-core test-cce build clean \
        demo serve serve-cce serve-ollama chat-ollama \
        benchmark benchmark-nsa63 benchmark-nsa64 benchmark-ollama \
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

serve: serve-ollama ## Start the default local Ollama server
serve-cce: serve-ollama ## Backwards-compatible CCE server alias
serve-ollama: ## Start the existing OpenAI/Ollama-compatible NSA server
	$(PYTHON) -m nsa.server.proxy --backend ollama --model $${NSA_MODEL:-qwen2.5:3b} --port $${NSA_PORT:-8000}

chat-ollama: ## Start the existing interactive Ollama cognitive demo
	$(PYTHON) experiments/nsa62/interactive_chat.py --backend ollama --model $${NSA_MODEL:-qwen2.5:3b}

benchmark: benchmark-nsa64 ## Primary research benchmark
benchmark-nsa64: ## Run NSA 6.4 replication/falsification benchmark
	$(PYTHON) experiments/nsa64/falsification_suite.py --trials $${NSA_TRIALS:-20}
benchmark-nsa63: ## Run NSA 6.3 scientific validation suite
	$(PYTHON) experiments/nsa63/scientific_validation_suite.py --backend mock --trials $${NSA_TRIALS:-40}
benchmark-ollama: ## Run canonical live NSA 6.3 Ollama benchmark
	$(PYTHON) experiments/nsa63/scientific_validation_suite.py --backend ollama --model $${NSA_MODEL:-qwen2.5:3b} --trials $${NSA_TRIALS:-20}
benchmark-live: benchmark-ollama ## Backwards-compatible live benchmark alias

evidence: ## Validate the machine-readable evidence manifest
	$(PYTHON) evidence/validate_evidence.py

research: evidence benchmark ## Validate evidence and run the primary benchmark

clean: ## Remove generated Python/build artifacts
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	find . -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
	rm -rf build dist *.egg-info
