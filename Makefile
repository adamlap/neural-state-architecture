# ==============================================================================
# Makefile for Neural State Architecture (NSA) - Powered by uv
# ==============================================================================

.PHONY: help install install-dev install-uv venv test experiment leakage-experiment multi-tier pretrain-lm pillar-1 benchmark-gpu pillar-2 retrofit-lora pillar-3 prompt-injection pillar-4 open-llm-retrofit llama-showcase showcase benchmarks prototype lint format clean summary

# Locate uv executable (PATH, ~/.local/bin/uv, or ~/.cargo/bin/uv)
UV := $(shell command -v uv 2>/dev/null || (test -f ~/.local/bin/uv && echo ~/.local/bin/uv) || (test -f ~/.cargo/bin/uv && echo ~/.cargo/bin/uv) || echo "uv")
UV_EXISTS := $(shell command -v $(UV) >/dev/null 2>&1 && echo yes || echo no)

PYTHON ?= python3
VENV_DIR ?= .venv

# Default target
.DEFAULT_GOAL := help

help: ## Display this help message
	@echo "Neural State Architecture (NSA) - Build & Utility Commands"
	@echo "=========================================================="
	@if [ "$(UV_EXISTS)" = "yes" ]; then \
		echo " [Engine: uv ($(UV))]"; \
	else \
		echo " [Engine: standard python3/pip (run 'make install-uv' to switch to uv)]"; \
	fi
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install-uv: ## Install uv package manager locally
	@curl -LsSf https://astral.sh/uv/install.sh | sh || $(PYTHON) -m pip install --user uv

venv: ## Create virtual environment using uv or venv
	@if [ "$(UV_EXISTS)" = "yes" ]; then \
		$(UV) venv $(VENV_DIR); \
	else \
		$(PYTHON) -m venv $(VENV_DIR); \
	fi

install: ## Install runtime requirements
	@if [ "$(UV_EXISTS)" = "yes" ]; then \
		$(UV) pip install -r prototype/requirements.txt; \
	else \
		$(PYTHON) -m pip install -r prototype/requirements.txt; \
	fi

install-dev: install ## Install runtime and development dependencies
	@if [ "$(UV_EXISTS)" = "yes" ]; then \
		$(UV) pip install pytest black ruff mypy; \
	else \
		$(PYTHON) -m pip install pytest black ruff mypy; \
	fi

test: ## Run unit tests
	@if [ "$(UV_EXISTS)" = "yes" ]; then \
		if $(UV) run python -m pytest --version >/dev/null 2>&1; then \
			$(UV) run python -m pytest -v tests/; \
		else \
			$(UV) run python -m unittest discover -s tests -p "test_*.py"; \
		fi; \
	elif $(PYTHON) -m pytest --version >/dev/null 2>&1; then \
		PYTHONPATH=. $(PYTHON) -m pytest -v tests/; \
	else \
		PYTHONPATH=. $(PYTHON) -m unittest discover -s tests -p "test_*.py"; \
	fi

experiment: ## Run toy experiment (Baseline vs NSA)
	@if [ "$(UV_EXISTS)" = "yes" ]; then \
		$(UV) run python prototype/toy_experiment.py; \
	else \
		PYTHONPATH=. $(PYTHON) prototype/toy_experiment.py; \
	fi

leakage-experiment: ## Run adversarial data leakage extraction benchmark
	@if [ "$(UV_EXISTS)" = "yes" ]; then \
		$(UV) run python prototype/leakage_attack.py; \
	else \
		PYTHONPATH=. $(PYTHON) prototype/leakage_attack.py; \
	fi

multi-tier: ## Run multi-tier security lattice benchmark
	@if [ "$(UV_EXISTS)" = "yes" ]; then \
		$(UV) run python prototype/multi_tier_experiment.py; \
	else \
		PYTHONPATH=. $(PYTHON) prototype/multi_tier_experiment.py; \
	fi

pretrain-lm: ## Run Pillar 1 Causal Language Model zero-degradation benchmark
	@if [ "$(UV_EXISTS)" = "yes" ]; then \
		$(UV) run python prototype/pretrain_lm.py; \
	else \
		PYTHONPATH=. $(PYTHON) prototype/pretrain_lm.py; \
	fi

pillar-1: pretrain-lm ## Run Pillar 1 validation suite

benchmark-gpu: ## Run Pillar 2 Fused GPU Attention throughput benchmark
	@if [ "$(UV_EXISTS)" = "yes" ]; then \
		$(UV) run python prototype/benchmark_gpu.py; \
	else \
		PYTHONPATH=. $(PYTHON) prototype/benchmark_gpu.py; \
	fi

pillar-2: benchmark-gpu ## Run Pillar 2 validation suite

retrofit-lora: ## Run Pillar 3 NSA-LoRA post-hoc retrofitting benchmark
	@if [ "$(UV_EXISTS)" = "yes" ]; then \
		$(UV) run python prototype/retrofit_lora.py; \
	else \
		PYTHONPATH=. $(PYTHON) prototype/retrofit_lora.py; \
	fi

pillar-3: retrofit-lora ## Run Pillar 3 validation suite

prompt-injection: ## Run Pillar 4 Empirical Red-Teaming Prompt Injection Firewall benchmark
	@if [ "$(UV_EXISTS)" = "yes" ]; then \
		$(UV) run python prototype/prompt_injection_bench.py; \
	else \
		PYTHONPATH=. $(PYTHON) prototype/prompt_injection_bench.py; \
	fi

pillar-4: prompt-injection ## Run Pillar 4 validation suite

open-llm-retrofit: ## Run Phase 3 open LLM scale retrofitting simulation benchmark
	@if [ "$(UV_EXISTS)" = "yes" ]; then \
		$(UV) run python prototype/open_llm_retrofit.py; \
	else \
		PYTHONPATH=. $(PYTHON) prototype/open_llm_retrofit.py; \
	fi

llama-showcase: ## Run interactive Llama retrofitting security showcase
	@if [ "$(UV_EXISTS)" = "yes" ]; then \
		$(UV) run python prototype/llama_security_showcase.py; \
	else \
		PYTHONPATH=. $(PYTHON) prototype/llama_security_showcase.py; \
	fi

showcase: llama-showcase ## Alias for llama-showcase

benchmarks: experiment leakage-experiment multi-tier pretrain-lm benchmark-gpu retrofit-lora prompt-injection open-llm-retrofit llama-showcase ## Run full NSA benchmark suite

prototype: ## Run prototype demonstration script
	@if [ "$(UV_EXISTS)" = "yes" ]; then \
		$(UV) run python prototype/state_transformer.py; \
	else \
		PYTHONPATH=. $(PYTHON) prototype/state_transformer.py; \
	fi

summary: ## Print default state lattice structure and model info
	@if [ "$(UV_EXISTS)" = "yes" ]; then \
		$(UV) run python -c "from nsa import DEFAULT_LATTICE, print_lattice; print_lattice(DEFAULT_LATTICE)"; \
	else \
		PYTHONPATH=. $(PYTHON) -c "from nsa import DEFAULT_LATTICE, print_lattice; print_lattice(DEFAULT_LATTICE)"; \
	fi

lint: ## Check code for syntax errors and style issues
	@if [ "$(UV_EXISTS)" = "yes" ]; then \
		$(UV) run ruff check nsa/ prototype/ tests/ 2>/dev/null || $(UV) run python -m compileall nsa prototype tests; \
	elif command -v ruff >/dev/null 2>&1; then \
		ruff check nsa/ prototype/ tests/; \
	else \
		PYTHONPATH=. $(PYTHON) -m compileall nsa prototype tests; \
	fi

format: ## Auto-format code using black or ruff
	@if [ "$(UV_EXISTS)" = "yes" ]; then \
		$(UV) run ruff format nsa/ prototype/ tests/ 2>/dev/null || $(UV) run black nsa/ prototype/ tests/ 2>/dev/null || echo "Install ruff or black to format code."; \
	elif command -v black >/dev/null 2>&1; then \
		black nsa/ prototype/ tests/; \
	else \
		echo "Neither black nor ruff found. Please install black or ruff for formatting."; \
	fi

clean: ## Clean up build artifacts, cache files, and bytecode
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".uv_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
