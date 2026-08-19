# ==============================================================================
# Makefile for Neural State Architecture (NSA) - Powered by uv
# ==============================================================================

.PHONY: help install install-dev install-uv venv test showcase demo eval-security eval-perf lint format clean showcase-web

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
		$(UV) pip install -r requirements.txt; \
	else \
		$(PYTHON) -m pip install -r requirements.txt; \
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

showcase: ## Run interactive Llama retrofitting security showcase
	@if [ "$(UV_EXISTS)" = "yes" ]; then \
		$(UV) run python demo/cli_showcase.py; \
	else \
		PYTHONPATH=. $(PYTHON) demo/cli_showcase.py; \
	fi

showcase-web: ## Open the interactive NSA web showcase (static HTML, no Python server needed)
	@echo "Opening NSA Interactive Showcase…"
	@echo "  → Serving at http://localhost:8080"
	@echo "  → Press Ctrl+C to stop"
	@cd showcase && $(PYTHON) -m http.server 8080

demo: ## Launch interactive Gradio web showcase UI
	@if [ "$(UV_EXISTS)" = "yes" ]; then \
		$(UV) run python demo/web_demo.py; \
	else \
		PYTHONPATH=. $(PYTHON) demo/web_demo.py; \
	fi

demo-dpo: ## Launch Gradio web showcase UI for the DPO-aligned model
	@if [ "$(UV_EXISTS)" = "yes" ]; then \
		$(UV) run python demo/web_demo_dpo.py; \
	else \
		PYTHONPATH=. $(PYTHON) demo/web_demo_dpo.py; \
	fi

train-dpo: ## Run the NSA-DPO prototype training script
	@if [ "$(UV_EXISTS)" = "yes" ]; then \
		$(UV) run python prototype/retrofit/nsa_dpo_train.py; \
	else \
		PYTHONPATH=. $(PYTHON) prototype/retrofit/nsa_dpo_train.py; \
	fi

train-audit: ## Run the functional training for the Speculative State Auditor
	@if [ "$(UV_EXISTS)" = "yes" ]; then \
		PYTHONIOENCODING=utf-8 $(UV) run python prototype/retrofit/nsa_auditor_train.py; \
	else \
		PYTHONPATH=. PYTHONIOENCODING=utf-8 $(PYTHON) prototype/retrofit/nsa_auditor_train.py; \
	fi

test-verifier: ## Run NSA 2.0 Speculative Verifier test suite
	@if [ "$(UV_EXISTS)" = "yes" ]; then \
		$(UV) run python -m pytest -v tests/test_verifier_nsa2.py; \
	else \
		PYTHONPATH=. $(PYTHON) -m pytest -v tests/test_verifier_nsa2.py; \
	fi

eval-security: ## Run NL multi-attack / AdvGLUE-style label firewall suite
	@if [ "$(UV_EXISTS)" = "yes" ]; then \
		$(UV) run python eval/security_eval.py; \
	else \
		PYTHONPATH=. $(PYTHON) eval/security_eval.py; \
	fi

eval-perf: ## Run Fused GPU Attention throughput benchmark
	@if [ "$(UV_EXISTS)" = "yes" ]; then \
		$(UV) run python eval/performance_bench.py; \
	else \
		PYTHONPATH=. $(PYTHON) eval/performance_bench.py; \
	fi

exp-self-state: ## Run Self-State perturbation sweep & statistical summary
	@echo "Running NSA Self-State Perturbation Sweep..."
	@mkdir -p results
	@if [ "$(UV_EXISTS)" = "yes" ]; then \
		$(UV) run python experiments/self_state/perturbation_sweep.py --seed 42 | tee results/self-state-sweep.json; \
		$(UV) run python experiments/self_state/summarize_sweep.py results/self-state-sweep.json; \
	else \
		PYTHONPATH=. $(PYTHON) experiments/self_state/perturbation_sweep.py --seed 42 | tee results/self-state-sweep.json; \
		PYTHONPATH=. $(PYTHON) experiments/self_state/summarize_sweep.py results/self-state-sweep.json; \
	fi

exp-hard-state: ## Run Adversarial Hard-State Integrity experiment
	@echo "Running NSA Hard-State Attack Experiment..."
	@mkdir -p results
	@if [ "$(UV_EXISTS)" = "yes" ]; then \
		$(UV) run python experiments/safety/hard_state_attack.py --seed 42 --attack 10.0; \
	else \
		PYTHONPATH=. $(PYTHON) experiments/safety/hard_state_attack.py --seed 42 --attack 10.0; \
	fi

exp-local-contraction: ## Run Local Self-State Contraction experiment
	@echo "Running NSA Local Contraction Analysis..."
	@if [ "$(UV_EXISTS)" = "yes" ]; then \
		$(UV) run python experiments/self_state/local_contraction.py --seed 42; \
	else \
		PYTHONPATH=. $(PYTHON) experiments/self_state/local_contraction.py --seed 42; \
	fi

exp-regulator-gain: ## Run Self-State Regulator Gain Sweep
	@echo "Running NSA Regulator Gain Sweep..."
	@if [ "$(UV_EXISTS)" = "yes" ]; then \
		$(UV) run python experiments/self_state/regulator_gain_sweep.py --seed 42; \
	else \
		PYTHONPATH=. $(PYTHON) experiments/self_state/regulator_gain_sweep.py --seed 42; \
	fi

exp-trained-reg: ## Run Trained Self-State Regulation experiment
	@echo "Running NSA Trained Regulation Experiment..."
	@if [ "$(UV_EXISTS)" = "yes" ]; then \
		$(UV) run python experiments/self_state/trained_regulation.py --seed 42; \
	else \
		PYTHONPATH=. $(PYTHON) experiments/self_state/trained_regulation.py --seed 42; \
	fi

exp-predictor-target: ## Run Predictor Target-Quality Evaluation
	@echo "Running NSA Predictor Target-Quality Evaluation..."
	@if [ "$(UV_EXISTS)" = "yes" ]; then \
		$(UV) run python experiments/self_state/predictor_target_quality.py --seed 42; \
	else \
		PYTHONPATH=. $(PYTHON) experiments/self_state/predictor_target_quality.py --seed 42; \
	fi

exp-all: exp-hard-state exp-self-state exp-local-contraction exp-regulator-gain exp-trained-reg exp-predictor-target ## Run all NSA experiment suites in sequence
	@echo "=========================================================="
	@echo "  ALL NSA EXPERIMENT SUITES COMPLETED SUCCESSFULLY"
	@echo "=========================================================="

lint: ## Check code for syntax errors and style issues
	@if [ "$(UV_EXISTS)" = "yes" ]; then \
		$(UV) run ruff check nsa/ demo/ eval/ tests/ 2>/dev/null || $(UV) run python -m compileall nsa demo eval tests; \
	elif command -v ruff >/dev/null 2>&1; then \
		ruff check nsa/ demo/ eval/ tests/; \
	else \
		PYTHONPATH=. $(PYTHON) -m compileall nsa demo eval tests; \
	fi

format: ## Auto-format code using black or ruff
	@if [ "$(UV_EXISTS)" = "yes" ]; then \
		$(UV) run ruff format nsa/ demo/ eval/ tests/ 2>/dev/null || $(UV) run black nsa/ demo/ eval/ tests/ 2>/dev/null || echo "Install ruff or black to format code."; \
	elif command -v black >/dev/null 2>&1; then \
		black nsa/ demo/ eval/ tests/; \
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
