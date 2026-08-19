# ==============================================================================
# Makefile for Neural State Architecture (NSA) - Modernized Research Platform
# ==============================================================================

.PHONY: help install install-dev venv test evidence demo demo-ollama demo-transformers \
        benchmark benchmark-nsa61 benchmark-nsa60 benchmark-ablation benchmark-gpse \
        benchmark-gtc benchmark-security redteam report clean legacy-showcase

UV := $(shell command -v uv 2>/dev/null || (test -f ~/.local/bin/uv && echo ~/.local/bin/uv) || (test -f ~/.cargo/bin/uv && echo ~/.cargo/bin/uv) || echo "uv")
UV_EXISTS := $(shell command -v $(UV) >/dev/null 2>&1 && echo yes || echo no)

PYTHON ?= python3
VENV_DIR ?= .venv

.DEFAULT_GOAL := help

help: ## Display available targets
	@echo "Neural State Architecture (NSA) — Master Command Suite"
	@echo "======================================================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

venv: ## Create virtual environment
	@if [ "$(UV_EXISTS)" = "yes" ]; then $(UV) venv $(VENV_DIR); else $(PYTHON) -m venv $(VENV_DIR); fi

install: ## Install runtime requirements
	@if [ "$(UV_EXISTS)" = "yes" ]; then $(UV) pip install -r requirements.txt; else $(PYTHON) -m pip install -r requirements.txt; fi

install-dev: install ## Install runtime and development dependencies
	@if [ "$(UV_EXISTS)" = "yes" ]; then $(UV) pip install pytest black ruff mypy; else $(PYTHON) -m pip install pytest black ruff mypy; fi

test: ## Run unit and integration test suite (239+ tests)
	@if [ "$(UV_EXISTS)" = "yes" ]; then \
		$(UV) run python -m pytest -v tests/; \
	else \
		PYTHONPATH=. $(PYTHON) -m pytest -v tests/; \
	fi

evidence: ## Validate and verify the machine-traceable formal evidence manifest
	@if [ "$(UV_EXISTS)" = "yes" ]; then \
		$(UV) run python evidence/validate_evidence.py; \
	else \
		PYTHONPATH=. $(PYTHON) evidence/validate_evidence.py; \
	fi

sync-metadata: ## Automatically synchronize test and claim counts across docs and Makefile
	@if [ "$(UV_EXISTS)" = "yes" ]; then \
		$(UV) run python scripts/sync_metadata.py; \
	else \
		PYTHONPATH=. $(PYTHON) scripts/sync_metadata.py; \
	fi

demo: ## Launch closed-loop cognitive runtime demonstration (fast mock)
	@if [ "$(UV_EXISTS)" = "yes" ]; then \
		PYTHONPATH=. $(UV) run python experiments/nsa62/live_cognitive_demo.py --backend mock; \
	else \
		PYTHONPATH=. $(PYTHON) experiments/nsa62/live_cognitive_demo.py --backend mock; \
	fi

demo-live-0.5b: ## Launch fast smoke demonstration with local cached Qwen2.5-0.5B-Instruct
	@if [ "$(UV_EXISTS)" = "yes" ]; then \
		PYTHONPATH=. $(UV) run python experiments/nsa62/live_cognitive_demo.py --backend cached --model Qwen/Qwen2.5-0.5B-Instruct; \
	else \
		PYTHONPATH=. $(PYTHON) experiments/nsa62/live_cognitive_demo.py --backend cached --model Qwen/Qwen2.5-0.5B-Instruct; \
	fi

demo-live-3b: ## Launch canonical live closed-loop demo with cached Qwen2.5-3B-Instruct
	@if [ "$(UV_EXISTS)" = "yes" ]; then \
		PYTHONPATH=. $(UV) run python experiments/nsa62/live_cognitive_demo.py --backend cached --model Qwen/Qwen2.5-3B-Instruct; \
	else \
		PYTHONPATH=. $(PYTHON) experiments/nsa62/live_cognitive_demo.py --backend cached --model Qwen/Qwen2.5-3B-Instruct; \
	fi

demo-live: demo-live-3b ## Alias for canonical 3B live demonstration

demo-live-ollama: ## Launch live demonstration connected to local Ollama (qwen2.5:3b)
	@if [ "$(UV_EXISTS)" = "yes" ]; then \
		PYTHONPATH=. $(UV) run python experiments/nsa62/live_cognitive_demo.py --backend ollama --model qwen2.5:3b; \
	else \
		PYTHONPATH=. $(PYTHON) experiments/nsa62/live_cognitive_demo.py --backend ollama --model qwen2.5:3b; \
	fi

benchmark: benchmark-nsa62 benchmark-ablation benchmark-gpse benchmark-gtc benchmark-security ## Run complete benchmark suite

benchmark-nsa63: ## Run NSA 6.3 procedural randomized validation & 6-arm ablation suite
	@if [ "$(UV_EXISTS)" = "yes" ]; then \
		PYTHONPATH=. $(UV) run python experiments/nsa63/scientific_validation_suite.py --backend mock --trials 40; \
	else \
		PYTHONPATH=. $(PYTHON) experiments/nsa63/scientific_validation_suite.py --backend mock --trials 40; \
	fi

benchmark-nsa63-ablation: benchmark-nsa63 ## Alias for NSA 6.3 ablation benchmark

benchmark-nsa62: ## Run NSA 6.2 closed-loop cognitive benchmark (fast mock)
	@if [ "$(UV_EXISTS)" = "yes" ]; then \
		PYTHONPATH=. $(UV) run python experiments/nsa62/qwen25_3b_cognitive_benchmark.py --backend mock; \
	else \
		PYTHONPATH=. $(PYTHON) experiments/nsa62/qwen25_3b_cognitive_benchmark.py --backend mock; \
	fi

benchmark-smoke: ## Run 4-trial fast smoke benchmark on cached Qwen2.5-0.5B
	@if [ "$(UV_EXISTS)" = "yes" ]; then \
		PYTHONPATH=. $(UV) run python experiments/nsa62/qwen25_3b_cognitive_benchmark.py --backend cached --model Qwen/Qwen2.5-0.5B-Instruct --trials 4 --output-dir results/nsa62/qwen2.5-0.5b; \
	else \
		PYTHONPATH=. $(PYTHON) experiments/nsa62/qwen25_3b_cognitive_benchmark.py --backend cached --model Qwen/Qwen2.5-0.5B-Instruct --trials 4 --output-dir results/nsa62/qwen2.5-0.5b; \
	fi

benchmark-canonical-3b: ## Run canonical live benchmark on cached Qwen2.5-3B-Instruct
	@if [ "$(UV_EXISTS)" = "yes" ]; then \
		PYTHONPATH=. $(UV) run python experiments/nsa62/qwen25_3b_cognitive_benchmark.py --backend cached --model Qwen/Qwen2.5-3B-Instruct --trials 20 --output-dir results/nsa62/qwen2.5-3b; \
	else \
		PYTHONPATH=. $(PYTHON) experiments/nsa62/qwen25_3b_cognitive_benchmark.py --backend cached --model Qwen/Qwen2.5-3B-Instruct --trials 20 --output-dir results/nsa62/qwen2.5-3b; \
	fi

benchmark-live: benchmark-smoke ## Default live benchmark runner

benchmark-nsa60: ## Run NSA 6.0 real-model cognitive transfer benchmark
	@if [ "$(UV_EXISTS)" = "yes" ]; then \
		PYTHONPATH=. $(UV) run python experiments/nsa60/real_model_transfer_suite.py; \
	else \
		PYTHONPATH=. $(PYTHON) experiments/nsa60/real_model_transfer_suite.py; \
	fi

benchmark-ablation: ## Run NSA 5.1 6-arm controlled cognitive ablation suite
	@if [ "$(UV_EXISTS)" = "yes" ]; then \
		PYTHONPATH=. $(UV) run python experiments/nsa51/ablation_suite.py; \
	else \
		PYTHONPATH=. $(PYTHON) experiments/nsa51/ablation_suite.py; \
	fi

benchmark-gpse: ## Run NSA 5.0 Governed Problem Solving Efficiency (GPSE) benchmark
	@if [ "$(UV_EXISTS)" = "yes" ]; then \
		PYTHONPATH=. $(UV) run python experiments/nsa50/gpse_benchmark.py; \
	else \
		PYTHONPATH=. $(PYTHON) experiments/nsa50/gpse_benchmark.py; \
	fi

benchmark-gtc: ## Run NSA 4.2 Governed Task Completion (GTC) benchmark
	@if [ "$(UV_EXISTS)" = "yes" ]; then \
		PYTHONPATH=. $(UV) run python experiments/nsa41/gtc_benchmark.py; \
	else \
		PYTHONPATH=. $(PYTHON) experiments/nsa41/gtc_benchmark.py; \
	fi

benchmark-security: ## Run NSA 4.0 strategic deceptive adversary benchmark
	@if [ "$(UV_EXISTS)" = "yes" ]; then \
		PYTHONPATH=. $(UV) run python experiments/security/strategic_deceptive_adversary.py; \
	else \
		PYTHONPATH=. $(PYTHON) experiments/security/strategic_deceptive_adversary.py; \
	fi

redteam: ## Run full red-team adversarial suite across threat vectors
	@if [ "$(UV_EXISTS)" = "yes" ]; then \
		PYTHONPATH=. $(UV) run python prototype/security/adversarial_suite.py; \
	else \
		PYTHONPATH=. $(PYTHON) prototype/security/adversarial_suite.py; \
	fi

report: ## Generate comprehensive verification report
	@echo "Generating NSA Empirical Verification Report..."
	@$(MAKE) test
	@$(MAKE) evidence
	@$(MAKE) benchmark-nsa61

legacy-showcase: ## Launch legacy Gradio / LoRA retrofit showcase
	@if [ "$(UV_EXISTS)" = "yes" ]; then \
		$(UV) run python demo/web_demo.py; \
	else \
		PYTHONPATH=. $(PYTHON) demo/web_demo.py; \
	fi

clean: ## Clean cached build artifacts and temporary files
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
