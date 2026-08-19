# ==============================================================================
# Neural State Architecture (NSA) — canonical research command interface
# ==============================================================================

.PHONY: help venv install install-dev test evidence sync-metadata \
        demo demo-live demo-live-0.5b demo-live-3b demo-live-ollama demo-lmstudio \
        benchmark benchmark-nsa63 benchmark-nsa63-3b benchmark-nsa63-ablation \
        benchmark-nsa62 benchmark-smoke benchmark-canonical-3b benchmark-live \
        benchmark-lmstudio benchmark-ollama benchmark-nsa60 benchmark-ablation \
        benchmark-gpse benchmark-gtc benchmark-security redteam report \
        legacy-showcase clean

UV := $(shell command -v uv 2>/dev/null || (test -f ~/.local/bin/uv && echo ~/.local/bin/uv) || (test -f ~/.cargo/bin/uv && echo ~/.cargo/bin/uv) || echo uv)
UV_EXISTS := $(shell command -v $(UV) >/dev/null 2>&1 && echo yes || echo no)
PYTHON ?= python3
VENV_DIR ?= .venv

.DEFAULT_GOAL := help

# Use uv when available, otherwise the active/system Python environment.
PYRUN = $(if $(filter yes,$(UV_EXISTS)),$(UV) run python,PYTHONPATH=. $(PYTHON))

help: ## Display the canonical NSA command suite
	@echo "Neural State Architecture (NSA)"
	@echo "=============================="
	@grep -E '^[a-zA-Z0-9_.-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-26s\033[0m %s\n", $$1, $$2}'

venv: ## Create the local virtual environment
	@if [ "$(UV_EXISTS)" = "yes" ]; then $(UV) venv $(VENV_DIR); else $(PYTHON) -m venv $(VENV_DIR); fi

install: ## Install runtime dependencies
	@if [ "$(UV_EXISTS)" = "yes" ]; then $(UV) pip install -r requirements.txt; else $(PYTHON) -m pip install -r requirements.txt; fi

install-dev: install ## Install runtime plus development dependencies
	@if [ "$(UV_EXISTS)" = "yes" ]; then $(UV) pip install pytest black ruff mypy; else $(PYTHON) -m pip install pytest black ruff mypy; fi

test: ## Run the full unit/integration/scientific test suite (current baseline: 239 tests)
	$(PYRUN) -m pytest -v tests/

evidence: ## Verify the machine-readable evidence manifest (current baseline: 31 claims)
	$(PYRUN) evidence/validate_evidence.py

sync-metadata: ## Synchronize test/claim counts in repository metadata
	$(PYRUN) scripts/sync_metadata.py

demo: ## Run the deterministic closed-loop NSA demonstration (no model download)
	$(PYRUN) experiments/nsa62/live_cognitive_demo.py --backend mock

demo-live-0.5b: ## Run a real cached Qwen2.5-0.5B neural smoke demo
	$(PYRUN) experiments/nsa62/live_cognitive_demo.py --backend cached --model Qwen/Qwen2.5-0.5B-Instruct

demo-live-3b: ## Run the canonical real cached Qwen2.5-3B neural demo
	$(PYRUN) experiments/nsa62/live_cognitive_demo.py --backend cached --model Qwen/Qwen2.5-3B-Instruct

demo-live: demo-live-3b ## Alias for the canonical 3B neural demo

demo-live-ollama: ## Run the real neural demo through local Ollama
	$(PYRUN) experiments/nsa62/live_cognitive_demo.py --backend ollama --model qwen2.5:3b

demo-lmstudio: ## Run the real neural demo through LM Studio on port 1234
	$(PYRUN) experiments/nsa62/live_cognitive_demo.py --backend lmstudio --model default

benchmark-nsa63: ## Run the NSA 6.3 procedural blind-world six-arm validation (40 mock trials)
	$(PYRUN) experiments/nsa63/scientific_validation_suite.py --backend mock --trials 40 --hypotheses 4 --noise 0.0 --seed 42 --output-dir results/nsa63/mock

benchmark-nsa63-3b: ## Run NSA 6.3 against cached Qwen2.5-3B-Instruct (40 trials)
	$(PYRUN) experiments/nsa63/scientific_validation_suite.py --backend cached --model Qwen/Qwen2.5-3B-Instruct --trials 40 --hypotheses 4 --noise 0.0 --seed 42 --output-dir results/nsa63/qwen2.5-3b

benchmark-nsa63-ablation: benchmark-nsa63 ## Alias for the NSA 6.3 six-arm ablation suite

benchmark-lmstudio: ## Run NSA 6.3 through a local LM Studio server
	$(PYRUN) experiments/nsa63/scientific_validation_suite.py --backend lmstudio --trials 20 --output-dir results/nsa63/lmstudio

benchmark-ollama: ## Run NSA 6.3 through local Ollama
	$(PYRUN) experiments/nsa63/scientific_validation_suite.py --backend ollama --model qwen2.5:3b --trials 20 --output-dir results/nsa63/ollama

benchmark-nsa62: ## Run NSA 6.2 closed-loop benchmark in deterministic mock mode
	$(PYRUN) experiments/nsa62/qwen25_3b_cognitive_benchmark.py --backend mock

benchmark-smoke: ## Run a four-trial cached Qwen2.5-0.5B neural smoke benchmark
	$(PYRUN) experiments/nsa62/qwen25_3b_cognitive_benchmark.py --backend cached --model Qwen/Qwen2.5-0.5B-Instruct --trials 4 --output-dir results/nsa62/qwen2.5-0.5b

benchmark-canonical-3b: ## Run the NSA 6.2 canonical 20-trial Qwen2.5-3B benchmark
	$(PYRUN) experiments/nsa62/qwen25_3b_cognitive_benchmark.py --backend cached --model Qwen/Qwen2.5-3B-Instruct --trials 20 --output-dir results/nsa62/qwen2.5-3b

benchmark-live: benchmark-canonical-3b ## Alias for the canonical cached 3B benchmark

benchmark-nsa60: ## Run the NSA 6.0 real-model transfer benchmark
	$(PYRUN) experiments/nsa60/real_model_transfer_suite.py

benchmark-ablation: ## Run the NSA 5.1 controlled cognitive ablation suite
	$(PYRUN) experiments/nsa51/ablation_suite.py

benchmark-gpse: ## Run the NSA 5.0 GPSE benchmark
	$(PYRUN) experiments/nsa50/gpse_benchmark.py

benchmark-gtc: ## Run the NSA 4.2 governed task-completion benchmark
	$(PYRUN) experiments/nsa41/gtc_benchmark.py

benchmark-security: ## Run the NSA 4.0 strategic adversary benchmark
	$(PYRUN) experiments/security/strategic_deceptive_adversary.py

benchmark: benchmark-nsa63 benchmark-nsa62 benchmark-ablation benchmark-gpse benchmark-gtc benchmark-security ## Run the core benchmark progression

redteam: ## Run the adversarial security suite
	$(PYRUN) prototype/security/adversarial_suite.py

report: ## Run tests, evidence validation, and the flagship NSA 6.3 mock benchmark
	@$(MAKE) test
	@$(MAKE) evidence
	@$(MAKE) benchmark-nsa63

legacy-showcase: ## Identify the old showcase as historical; use the canonical runtime instead
	@echo "The old Gradio/LoRA showcase is retained for historical compatibility."
	@echo "Use 'make demo' or 'make demo-live-3b' for the current NSA runtime."

clean: ## Remove Python bytecode/cache directories
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
