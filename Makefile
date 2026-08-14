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
