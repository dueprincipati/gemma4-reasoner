.PHONY: install install-all test lint format clean docker-build docker-run help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install with minimal dependencies
	pip install -e .

install-all: ## Install with all backends
	pip install -e ".[all]"

install-dev: ## Install with dev dependencies
	pip install -e ".[all,dev]"

test: ## Run tests
	pytest tests/ -v

test-cov: ## Run tests with coverage
	pytest tests/ -v --cov=src/gemma4_reasoner --cov-report=html

lint: ## Lint with ruff
	ruff check src/ tests/

format: ## Format with ruff
	ruff format src/ tests/

typecheck: ## Type check with mypy
	mypy src/gemma4_reasoner/

clean: ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +

docker-build: ## Build Docker image
	docker compose build reasoner

docker-run: ## Run with Docker Compose (full stack)
	docker compose --profile full up

docker-standalone: ## Run with Docker Compose (connect to existing server)
	docker compose --profile standalone run --rm reasoner-standalone

pull-model: ## Pull Gemma 4 model via Ollama
	ollama pull gemma4:e4b
