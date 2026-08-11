.DEFAULT_GOAL := help
PY ?= python

.PHONY: help install install-dev lint format typecheck test cov check serve batch audit cache-warm cache-prune docker docker-run clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Install runtime dependencies (pinned)
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e . --no-deps

install-dev: ## Install everything needed to develop and test
	$(PY) -m pip install -r requirements-dev.txt
	$(PY) -m pip install -e . --no-deps

lint: ## Run ruff
	$(PY) -m ruff check .

format: ## Auto-format and fix what can be fixed
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

typecheck: ## Run mypy
	$(PY) -m mypy

test: ## Run the test suite
	$(PY) -m pytest

cov: ## Run tests with a coverage report
	$(PY) -m pytest --cov --cov-report=term-missing --cov-report=html

check: lint typecheck test ## Everything CI runs

serve: ## Run the API locally with reload
	ENVIRONMENT=development LOG_FORMAT=text DOCS_ENABLED=true \
		$(PY) -m pharmacy_ocr serve --reload

batch: ## Extract every invoice in data/invoices
	$(PY) -m pharmacy_ocr batch

audit: ## Summarize extraction outputs
	$(PY) -m pharmacy_ocr audit

cache-warm: ## Pre-compute OCR so parser iteration is instant
	$(PY) -m pharmacy_ocr cache warm --retry-profiles

cache-prune: ## Drop expired and surplus cache entries
	$(PY) -m pharmacy_ocr cache prune

docker: ## Build the production image
	docker build --target runtime -t pharmacy-ocr-service:3.0.0 .

docker-run: ## Run the production image against ./data
	docker run --rm -p 8000:8000 -v "$(PWD)/data:/data" \
		-e ENVIRONMENT=development -e LOG_FORMAT=text \
		pharmacy-ocr-service:3.0.0

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml build dist
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
