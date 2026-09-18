FUNCTIONS ?= data/input/functions_definition.json
INPUT ?= data/input/function_calling_tests.json
OUTPUT ?= data/output/function_calls.json
export UV_PROJECT_ENVIRONMENT := $(HOME)/goinfre/.venv
export UV_CACHE_DIR := $(HOME)/goinfre/.uv_cache
export HF_HOME := $(HOME)/goinfre/.cache/huggingface
export TMPDIR := $(HOME)/goinfre/tmp

install:
	@mkdir -p $(TMPDIR) $(UV_CACHE_DIR) $(HF_HOME)
	uv sync

run:
	uv run python -m src --functions_definition $(FUNCTIONS) --input $(INPUT) --output $(OUTPUT)

debug:
	uv run python -m pdb -m src $(FUNCTIONS) $(INPUT) $(OUTPUT)

clean:
	rm -rf __pycache__ .mypy_cache .pytest_cache graph_pac/.mypy_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

lint:
	flake8 .
	mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

.PHONY: install run debug clean lint