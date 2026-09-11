TARGET = $(HOME)/goinfre/
VENV = $(TARGET)CallMe
UV_CACHE_DIR := $(TARGET)/uv-cache
HF_HOME := $(TARGET)/uv-hugging-cache
UV_LINK_MODE := copy

export UV_CACHE_DIR
export HF_HOME
export UV_LINK_MODE


install:
	UV_PROJECT_ENVIRONMENT=$(VENV) uv sync


clean:
	rm -rf $(TARGET)/uv-cache


fclean: clean
	rm -rf .venv
