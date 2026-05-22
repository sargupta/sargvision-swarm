.PHONY: install install-dev install-llm install-sim demo sim test lint format check clean snapshot

PY ?= uv run python

install:
	uv sync

install-dev:
	uv sync --extra dev

install-llm:
	uv sync --extra llm

install-sim:
	uv sync --extra sim

install-all:
	uv sync --all-extras

demo:
	$(PY) -m sargvision_swarm.demo.app

sim:
	uv run swarm sim --n 30 --scenario flock --steps 200

snapshot:
	uv run swarm sim --n 30 --scenario flock --steps 200 --snapshot snapshot.png

test:
	uv run pytest -q

lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests

check: lint test

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache __pycache__ */__pycache__ */*/__pycache__ dist build *.egg-info
