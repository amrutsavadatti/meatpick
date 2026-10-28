.PHONY: dataset test lint
dataset:
	.venv/bin/python -m meatpick dataset --out outputs
test:
	.venv/bin/pytest -q
lint:
	.venv/bin/ruff check .
