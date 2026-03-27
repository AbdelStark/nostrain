PYTHON ?= python3

.PHONY: install-dev lint test test-relay coverage package

install-dev:
	$(PYTHON) -m pip install -e ".[dev,numpy]"

lint:
	$(PYTHON) -m ruff check src tests demo

test:
	$(PYTHON) -m pytest -q

test-relay:
	$(PYTHON) -m pytest tests/test_relay.py -q

coverage:
	$(PYTHON) -m pytest --cov=nostrain --cov-report=term-missing -q

package:
	$(PYTHON) -m build
