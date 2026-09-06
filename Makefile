.PHONY: install lint test bench calibrate spatial-calibrate ablation serve demo check

install:
	pip install -e ".[api,dev]"

lint:
	ruff check src tests

test:
	pytest -q

bench:
	python -m skyguard.cli benchmark

calibrate:
	python -m skyguard.cli calibrate

spatial-calibrate:
	python -m skyguard.cli spatial-calibrate

ablation:
	python -m skyguard.cli ablation

serve:
	python -m skyguard.cli serve

demo:
	python -m skyguard.cli demo

# Run before every commit. If clean-stream false-positive rate has risen,
# the change is not ready, whatever it did to recall.
check: lint test
	python -m skyguard.cli benchmark --quick
