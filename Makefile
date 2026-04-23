PYTHON ?= python
PIP ?= $(PYTHON) -m pip

.PHONY: setup setup-benchmark test format lint prepare-data det-benchmark det-benchmark-fair det-benchmark-fair-prepare anom-benchmark

setup:
	$(PIP) install -e .[dev]

setup-benchmark:
	$(PIP) install -e .[dev,benchmark]

test:
	$(PYTHON) -m pytest

format:
	$(PYTHON) -m ruff format .

lint:
	$(PYTHON) -m ruff check .

prepare-data:
	$(PYTHON) scripts/prepare_dataset.py
	$(PYTHON) scripts/make_splits.py

det-benchmark:
	$(PYTHON) scripts/train_detector.py
	$(PYTHON) scripts/infer_detector.py
	$(PYTHON) scripts/crop_rois.py --source both

det-benchmark-fair-prepare:
	$(PYTHON) scripts/run_detection_benchmark.py

det-benchmark-fair:
	$(PYTHON) scripts/run_detection_benchmark.py --execute

anom-benchmark:
	$(PYTHON) scripts/train_anomaly.py
	$(PYTHON) scripts/evaluate_pipeline.py
