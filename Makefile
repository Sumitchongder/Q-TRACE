PY ?= python

.PHONY: install test lint smoke benchmark rehearsal figures reproduce clean

install:
	$(PY) -m pip install -r requirements.txt && $(PY) -m pip install -e .

test:
	$(PY) -m pytest -q

lint:
	ruff check src tests experiments

smoke:
	$(PY) experiments/run_benchmark.py --smoke --out results_smoke

benchmark:          ## 5 held-out seeds (~1.5 h on one CPU core)
	$(PY) experiments/run_benchmark.py --config experiments/configs/benchmark.yaml --out results/benchmark

rehearsal:          ## noisy dress rehearsal of the QPU experiment (no QPU time)
	$(PY) experiments/run_rehearsal.py --backend FakeMarrakesh --shots 2000

figures:
	$(PY) experiments/make_figures.py && $(PY) experiments/update_readme.py
figures-only:
	$(PY) experiments/make_figures.py

reproduce: benchmark rehearsal figures

clean:
	rm -rf results_smoke .pytest_cache
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
