.PHONY: install lint smoke-test data experiments multiseed sweep evasion clean help

help:
	@echo "Q-TRACE — common commands"
	@echo ""
	@echo "  make install       Install dependencies into the active environment"
	@echo "  make lint          Run flake8 (matches CI exactly)"
	@echo "  make smoke-test    Fast correctness check of the full pipeline (~1 min)"
	@echo "  make data          Generate the main dataset (dataset/qkd_trace_v1)"
	@echo "  make experiments   Run the full experiment suite on the main dataset"
	@echo "  make multiseed     Run the 5-seed statistical-significance sweep (slow)"
	@echo "  make sweep         Run the channel-loss x detector-efficiency heatmap"
	@echo "  make evasion       Run adversarial evasion for all six attack families"
	@echo "  make clean         Remove generated datasets/results (NOT source code)"

install:
	pip install -r requirements.txt

lint:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

smoke-test:
	python -m dataset.generate --out dataset/smoke_test --runs-per-class 3 --windows-per-run 20 --block-size 10 --seed 1
	python -m evaluation.run_experiments --data dataset/smoke_test/qkd_trace_telemetry.csv --block-size 10 --smoke --out results_smoke_test
	python -m evaluation.run_sweep_experiment --smoke --out results_smoke_test
	python -m evaluation.run_adversarial_evasion --attack pns --smoke --out results_smoke_test
	@echo "Smoke test passed."

data:
	python -m dataset.generate --out dataset/qkd_trace_v1 --runs-per-class 40 --windows-per-run 120 --block-size 30 --seed 42

experiments:
	python -m evaluation.run_experiments --data dataset/qkd_trace_v1/qkd_trace_telemetry.csv --block-size 30 --n-qubits 6 --out results

multiseed:
	python -m evaluation.run_multiseed --seeds 42 43 44 45 46 --runs-per-class 40 --windows-per-run 120 --block-size 30 --n-qubits 6 --out results_multiseed

sweep:
	python -m evaluation.run_sweep_experiment --n-loss 8 --n-eta 8 --runs-per-point 8 --windows-per-run 80 --out results

evasion:
	for atk in pns intercept_resend trojan_horse blinding time_shift rng_manipulation; do \
		python -m evaluation.run_adversarial_evasion --attack $$atk --n-iterations 60 --population 16 --out results; \
	done

clean:
	rm -rf dataset/smoke_test dataset/ci_smoke dataset/dev_smoke results_smoke_test results_ci results_dev
	find . -name "__pycache__" -exec rm -rf {} +
	find . -name "*.pyc" -delete
