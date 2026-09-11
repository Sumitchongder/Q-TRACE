# Contributing to Q-TRACE

Thanks for your interest in contributing. This is a research-code
repository accompanying a Q1-journal submission, so the bar for changes
that affect reported results is intentionally high; documentation,
tooling, and new experiment scripts are very welcome with a lower bar.

## Ground rules

1. **Never commit generated datasets, results, or credentials.** See
   `.gitignore`. If you're not sure whether a file should be tracked,
   ask in the PR rather than committing it.
2. **Every experiment script must support `--smoke`** for a fast
   correctness check, and CI (`.github/workflows/ci.yml`) runs these on
   every PR. If you add a new experiment script, add a smoke-test job
   for it.
3. **Changes that affect reported numbers need more than "it runs."**
   If you touch `evaluation/metrics.py`, `models/qtrace_detector.py`,
   `qkd_simulator/`, or `attacks/`, run at least one non-trivial-scale
   experiment (not just `--smoke`) and sanity-check that the resulting
   numbers are physically plausible before opening a PR -- see the
   README's "Known-issue changelog" section for two real examples of
   bugs that ran without crashing but produced methodologically invalid
   numbers.
4. **Cite your sources.** If you add a new attack model or change the
   physics in `qkd_simulator/`, reference the paper/mechanism it's
   modeling in the module docstring.

## Development setup

```bash
git clone https://github.com/<your-username>/Q-TRACE.git
cd Q-TRACE
conda env create -f environment.yml
conda activate qgss
pip install -r requirements.txt
pip install flake8  # for local linting, matches CI
```

## Before opening a PR

```bash
# lint (matches CI exactly)
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

# smoke-test whatever you changed, e.g.:
python -m dataset.generate --out dataset/dev_smoke --runs-per-class 3 --windows-per-run 20 --block-size 10 --seed 1
python -m evaluation.run_experiments --data dataset/dev_smoke/qkd_trace_telemetry.csv --block-size 10 --smoke --out results_dev
```

## Reporting bugs

Use the Bug Report issue template. Include the exact command, full
traceback, and your environment (`pip freeze | grep -E "qiskit|xgboost|scikit-learn"`).

## Questions

Open a Discussion or an issue with the Feature Request template.
