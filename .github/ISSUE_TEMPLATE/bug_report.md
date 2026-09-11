---
name: Bug report
about: Report a problem reproducing results or running the code
title: "[BUG] "
labels: bug
assignees: ''
---

**Describe the bug**
A clear description of what went wrong.

**Command run**
```bash
# paste the exact command(s) you ran
```

**Full error output / traceback**
```
paste the full traceback here
```

**Environment**
- OS: [e.g. WSL2 Ubuntu 22.04, native Linux, macOS]
- Python version: `python --version`
- Package versions: `pip freeze | grep -E "qiskit|xgboost|scikit-learn|numpy|pandas"`
- Installed via: `requirements.txt` or `environment.yml`?

**Which script / stage**
- [ ] `dataset.generate`
- [ ] `evaluation.run_experiments`
- [ ] `evaluation.run_multiseed`
- [ ] `evaluation.run_sweep_experiment`
- [ ] `evaluation.run_adversarial_evasion`
- [ ] `quantum.ibm_runtime_job` (real hardware)
- [ ] `quantum.ibm_runtime_job` (fake backend / `--use-fake-backend`)
- [ ] Notebook (`notebooks/01_end_to_end_demo.ipynb`)
- [ ] Other (describe)

**Additional context**
Anything else relevant (dataset scale used, whether `--smoke` was passed, etc).
