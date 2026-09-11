## Summary

<!-- What does this PR change, and why? -->

## Type of change

- [ ] Bug fix
- [ ] New feature / experiment
- [ ] Documentation
- [ ] Refactor (no behavior change)

## Checklist

- [ ] I ran the relevant script(s) with `--smoke` locally and confirmed no errors
- [ ] I ran `flake8 . --select=E9,F63,F7,F82` locally and it passes
- [ ] If I changed anything in `evaluation/metrics.py`, `models/qtrace_detector.py`,
      or the attack/simulator physics, I re-ran at least one non-smoke experiment
      and confirmed the results still make physical sense (not just "no crash")
- [ ] I did not commit any generated dataset files, results from `--smoke` runs,
      or credentials (see `.gitignore`)
- [ ] I updated `README.md` / docstrings if this changes how someone would
      reproduce results

## How was this tested?

<!-- Commands run, output observed, etc. -->
