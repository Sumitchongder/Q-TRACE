# Reproducibility

| Item | Command | Output |
|---|---|---|
| Unit tests | `pytest -q` | 18 tests |
| Benchmark (5 seeds) | `make benchmark` | `results/benchmark/seed_*/` |
| Hardware rehearsal | `make rehearsal` | `results/hardware/rehearsal_fake_marrakesh.npz` |
| Tables, figures, README block | `make figures` | `results/summary/`, `figures/` |

* Seeds: development 0–3 (not reported); evaluation 101–105. Hardware schedule seed 2026.
* Every random draw in the benchmark derives from the seed via `numpy.random.default_rng(seed)`; with the
  pinned package versions a seed reproduces its CSVs exactly.
* Software: Python 3.12, versions in `requirements.txt`. The benchmark takes about 17 min per seed on one
  CPU core.
* Each `meta.json` records configuration, thresholds, honest-null parameters, versions and wall time.
