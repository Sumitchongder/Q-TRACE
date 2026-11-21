"""Run the QKD-TRACE v2 benchmark.

    python experiments/run_benchmark.py --config experiments/configs/benchmark.yaml \
        --out results/benchmark

Each seed is written to ``<out>/seed_<s>/`` as soon as it finishes, so an
interrupted sweep resumes where it stopped (use ``--force`` to recompute).
``--smoke`` runs a tiny configuration for continuous integration.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import platform
import time
import warnings

import numpy as np
import yaml

from qtrace import __version__
from qtrace.detect.detector import DetectorConfig
from qtrace.evaluation.benchmark import BenchmarkConfig, run_seed

SMOKE_RUNS = dict(train_honest=6, train_attack=3, val_honest=6, val_loss_fault=2,
                  test_honest=5, test_loss_fault=2, test_device_fault=1, test_attack=2,
                  test_composite=1, shift_honest=2, shift_attack=1)


def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def save_examples(runs, path, n_per_family=3):
    keep = {}
    counter = {}
    for r in runs:
        sp = r["spec"]
        if sp["split"] != "test" or "scores" not in r:
            continue
        fam = sp["family"]
        if counter.get(fam, 0) >= n_per_family:
            continue
        i = counter.get(fam, 0)
        counter[fam] = i + 1
        pre = f"{fam}__{i}__"
        for m, s in r["scores"].items():
            keep[pre + "score__" + m] = s.astype(np.float32)
        keep[pre + "label"] = r["label"].astype(np.int8)
        keep[pre + "param_z"] = r["param_z"]
        keep[pre + "onset"] = np.array(sp["onset"])
        keep[pre + "strength"] = np.array(r["strength"])
    np.savez_compressed(path, **keep)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="experiments/configs/benchmark.yaml")
    ap.add_argument("--out", default="results/benchmark")
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    conf = yaml.safe_load(open(args.config))
    runs = SMOKE_RUNS if args.smoke else conf["runs"]
    cfg = BenchmarkConfig(n_windows=conf["n_windows"], n_cal=conf["n_cal"],
                          pulses_per_window=conf["pulses_per_window"],
                          onset_range=tuple(conf["onset_range"]),
                          afterpulse_spec_error=conf["afterpulse_spec_error"], runs=runs)
    det = conf["detector"]
    det_cfg = DetectorConfig(horizons=tuple(det["horizons"]),
                             local_horizons=tuple(det["local_horizons"]),
                             local_reference=det["local_reference"])
    seeds = args.seeds or (conf["seeds"][:1] if args.smoke else conf["seeds"])
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        d = out / f"seed_{seed}"
        if (d / "meta.json").exists() and not args.force:
            print(f"[seed {seed}] exists, skipping")
            continue
        d.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        wdf, rdf, meta, all_runs = run_seed(seed, cfg, det_cfg,
                                            fars=tuple(conf["false_alarm_rates"]))
        wdf.to_csv(d / "window_metrics.csv", index=False)
        rdf.to_csv(d / "run_metrics.csv", index=False)
        if seed == seeds[0]:
            save_examples(all_runs, d / "examples.npz")
        meta.update(dict(qtrace_version=__version__, python=platform.python_version(),
                         config=conf, wall_time_s=time.time() - t0, smoke=args.smoke))
        json.dump(to_jsonable(meta), open(d / "meta.json", "w"), indent=2)
        print(f"[seed {seed}] done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
