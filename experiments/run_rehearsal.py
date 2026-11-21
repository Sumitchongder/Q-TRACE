"""Noisy dress rehearsal of the hardware experiment on a fake IBM backend.

    python experiments/run_rehearsal.py --backend FakeMarrakesh --shots 2000

Produces ``results/hardware/rehearsal_<backend>.npz`` in exactly the format
written by the QPU notebook, so the analysis pipeline can be validated
before any QPU time is spent. These data are simulated and are never
reported as hardware results.
"""
import argparse
import time
import warnings

from qiskit_ibm_runtime import fake_provider

from qtrace.hardware.runner import HardwareExperiment

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--backend", default="FakeMarrakesh")
ap.add_argument("--shots", type=int, default=2000)
ap.add_argument("--links", type=int, default=40)
ap.add_argument("--out", default=None)
a = ap.parse_args()
be = getattr(fake_provider, a.backend)()
ex = HardwareExperiment(be, n_links=a.links)
t0 = time.time()
bob, eve = ex.rehearse(shots=a.shots)
out = a.out or f"results/hardware/rehearsal_{be.name}.npz"
ex.save(out, bob, eve, "rehearsal")
print(f"saved {out} in {time.time() - t0:.0f}s")
