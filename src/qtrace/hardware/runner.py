"""Execution of the hardware experiment on IBM Quantum (or a noisy rehearsal).

Budget control
--------------
IBM charges *quantum seconds*. Before the main job a small pilot job (one
honest pass, few shots) is executed and its reported usage gives a
conservative cost per shot (fixed per-job overhead is attributed to the
shots, so the estimate errs high). The main job's shots per circuit are
then chosen so that ``pilot + main`` stays below ``budget_s``.

Raw data
--------
Everything needed to reproduce the analysis is written to one ``.npz``:
per-shot Bob and Eve bits ``(circuits, shots, links)``, the design, the
selected physical qubits with their calibration data at run time, job
identifiers and reported usage.
"""

from __future__ import annotations

import datetime as _dt
import json
import time

import numpy as np

from qtrace.hardware.circuits import build_circuit, select_links, transpile_design
from qtrace.hardware.design import ExperimentDesign, build_design


def bitstrings_to_array(strings, n_bits):
    """Qiskit bitstrings (clbit 0 rightmost) -> uint8 array (shots, n_bits), column = clbit."""
    b = np.frombuffer("".join(strings).encode(), dtype=np.uint8).reshape(len(strings), n_bits)
    return (b[:, ::-1] - ord("0")).astype(np.uint8)


def job_quantum_seconds(job):
    for getter in (lambda j: j.usage(), lambda j: j.metrics()["usage"]["quantum_seconds"],
                   lambda j: j.metrics()["usage"]["seconds"]):
        try:
            v = getter(job)
            if v is not None:
                return float(v)
        except Exception:      # noqa: BLE001 - API differs across versions/plans
            continue
    return float("nan")


def _sampler(backend, shots):
    from qiskit_ibm_runtime import SamplerV2
    sampler = SamplerV2(mode=backend)
    sampler.options.default_shots = int(shots)
    try:
        sampler.options.dynamical_decoupling.enable = False
        sampler.options.twirling.enable_gates = False
        sampler.options.twirling.enable_measure = False
    except AttributeError:
        pass
    return sampler


def extract(result, n_links):
    bob = np.stack([bitstrings_to_array(r.data.bob.get_bitstrings(), n_links) for r in result])
    eve = np.stack([bitstrings_to_array(r.data.eve.get_bitstrings(), n_links) for r in result])
    return bob, eve


def calibration_snapshot(backend, pairs):
    t = backend.target
    snap = []
    props = getattr(backend, "properties", None)
    props = props() if callable(props) else None
    for d, e in pairs:
        row = {"data": d, "ancilla": e}
        for role, q in (("data", d), ("ancilla", e)):
            try:
                row[f"readout_error_{role}"] = float(t["measure"][(q,)].error)
            except Exception:  # noqa: BLE001
                pass
            if props is not None:
                try:
                    row[f"T1_us_{role}"] = props.t1(q) * 1e6
                    row[f"T2_us_{role}"] = props.t2(q) * 1e6
                except Exception:  # noqa: BLE001
                    pass
        snap.append(row)
    return snap


class HardwareExperiment:
    def __init__(self, backend, n_links=40, n_cal_passes=2, n_epochs=6, seed=2026,
                 isolate=True):
        self.backend = backend
        pairs, table = select_links(backend, n_links=n_links, isolate=isolate)
        self.design = build_design(n_links=len(pairs), n_cal_passes=n_cal_passes,
                                   n_epochs=n_epochs, seed=seed)
        self.design.physical_pairs = pairs
        self.link_table = table
        self.log = []

    # ------------------------------------------------------------ QPU path
    def transpiled(self):
        if not hasattr(self, "_isa"):
            self._isa = transpile_design(self.design, self.backend)
        return self._isa

    def pilot(self, shots=400):
        """One honest pass; returns quantum seconds per shot (conservative)."""
        isa = self.transpiled()[:8]
        job = _sampler(self.backend, shots).run(isa)
        job.result()
        qs = job_quantum_seconds(job)
        per_shot = qs / (8 * shots) if np.isfinite(qs) else float("nan")
        self.log.append(dict(stage="pilot", job_id=job.job_id(), shots=shots, circuits=8,
                             quantum_seconds=qs))
        return per_shot, qs

    def plan_shots(self, per_shot_s, budget_s, already_used_s, min_shots=1000,
                   max_shots=8000):
        avail = budget_s - already_used_s
        n_circ = self.design.n_circuits
        if not np.isfinite(per_shot_s) or per_shot_s <= 0:
            return min_shots
        shots = int(np.floor(avail / (n_circ * per_shot_s) / 100) * 100)
        return int(np.clip(shots, min_shots, max_shots))

    def run(self, shots):
        isa = self.transpiled()
        t0 = time.time()
        job = _sampler(self.backend, shots).run(isa)
        print(f"submitted job {job.job_id()} ({len(isa)} circuits x {shots} shots); waiting...")
        result = job.result()
        bob, eve = extract(result, self.design.n_links)
        qs = job_quantum_seconds(job)
        self.log.append(dict(stage="main", job_id=job.job_id(), shots=shots,
                             circuits=len(isa), quantum_seconds=qs,
                             wall_seconds=time.time() - t0))
        return bob, eve

    # ----------------------------------------------------- noisy rehearsal
    def rehearse(self, shots=2000, progress=True):
        """Noisy simulation on a fake backend, one link at a time.

        Links share no gates, so simulating them separately with the
        backend noise model is exact for that model (it contains no
        crosstalk); Aer truncates idle qubits, so every simulation involves
        two qubits. The circuits are the same ISA circuits as on hardware.
        """
        from qiskit_aer import AerSimulator
        d = self.design
        nl, nc = d.n_links, d.n_circuits
        sim = AerSimulator.from_backend(self.backend)
        bob = np.zeros((nc, shots, nl), dtype=np.uint8)
        eve = np.zeros_like(bob)
        for i in range(nl):
            circs = [build_circuit(d, c, links=[i]) for c in range(nc)]
            isa = transpile_design(d, self.backend, circuits=circs)
            res = sim.run(isa, shots=shots, memory=True, seed_simulator=d.seed + i).result()
            for c in range(nc):
                mem = res.get_memory(c)                    # "eve bob", clbit 0 rightmost
                e_str = [m.split()[0] for m in mem]
                b_str = [m.split()[1] for m in mem]
                bob[c, :, i] = bitstrings_to_array(b_str, nl)[:, i]
                eve[c, :, i] = bitstrings_to_array(e_str, nl)[:, i]
            if progress and (i + 1) % 10 == 0:
                print(f"  rehearsal: {i + 1}/{nl} links simulated", flush=True)
        self.log.append(dict(stage="rehearsal", backend=self.backend.name, shots=shots))
        return bob, eve

    # ------------------------------------------------------------------ I/O
    def save(self, path, bob, eve, mode):
        meta = dict(mode=mode, backend=self.backend.name,
                    timestamp_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(),
                    link_table=self.link_table,
                    calibration=calibration_snapshot(self.backend, self.design.physical_pairs),
                    log=self.log)
        np.savez_compressed(path, bob=bob, eve=eve, design=self.design.to_json(),
                            meta=json.dumps(meta, default=str))
        return path


def load_raw(path):
    z = np.load(path, allow_pickle=False)
    design = ExperimentDesign.from_json(str(z["design"]))
    meta = json.loads(str(z["meta"]))
    return z["bob"], z["eve"], design, meta
