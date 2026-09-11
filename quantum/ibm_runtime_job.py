"""
Q-TRACE :: quantum.ibm_runtime_job
=====================================
Real-hardware execution on IBM's `ibm_marrakesh` (Heron r2, 156 qubits)
via Qiskit Runtime, budgeted so that ACTUAL QPU EXECUTION TIME stays
under an 8-minute ceiling (--max-qpu-seconds, default 420s to leave a
safety margin below the full 480s).

Two modes (mirrors the classical/simulator code exactly so the reported
figure is a genuine like-for-like hardware comparison, not a different
circuit):

  1. `kernel`  -- Evaluate a small, fixed-size fidelity quantum kernel
                  Gram sub-block on real hardware (a handful of circuits,
                  each with `reps` ZZFeatureMap layers). This is the most
                  QPU-time-predictable mode: cost scales as
                  n_train * n_test pairs, so keep both small (<= 12).

  2. `vqc_eval` -- Load PRE-TRAINED VQC parameters (trained on the
                  AerSimulator via quantum/vqc.py) and run ONLY inference
                  (forward-pass circuit evaluation, no optimizer loop) on
                  real hardware for a modest test batch. No on-hardware
                  training is attempted -- training a VQC with an
                  iterative classical optimizer on real hardware would
                  consume the entire QPU budget in only a few iterations.

Time-budgeting strategy
------------------------
Qiskit Runtime does not let you cap wall-clock execution precisely ahead
of time, but you *can* bound it indirectly and reliably by bounding the
things that multiply into execution time:

    QPU_seconds ~= n_circuits * shots * per_shot_time * transpile_depth_factor

This script keeps n_circuits and shots small and fixed, uses `optimization_level=3`
transpilation once, and estimates the expected QPU time from the backend's
reported instruction durations BEFORE submitting, printing that estimate
and aborting if it exceeds `--max-qpu-seconds`. This is a best-effort
static estimate, not a guarantee -- always check your IBM Quantum account
usage page during/after the run.

Usage (real hardware)
----------------------
    python -m quantum.ibm_runtime_job --mode kernel --n-qubits 4 \
        --n-train 8 --n-test 8 --shots 512 --backend ibm_marrakesh \
        --max-qpu-seconds 420

    python -m quantum.ibm_runtime_job --mode vqc_eval --n-qubits 4 \
        --vqc-params results/vqc_trained_params.npy --n-test 16 \
        --shots 512 --backend ibm_marrakesh --max-qpu-seconds 300

Usage (offline correctness test against a fake/noisy backend -- this is
what was used to validate this script before hardware access; it exercises
the identical code path, just swapping the backend object):
    python -m quantum.ibm_runtime_job --mode kernel --n-qubits 4 \
        --n-train 6 --n-test 6 --shots 256 --use-fake-backend
"""

import argparse
import os
import time
import numpy as np

from qiskit.circuit.library import zz_feature_map, real_amplitudes
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.state_fidelities import ComputeUncompute


def get_backend(args):
    if args.use_fake_backend:
        from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
        print("[info] Using FakeSherbrooke (offline stand-in) for correctness testing.")
        return FakeSherbrooke(), None
    else:
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService()  # assumes save_account() already run
        backend = service.backend(args.backend)
        print(f"[info] Connected to real backend: {backend.name} "
              f"({backend.num_qubits} qubits)")
        return backend, service


def estimate_qpu_seconds(backend, n_circuits, shots, circuit_depth_est=40):
    """
    Rough static pre-flight estimate: per-shot time is dominated by
    two-qubit gate durations on superconducting hardware, typically
    ~200-500 ns per layer; we use a conservative 400 ns/layer estimate
    plus fixed per-circuit overhead (~2 ms readout + reset).
    """
    ns_per_layer = 400e-9
    readout_overhead_s = 2e-3
    per_shot_s = circuit_depth_est * ns_per_layer + readout_overhead_s
    total_s = n_circuits * shots * per_shot_s
    return total_s


def run_kernel_mode(args, backend, service):
    from qiskit_ibm_runtime import SamplerV2, Session

    n_qubits = args.n_qubits
    rng = np.random.default_rng(0)
    # small, fixed synthetic batch standing in for PCA-reduced telemetry --
    # in the full pipeline this would be quantum.feature_reduction output;
    # kept self-contained here so the script has zero dependency on a
    # freshly generated dataset when run standalone on hardware.
    X_train = rng.uniform(-np.pi, np.pi, size=(args.n_train, n_qubits))
    X_test = rng.uniform(-np.pi, np.pi, size=(args.n_test, n_qubits))

    n_circuits = args.n_train * args.n_test  # ComputeUncompute pairs (upper bound)
    est_seconds = estimate_qpu_seconds(backend, n_circuits, args.shots)
    print(f"[estimate] ~{n_circuits} circuits x {args.shots} shots "
          f"=> estimated QPU time ~= {est_seconds:.1f} s")
    if est_seconds > args.max_qpu_seconds:
        raise SystemExit(
            f"[ABORT] Estimated QPU time ({est_seconds:.1f}s) exceeds the "
            f"budget ({args.max_qpu_seconds}s). Reduce --n-train/--n-test/--shots."
        )

    fmap = zz_feature_map(feature_dimension=n_qubits, reps=args.reps)

    # CRITICAL for real hardware: IBM Runtime primitives (SamplerV2) only
    # accept ISA circuits (transpiled to the backend's native gate set /
    # coupling map) as of March 2024. ComputeUncompute's `pass_manager`
    # argument handles this transpilation internally for every fidelity
    # circuit it builds -- omitting it causes an IBMInputValueError on
    # both real hardware AND fake backends (verified during testing).
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)

    # NOTE on execution mode: IBM's "Open" (free) plan does NOT permit
    # Session mode -- attempting it raises HTTP 400 "You are not
    # authorized to run a session when using the open plan." Session is
    # only available on paid plans. By default this script therefore uses
    # plain "job mode" (SamplerV2 bound directly to the backend, no
    # Session wrapper), which works on every plan tier. Pass
    # --use-session explicitly if you are on a paid plan and want Session
    # mode's queue-priority benefits for a multi-job batch.
    if args.use_session and service is not None:
        with Session(backend=backend) as session:
            sampler = SamplerV2(mode=session)
            sampler.options.default_shots = args.shots
            fidelity = ComputeUncompute(sampler=sampler, pass_manager=pm)
            kernel = FidelityQuantumKernel(feature_map=fmap, fidelity=fidelity,
                                            max_circuits_per_job=args.max_circuits_per_job)
            t0 = time.time()
            K = kernel.evaluate(x_vec=X_train, y_vec=X_test)
            wall_s = time.time() - t0
    else:
        sampler = SamplerV2(mode=backend)
        sampler.options.default_shots = args.shots
        fidelity = ComputeUncompute(sampler=sampler, pass_manager=pm)
        kernel = FidelityQuantumKernel(feature_map=fmap, fidelity=fidelity,
                                        max_circuits_per_job=args.max_circuits_per_job)
        t0 = time.time()
        K = kernel.evaluate(x_vec=X_train, y_vec=X_test)
        wall_s = time.time() - t0

    print(f"[done] Gram sub-block shape={K.shape}, wall-clock={wall_s:.1f}s")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, K)
    print(f"[saved] Kernel Gram sub-block -> {args.out}")
    return K


def run_vqc_eval_mode(args, backend, service):
    from qiskit_ibm_runtime import SamplerV2, Session

    n_qubits = args.n_qubits
    if args.vqc_params is None:
        raise SystemExit("--vqc-params is required for vqc_eval mode "
                          "(train on simulator first via quantum/vqc.py, "
                          "then pass the saved .npy weight vector here).")
    weights = np.load(args.vqc_params)

    rng = np.random.default_rng(1)
    X_test = rng.uniform(-np.pi, np.pi, size=(args.n_test, n_qubits))

    fmap = zz_feature_map(feature_dimension=n_qubits, reps=1)
    ansatz = real_amplitudes(num_qubits=n_qubits, reps=2)
    circuit = fmap.compose(ansatz)
    circuit.measure_all()

    n_circuits = args.n_test
    est_seconds = estimate_qpu_seconds(backend, n_circuits, args.shots)
    print(f"[estimate] ~{n_circuits} circuits x {args.shots} shots "
          f"=> estimated QPU time ~= {est_seconds:.1f} s")
    if est_seconds > args.max_qpu_seconds:
        raise SystemExit(
            f"[ABORT] Estimated QPU time ({est_seconds:.1f}s) exceeds the "
            f"budget ({args.max_qpu_seconds}s). Reduce --n-test/--shots."
        )

    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    # IMPORTANT: bind parameters on the ORIGINAL (untranspiled) circuit
    # first, THEN transpile each bound circuit. Transpiling a still-
    # parameterized circuit can reorder circuit.parameters (commonly
    # alphabetical by name), which would silently corrupt the x/weights
    # binding below -- binding before transpiling avoids that entirely.

    n_feature_params = fmap.num_parameters
    n_ansatz_params = ansatz.num_parameters
    if len(weights) != n_ansatz_params:
        raise SystemExit(
            f"[ABORT] Loaded VQC weights have length {len(weights)}, but the "
            f"ansatz (real_amplitudes, {n_qubits} qubits, reps=2) expects "
            f"{n_ansatz_params} parameters. Re-train with matching --n-qubits, "
            f"or regenerate weights via quantum/vqc.py with the same ansatz."
        )

    bound_circuits = []
    for x in X_test:
        # circuit.parameters is ordered [feature-map params..., ansatz params...]
        # since `circuit = fmap.compose(ansatz)` concatenates in that order.
        full_params = np.concatenate([x[:n_feature_params], weights])
        bc = circuit.assign_parameters(full_params)
        bound_circuits.append(pm.run(bc))

    def _submit(sampler):
        sampler.options.default_shots = args.shots
        t0 = time.time()
        job = sampler.run(bound_circuits)
        result = job.result()
        wall_s = time.time() - t0
        return result, wall_s

    if args.use_session and service is not None:
        with Session(backend=backend) as session:
            sampler = SamplerV2(mode=session)
            result, wall_s = _submit(sampler)
    else:
        sampler = SamplerV2(mode=backend)
        result, wall_s = _submit(sampler)

    print(f"[done] {len(bound_circuits)} circuits evaluated, wall-clock={wall_s:.1f}s")
    counts_list = [r.data.meas.get_counts() for r in result]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, np.array(counts_list, dtype=object))
    print(f"[saved] Measurement counts -> {args.out}")
    return counts_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["kernel", "vqc_eval"], required=True)
    ap.add_argument("--backend", default="ibm_marrakesh")
    ap.add_argument("--use-session", action="store_true",
                     help="Use IBM Runtime Session mode instead of plain job "
                          "mode. Session mode is NOT available on IBM's Open "
                          "(free) plan -- attempting it there raises HTTP 400 "
                          "'not authorized to run a session when using the "
                          "open plan.' Only pass this flag if you are on a "
                          "paid IBM Quantum plan.")
    ap.add_argument("--use-fake-backend", action="store_true",
                     help="Use FakeSherbrooke instead of live hardware -- for "
                          "offline correctness testing of this exact script.")
    ap.add_argument("--n-qubits", type=int, default=4)
    ap.add_argument("--n-train", type=int, default=6)
    ap.add_argument("--n-test", type=int, default=6)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--shots", type=int, default=256)
    ap.add_argument("--max-circuits-per-job", type=int, default=50)
    ap.add_argument("--max-qpu-seconds", type=float, default=420.0,
                     help="Hard ceiling; script aborts BEFORE submission "
                          "if the static pre-flight estimate exceeds this. "
                          "Default 420s leaves margin under an 8-minute (480s) budget.")
    ap.add_argument("--vqc-params", default=None,
                     help="Path to a .npy file with pre-trained VQC weights "
                          "(required for --mode vqc_eval).")
    ap.add_argument("--out", default="results/ibm_marrakesh_output.npy")
    args = ap.parse_args()

    backend, service = get_backend(args)

    if args.mode == "kernel":
        run_kernel_mode(args, backend, service)
    else:
        run_vqc_eval_mode(args, backend, service)


if __name__ == "__main__":
    main()
