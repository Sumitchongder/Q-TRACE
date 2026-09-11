"""
Q-TRACE :: quantum.decode_hardware_counts
=============================================
Converts raw measurement-count dictionaries returned by
quantum/ibm_runtime_job.py's --mode vqc_eval into class-1 probabilities,
using the SAME parity-based interpret function that qiskit_machine_learning's
VQC uses internally by default (see qiskit_machine_learning.algorithms.
classifiers.vqc.VQC._get_interpret): the measured bitstring, read as an
unsigned integer, is mapped to a class via `bin(x).count('1') % num_classes`.

This lets you take the raw .npy of Counts dicts saved by a real-hardware
--mode vqc_eval run and get a proba vector directly comparable to
predict_proba_vqc()'s simulator output, for use in evaluation/metrics.py's
detection_metrics() or in your own manuscript-table code.

Usage:
    python -m quantum.decode_hardware_counts \
        --counts results/vqc_ibm_marrakesh.npy \
        --out results/vqc_ibm_marrakesh_proba.npy
"""

import argparse
import numpy as np


def parity_class(bitstring_int: int, num_classes: int = 2) -> int:
    """Identical logic to VQC's default interpret function."""
    return bin(bitstring_int).count("1") % num_classes


def counts_to_class1_proba(counts: dict, num_classes: int = 2) -> float:
    """
    counts: dict mapping bitstring (e.g. '0110') -> shot count.
    Returns P(class == 1) under the parity interpretation.
    """
    total = sum(counts.values())
    class1 = 0
    for bitstring, n in counts.items():
        x_int = int(bitstring, 2)
        if parity_class(x_int, num_classes) == 1:
            class1 += n
    return class1 / total if total > 0 else float("nan")


def decode_all(counts_array, num_classes: int = 2) -> np.ndarray:
    return np.array([counts_to_class1_proba(c, num_classes) for c in counts_array])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", required=True,
                     help="Path to the .npy saved by ibm_runtime_job.py --mode vqc_eval")
    ap.add_argument("--num-classes", type=int, default=2)
    ap.add_argument("--out", default=None,
                     help="Where to save the decoded probability array. "
                          "Defaults to <counts>_proba.npy")
    args = ap.parse_args()

    counts_array = np.load(args.counts, allow_pickle=True)
    proba = decode_all(counts_array, args.num_classes)

    out_path = args.out or args.counts.replace(".npy", "_proba.npy")
    np.save(out_path, proba)

    print(f"Decoded {len(proba)} circuit results.")
    print("P(class=1) per test point:", np.round(proba, 4))
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
