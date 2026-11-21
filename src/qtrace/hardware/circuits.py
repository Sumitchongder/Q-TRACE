"""Qubit-pair selection and circuit construction for the hardware experiment."""

from __future__ import annotations

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qtrace.hardware.design import ExperimentDesign, Probe


def _two_qubit_gate(target):
    for name in ("cz", "ecr", "cx"):
        if name in target.operation_names:
            return name
    raise RuntimeError("backend exposes no supported two-qubit gate")


def select_links(backend, n_links=40, isolate=True, exclude=(), max_readout_error=0.05,
                 max_two_qubit_error=0.02):
    """Choose disjoint, well-calibrated (data, ancilla) pairs.

    Edges are ranked by two-qubit gate error plus both readout errors. With
    ``isolate=True`` no chosen qubit is coupled to a qubit of another
    chosen pair, which suppresses nearest-neighbour crosstalk between
    links. The qubit with the lower readout error becomes the data qubit.

    Returns ``(pairs, table)`` where ``table`` lists the calibration data
    of every selected pair.
    """
    target = backend.target
    gate = _two_qubit_gate(target)
    ro = {}
    for q in range(target.num_qubits):
        try:
            ro[q] = target["measure"][(q,)].error or 1.0
        except (KeyError, TypeError):
            ro[q] = 1.0
    edges = {}
    for qargs, props in target[gate].items():
        if props is None or props.error is None:
            continue
        a, b = sorted(qargs)
        if a in exclude or b in exclude:
            continue
        if (props.error > max_two_qubit_error or ro[a] > max_readout_error
                or ro[b] > max_readout_error):
            continue
        err = props.error + ro[a] + ro[b]
        edges[(a, b)] = min(err, edges.get((a, b), np.inf))
    neighbours = {}
    for (a, b) in edges:
        neighbours.setdefault(a, set()).add(b)
        neighbours.setdefault(b, set()).add(a)
    used, blocked, pairs, table = set(), set(), [], []
    for (a, b), err in sorted(edges.items(), key=lambda kv: kv[1]):
        if err >= 0.5 or a in used or b in used:
            continue
        if isolate and (a in blocked or b in blocked):
            continue
        d, e = (a, b) if ro[a] <= ro[b] else (b, a)
        pairs.append((int(d), int(e)))
        props = target[gate][(a, b)] if (a, b) in target[gate] else target[gate][(b, a)]
        table.append(dict(data=int(d), ancilla=int(e), two_qubit_error=float(props.error),
                          readout_error_data=float(ro[d]), readout_error_ancilla=float(ro[e])))
        used |= {a, b}
        blocked |= {a, b} | neighbours.get(a, set()) | neighbours.get(b, set())
        if len(pairs) >= n_links:
            break
    return pairs, table


def apply_probe(qc, probe: Probe, d, e, eve_basis):
    if probe.kind == "probe":
        if eve_basis == 0:
            qc.rzx(probe.strength, d, e)
        else:
            qc.rxx(probe.strength, d, e)
    elif probe.kind == "ad":
        qc.cry(2 * np.arcsin(np.sqrt(probe.strength)), d, e)
        qc.cx(e, d)


def build_circuit(design: ExperimentDesign, c: int, links=None) -> QuantumCircuit:
    """Circuit ``c`` of the experiment on ``2L`` virtual qubits.

    Virtual qubit ``2i`` is link ``i``'s data qubit and ``2i+1`` its
    ancilla. Register ``bob`` holds Bob's results and ``eve`` Eve's.
    ``links`` restricts the circuit to a subset (used by the noisy
    rehearsal, which simulates links one at a time).
    """
    links = range(design.n_links) if links is None else list(links)
    nl = design.n_links
    q = QuantumRegister(2 * nl, "q")
    bob = ClassicalRegister(nl, "bob")
    eve = ClassicalRegister(nl, "eve")
    qc = QuantumCircuit(q, bob, eve, name=f"bb84_c{c}")
    for i in links:
        a, al, _ = design.setting(c, i)
        if a:
            qc.x(q[2 * i])
        if al:
            qc.h(q[2 * i])
    qc.barrier()
    for i in links:
        apply_probe(qc, design.probe(c, i), q[2 * i], q[2 * i + 1], design.eve_basis[c][i])
    qc.barrier()
    for i in links:
        _, _, be = design.setting(c, i)
        if be:
            qc.h(q[2 * i])
        p = design.probe(c, i)
        if p.kind == "probe":           # Helstrom measurement: Y basis
            qc.sdg(q[2 * i + 1])
            qc.h(q[2 * i + 1])
        qc.measure(q[2 * i], bob[i])
        qc.measure(q[2 * i + 1], eve[i])
    return qc


def transpile_design(design: ExperimentDesign, backend, circuits=None, optimization_level=1):
    """Transpile with a fixed layout onto ``design.physical_pairs``.

    Raises if routing inserted SWAPs (pairs must be physically coupled).
    """
    layout = [q for pair in design.physical_pairs for q in pair]
    pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend,
                                      initial_layout=layout)
    circuits = circuits or [build_circuit(design, c) for c in range(design.n_circuits)]
    isa = pm.run(circuits)
    gate = _two_qubit_gate(backend.target)
    for logical, phys in zip(circuits, isa):
        n_ent = sum(1 for inst in logical.data if inst.operation.num_qubits == 2
                    and inst.operation.name != "barrier")
        n_phys = phys.count_ops().get(gate, 0) + phys.count_ops().get("swap", 0)
        if phys.count_ops().get("swap", 0):
            raise RuntimeError("routing inserted SWAP gates; check the selected pairs")
        if n_ent == 0 and n_phys:
            raise RuntimeError("unexpected two-qubit gates in an honest circuit")
    return isa
