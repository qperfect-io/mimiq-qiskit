"""Convert Qiskit ``QuantumCircuit`` ↔ MIMIQ ``Circuit``.

Coverage:

- All standard gates from :data:`mimiq_qiskit.gate_map.QISKIT_TO_MIMIQ`.
- ``Measure``, ``Reset``, ``Barrier``.
- ``UnitaryGate`` (Qiskit ``"unitary"``) maps to
  :class:`mimiqcircuits.GateCustom`, so arbitrary matrix gates (state
  preparation, custom ansätze) pass through without a manual
  decomposition.
- ``IfElseOp`` (mid-circuit measurement feed-forward) maps to
  :class:`mimiqcircuits.IfStatement`, for the common case of a single
  conditional gate with no ``else`` branch.
- Multi-register circuits: qubits and clbits are flattened in the order
  Qiskit assigns them on the circuit, so the resulting MIMIQ circuit
  uses 0-based indexing matching ``circuit.find_bit(q).index``.
- Global phase is dropped, because it does not affect measurement
  statistics or expectation values, the only quantities this bridge
  computes.
- Free :class:`qiskit.circuit.Parameter` references are not resolved;
  bind them with ``QuantumCircuit.assign_parameters`` before conversion.

Unknown gates raise :class:`UnsupportedGateError` with the Qiskit name,
so callers can decide whether to decompose upstream or extend the
mapping.
"""

from __future__ import annotations

from typing import Sequence

import mimiqcircuits as mc

from mimiq_qiskit.gate_map import MIMIQ_TO_QISKIT, QISKIT_TO_MIMIQ


class UnsupportedGateError(NotImplementedError):
    """Raised when a Qiskit or MIMIQ operation has no mapping."""


def _resolve_param(p) -> float:
    """Coerce a Qiskit parameter into a float.

    Symbolic ``Parameter`` references are not supported; bind them before
    conversion. The guard surfaces the error here rather than deep inside
    a MIMIQ gate constructor.
    """
    try:
        return float(p)
    except (TypeError, ValueError) as exc:
        raise UnsupportedGateError(
            f"unbound symbolic parameter {p!r}; call "
            "QuantumCircuit.assign_parameters before conversion"
        ) from exc


def _gate_for(name: str, params: Sequence[float]) -> mc.Operation:
    """Return the MIMIQ gate for a plain Qiskit gate ``name``.

    Only covers unitary gates from the gate map, not measure, reset,
    barrier, unitary, or control flow, which the caller handles
    separately.
    """
    factory = QISKIT_TO_MIMIQ.get(name)
    if factory is None:
        raise UnsupportedGateError(
            f"Qiskit operation {name!r} has no MIMIQ mapping; "
            "decompose it upstream or extend gate_map.QISKIT_TO_MIMIQ"
        )
    return factory(params)


def _condition_bitstring(condition, qc) -> tuple[mc.BitString, list[int]]:
    """Translate a Qiskit instruction ``condition`` into a MIMIQ
    ``BitString`` and the global clbit indices it tests.

    ``condition`` is ``(target, value)`` where ``target`` is a single
    ``Clbit`` or a ``ClassicalRegister``. The returned bitstring is
    LSB-first over the register bits, matching MIMIQ's convention that
    bit ``i`` of the condition reads clbit ``i`` of the listed targets.
    """
    from qiskit.circuit import Clbit, ClassicalRegister

    target, value = condition
    if isinstance(target, ClassicalRegister):
        bits = list(target)
    elif isinstance(target, Clbit):
        bits = [target]
    else:
        raise UnsupportedGateError(
            f"unsupported condition target {type(target).__name__}"
        )
    indices = [qc.find_bit(b).index for b in bits]
    bitstring = mc.BitString.fromint(len(bits), int(value))
    return bitstring, indices


def _convert_if_else(out: mc.Circuit, instr, qc, qidx) -> None:
    """Translate an ``IfElseOp`` into a MIMIQ ``IfStatement``.

    Supports the common feed-forward shape: a single conditional gate and
    no ``else`` branch. Richer control flow (multi-instruction bodies,
    ``else`` branches, loops) raises :class:`UnsupportedGateError`.
    """
    op = instr.operation
    blocks = op.blocks
    if len(blocks) != 1:
        raise UnsupportedGateError(
            "IfElseOp with an else branch is not supported; "
            "split it into separate conditionals"
        )
    if op.condition is None:
        raise UnsupportedGateError("IfElseOp without a condition")

    body = blocks[0]
    body_instrs = list(body.data)
    if len(body_instrs) != 1:
        raise UnsupportedGateError(
            "only single-gate conditional bodies are supported; "
            f"got {len(body_instrs)} instructions"
        )
    inner = body_instrs[0]
    if inner.clbits or inner.operation.name in ("measure", "reset", "barrier"):
        raise UnsupportedGateError(
            "conditional body must be a single unitary gate "
            "(no measure/reset/barrier/nested classical ops)"
        )

    inner_params = (
        [_resolve_param(p) for p in inner.operation.params]
        if inner.operation.params
        else []
    )
    inner_gate = _gate_for(inner.operation.name, inner_params)

    # The body circuit's qubit ``j`` is the outer instruction's qubit ``j``.
    body_qubit_index = {q: i for i, q in enumerate(body.qubits)}
    inner_qubits = [
        qidx(instr.qubits[body_qubit_index[q]]) for q in inner.qubits
    ]

    bitstring, cond_indices = _condition_bitstring(op.condition, qc)

    # IfStatement target layout is [op qubits..., condition bits...].
    out.push(
        mc.IfStatement(inner_gate, bitstring), *inner_qubits, *cond_indices
    )


def qiskit_to_mimiq(qc) -> mc.Circuit:
    """Convert a Qiskit :class:`QuantumCircuit` to a MIMIQ
    :class:`mimiqcircuits.Circuit`.

    Args:
        qc: A Qiskit ``QuantumCircuit``.

    Returns:
        A MIMIQ ``Circuit`` with operations pushed in the same order.

    Raises:
        UnsupportedGateError: An operation in ``qc`` has no mapping.
    """
    import numpy as np

    out = mc.Circuit()

    def qidx(qubit) -> int:
        return qc.find_bit(qubit).index

    def cidx(clbit) -> int:
        return qc.find_bit(clbit).index

    for instr in qc.data:
        op = instr.operation
        name = op.name
        qubits = [qidx(q) for q in instr.qubits]
        clbits = [cidx(c) for c in instr.clbits]

        if name == "measure":
            if len(qubits) != 1 or len(clbits) != 1:
                raise UnsupportedGateError(
                    "Measure must target exactly one qubit and one clbit"
                )
            out.push(mc.Measure(), qubits[0], clbits[0])
            continue

        if name == "reset":
            if len(qubits) != 1:
                raise UnsupportedGateError("Reset must target one qubit")
            out.push(mc.Reset(), qubits[0])
            continue

        if name == "barrier":
            out.push(mc.Barrier(len(qubits)), *qubits)
            continue

        if name == "unitary":
            matrix = np.asarray(op.to_matrix(), dtype=complex)
            out.push(mc.GateCustom(matrix), *qubits)
            continue

        if name == "if_else":
            _convert_if_else(out, instr, qc, qidx)
            continue

        params = [_resolve_param(p) for p in op.params] if op.params else []
        out.push(_gate_for(name, params), *qubits)

    return out


def mimiq_to_qiskit(circuit: mc.Circuit):
    """Convert a MIMIQ :class:`Circuit` back to a Qiskit
    :class:`QuantumCircuit`.

    Gates map onto concrete Qiskit gate classes, so the result is a fully
    defined circuit that Qiskit can transpile and simulate. The result
    uses single anonymous quantum and classical registers sized to the
    circuit; register identity from any original Qiskit circuit is not
    preserved.
    """
    import numpy as np
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    from qiskit.circuit.library import UnitaryGate

    nq = circuit.num_qubits()
    nb = circuit.num_bits()

    qreg = QuantumRegister(max(nq, 1), name="q")
    if nb > 0:
        creg = ClassicalRegister(nb, name="c")
        qc = QuantumCircuit(qreg, creg)
    else:
        qc = QuantumCircuit(qreg)

    for instruction in circuit:
        op = instruction.operation
        qubits = list(instruction.get_qubits())
        clbits = list(instruction.get_bits())

        if isinstance(op, mc.Measure):
            qc.measure(qubits[0], clbits[0])
            continue
        if isinstance(op, mc.Reset):
            qc.reset(qubits[0])
            continue
        if isinstance(op, mc.Barrier):
            qc.barrier(*qubits)
            continue
        if isinstance(op, mc.GateCustom):
            # ``op.matrix`` is a symengine matrix; coerce entrywise to a
            # numpy complex array for a Qiskit ``UnitaryGate``.
            sym = op.matrix
            rows, cols = sym.rows, sym.cols
            matrix = np.array(
                [[complex(sym[r, c]) for c in range(cols)] for r in range(rows)],
                dtype=complex,
            )
            qc.append(UnitaryGate(matrix), qubits)
            continue

        entry = MIMIQ_TO_QISKIT.get(type(op))
        if entry is None:
            raise UnsupportedGateError(
                f"MIMIQ operation {type(op).__name__} has no Qiskit mapping"
            )
        gate_cls, nparams = entry
        params = _extract_mimiq_params(op, nparams)
        qc.append(gate_cls(*params), qubits)

    return qc


def _extract_mimiq_params(op: mc.Operation, nparams: int) -> list[float]:
    """Pull ``nparams`` numeric parameters off a MIMIQ gate instance.

    MIMIQ gates declare their constructor argument names on the
    ``_parnames`` class tuple and store each as a matching instance
    attribute (see ``mimiqcircuits.operations.gates.standard``).
    """
    if nparams == 0:
        return []
    parnames: Sequence[str] = getattr(type(op), "_parnames", ())
    if len(parnames) < nparams:
        raise UnsupportedGateError(
            f"could not extract {nparams} parameters from "
            f"{type(op).__name__}; expected ``_parnames`` of length "
            f">= {nparams}, got {parnames!r}"
        )
    return [float(getattr(op, name)) for name in parnames[:nparams]]
