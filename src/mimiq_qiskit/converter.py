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
import numpy as np
from qiskit.circuit import ControlledGate

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


# A gate whose name is not in the map is decomposed through its own Qiskit
# ``definition``. Nesting is shallow in practice (``mcx`` on many controls
# is the deepest thing in the standard library), so the cap only exists to
# turn a pathological or self-referential definition into a clear error
# instead of a RecursionError.
_MAX_DEFINITION_DEPTH = 32


def _push_operation(out: mc.Circuit, op, qubits, clbits, depth: int) -> None:
    """Push one Qiskit operation onto ``out``, in terms of global indices.

    ``qubits`` and ``clbits`` are the MIMIQ indices the operation acts on.
    Unmapped gates recurse through their Qiskit ``definition``, so the
    converter covers the whole standard library rather than only the names
    listed in :data:`~mimiq_qiskit.gate_map.QISKIT_TO_MIMIQ`.
    """
    name = op.name

    if name == "measure":
        if len(qubits) != 1 or len(clbits) != 1:
            raise UnsupportedGateError(
                "Measure must target exactly one qubit and one clbit"
            )
        out.push(mc.Measure(), qubits[0], clbits[0])
        return

    if name == "reset":
        if len(qubits) != 1:
            raise UnsupportedGateError("Reset must target one qubit")
        out.push(mc.Reset(), qubits[0])
        return

    if name == "barrier":
        out.push(mc.Barrier(len(qubits)), *qubits)
        return

    if name == "delay":
        # Qiskit stores a duration plus a unit; MIMIQ ``Delay`` takes a
        # plain time, so the duration is normalised to seconds. ``dt`` is
        # backend-relative with no seconds equivalent here, so it is
        # passed through as given.
        out.push(mc.Delay(_delay_seconds(op)), *qubits)
        return

    if name == "unitary":
        matrix = np.asarray(op.to_matrix(), dtype=complex)
        # Qiskit indexes a multi-qubit matrix little-endian (its first
        # wire is the least significant bit), MIMIQ big-endian. Handing
        # the matrix over unchanged on reversed wires converts between
        # the two without permuting 4**n entries.
        out.push(mc.GateCustom(matrix), *reversed(qubits))
        return

    if name == "global_phase":
        # A global phase changes no measurement statistic or expectation
        # value, the only quantities this bridge computes.
        return

    factory = QISKIT_TO_MIMIQ.get(name)
    if factory is not None:
        params = [_resolve_param(p) for p in op.params] if op.params else []
        out.push(factory(params), *qubits)
        return

    control = _control_for(op)
    if control is not None:
        out.push(control, *qubits)
        return

    _push_definition(out, op, qubits, clbits, depth)


def _control_for(op):
    """Wrap a Qiskit ``ControlledGate`` as a MIMIQ ``Control``, or return
    ``None`` if it has no direct form.

    Worth the special case because Qiskit synthesises a controlled gate
    into dozens of primitives (a 4-control ``mcx`` is 69 of them) while
    MIMIQ carries the control natively and decomposes it itself. Only the
    all-ones control state maps directly; an open control needs the X
    conjugation that Qiskit's own definition already encodes.
    """
    if not isinstance(op, ControlledGate):
        return None
    nctrl = op.num_ctrl_qubits
    if op.ctrl_state != (1 << nctrl) - 1:
        return None

    base = op.base_gate
    factory = QISKIT_TO_MIMIQ.get(base.name)
    if factory is None:
        return None
    base_params = [_resolve_param(p) for p in base.params] if base.params else []
    return mc.Control(nctrl, factory(base_params))


def _push_definition(out: mc.Circuit, op, qubits, clbits, depth: int) -> None:
    """Decompose an unmapped gate through its Qiskit ``definition``.

    This is what lets composites Qiskit builds from other gates
    (``mcx`` with three or more controls, ``initialize``, ``r``, anything
    from ``QuantumCircuit.to_gate()``) convert without an entry of their
    own. Control flow and non-unitary operations have no usable
    definition and raise instead.
    """
    if depth >= _MAX_DEFINITION_DEPTH:
        raise UnsupportedGateError(
            f"Qiskit operation {op.name!r} nests definitions more than "
            f"{_MAX_DEFINITION_DEPTH} levels deep; decompose it upstream"
        )

    try:
        definition = op.definition
    except Exception:  # a definition Qiskit cannot build at all
        definition = None

    if definition is None:
        raise UnsupportedGateError(
            f"Qiskit operation {op.name!r} has no MIMIQ mapping and no "
            "definition to decompose; add it to "
            "gate_map.QISKIT_TO_MIMIQ or decompose it upstream"
        )

    if len(definition.qubits) != len(qubits):
        raise UnsupportedGateError(
            f"definition of {op.name!r} spans {len(definition.qubits)} "
            f"qubits but the operation acts on {len(qubits)}; ancilla-using "
            "definitions are not supported"
        )

    # The definition's own wires are positional: its qubit ``j`` is the
    # outer operation's qubit ``j``.
    qmap = dict(zip(definition.qubits, qubits))
    cmap = dict(zip(definition.clbits, clbits))
    for inner in definition.data:
        _push_operation(
            out,
            inner.operation,
            [qmap[q] for q in inner.qubits],
            [cmap[c] for c in inner.clbits],
            depth + 1,
        )


def _delay_seconds(op) -> float:
    """Duration of a Qiskit ``Delay`` in seconds."""
    scale = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9, "ps": 1e-12}
    return float(op.duration) * scale.get(op.unit, 1.0)


def qiskit_to_mimiq(qc) -> mc.Circuit:
    """Convert a Qiskit :class:`QuantumCircuit` to a MIMIQ
    :class:`mimiqcircuits.Circuit`.

    Args:
        qc: A Qiskit ``QuantumCircuit``.

    Returns:
        A MIMIQ ``Circuit`` with operations pushed in the same order.

    Raises:
        UnsupportedGateError: An operation in ``qc`` has neither a MIMIQ
            mapping nor a Qiskit definition to decompose.
    """
    out = mc.Circuit()

    def qidx(qubit) -> int:
        return qc.find_bit(qubit).index

    def cidx(clbit) -> int:
        return qc.find_bit(clbit).index

    for instr in qc.data:
        if instr.operation.name == "if_else":
            _convert_if_else(out, instr, qc, qidx)
            continue
        _push_operation(
            out,
            instr.operation,
            [qidx(q) for q in instr.qubits],
            [cidx(c) for c in instr.clbits],
            0,
        )

    return out


def mimiq_to_qiskit(circuit: mc.Circuit):
    """Convert a MIMIQ :class:`Circuit` back to a Qiskit
    :class:`QuantumCircuit`.

    Gates map onto concrete Qiskit gate classes, so the result is a fully
    defined circuit that Qiskit can transpile and simulate. Any unitary
    operation with no named counterpart -- ``Control``, ``Inverse``,
    ``Power``, ``Parallel``, ``GateCustom``, and gates Qiskit has no
    equivalent for such as ``GateSY`` or ``GateRNZ`` -- becomes a
    ``UnitaryGate`` carrying its matrix. That is operator-faithful but not
    structure-faithful, so a round trip through both converters preserves
    the unitary, not the gate names.

    The result uses single anonymous quantum and classical registers sized
    to the circuit; register identity from any original Qiskit circuit is
    not preserved.

    Raises:
        UnsupportedGateError: The circuit holds a non-unitary operation
            with no Qiskit equivalent (a noise channel, for instance), or
            a gate with unbound symbolic parameters.
    """
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

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
        if isinstance(op, mc.Delay):
            qc.delay(float(op.t), qubits[0], unit="s")
            continue
        if isinstance(op, mc.IfStatement):
            _append_if_statement(qc, op, qubits, clbits)
            continue

        entry = MIMIQ_TO_QISKIT.get(type(op))
        if entry is not None:
            gate_cls, nparams = entry
            params = _extract_mimiq_params(op, nparams)
            qc.append(gate_cls(*params), qubits)
            continue

        # No named counterpart: fall back to the operation's matrix. This
        # covers GateCustom, the composite wrappers, and MIMIQ-only gates.
        qc.append(_unitary_gate_for(op), list(reversed(qubits)))

    return qc


def _matrix_to_numpy(op):
    """Materialise a MIMIQ operation's matrix as a numpy complex array.

    ``matrix()`` hands back a symengine matrix, whose ``__array__`` numpy
    2 warns about, so the entries are coerced one at a time.
    """
    sym = op.matrix()
    return np.array(
        [
            [complex(sym[r, c]) for c in range(sym.cols)]
            for r in range(sym.rows)
        ],
        dtype=complex,
    )


def _unitary_gate_for(op):
    """Wrap an unmapped MIMIQ unitary as a Qiskit ``UnitaryGate``.

    The caller applies it on reversed wires, for the same endianness
    reason as the forward path.
    """
    from qiskit.circuit.library import UnitaryGate

    if not isinstance(op, mc.Gate):
        raise UnsupportedGateError(
            f"MIMIQ operation {type(op).__name__} is not unitary and has "
            "no Qiskit equivalent"
        )
    try:
        matrix = _matrix_to_numpy(op)
    except (TypeError, RuntimeError) as exc:
        raise UnsupportedGateError(
            f"could not read a numeric matrix off {type(op).__name__}; "
            "evaluate its symbolic parameters first"
        ) from exc
    return UnitaryGate(matrix, label=op.name)


def _append_if_statement(qc, op, qubits, clbits) -> None:
    """Rebuild a MIMIQ ``IfStatement`` as a Qiskit ``if_test`` block.

    An ``IfStatement``'s targets are laid out as ``[op qubits...,
    condition bits...]``, and its bitstring is LSB-first over those
    condition bits, matching how :func:`qiskit_to_mimiq` emits it.
    """
    inner = op.get_operation()
    bitstring = op.get_bitstring()
    value = sum(1 << i for i, bit in enumerate(bitstring.to01()) if bit == "1")

    cond_bits = [qc.clbits[i] for i in clbits]
    body_qubits = [qc.qubits[i] for i in qubits]

    # Qiskit conditions on a single Clbit or a whole register. A MIMIQ
    # condition over several loose bits has no single-expression Qiskit
    # form, so build one over a synthetic register only when the bits are
    # already contiguous from the start of the circuit's register.
    if len(cond_bits) == 1:
        condition = (cond_bits[0], value)
    elif any(list(reg) == cond_bits for reg in qc.cregs):
        condition = (next(r for r in qc.cregs if list(r) == cond_bits), value)
    else:
        raise UnsupportedGateError(
            "IfStatement conditions on classical bits "
            f"{clbits}, which Qiskit cannot express as a single "
            "clbit or register comparison"
        )

    entry = MIMIQ_TO_QISKIT.get(type(inner))
    if entry is not None:
        gate_cls, nparams = entry
        gate = gate_cls(*_extract_mimiq_params(inner, nparams))
        targets = body_qubits
    else:
        gate = _unitary_gate_for(inner)
        targets = list(reversed(body_qubits))

    with qc.if_test(condition):
        qc.append(gate, targets)


def _extract_mimiq_params(op: mc.Operation, nparams: int) -> list[float]:
    """Pull the first ``nparams`` numeric parameters off a MIMIQ gate.

    ``getparams()`` is the public accessor every operation implements, and
    it can report more than the Qiskit counterpart takes: ``GateU`` adds a
    trailing ``gamma`` global phase that ``UGate`` has no slot for. Extra
    parameters are dropped, which is why the mapping records how many
    Qiskit wants.
    """
    if nparams == 0:
        return []
    params = op.getparams()
    if len(params) < nparams:
        raise UnsupportedGateError(
            f"could not extract {nparams} parameters from "
            f"{type(op).__name__}; getparams() gave {params!r}"
        )
    return [float(v) for v in params[:nparams]]
