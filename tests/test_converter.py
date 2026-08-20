"""Converter unit tests; no MIMIQ backend required."""

from __future__ import annotations

import math

import mimiqcircuits as mc
import numpy as np
import pytest
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from mimiq_qiskit.converter import (
    UnsupportedGateError,
    mimiq_to_qiskit,
    qiskit_to_mimiq,
)


def _to_complex(symengine_matrix) -> np.ndarray:
    """Entrywise coercion of a symengine matrix; ``np.array`` on one goes
    through its ``__array__``, which numpy 2 warns about."""
    return np.array(symengine_matrix.tolist(), dtype=complex)


def test_empty_circuit_roundtrip():
    qc = QuantumCircuit(2, 2)
    c = qiskit_to_mimiq(qc)
    assert c.num_qubits() == 0  # nothing pushed → MIMIQ infers from ops
    assert len(list(c)) == 0


def test_standard_gates_mapped():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.t(1)
    qc.swap(0, 1)

    c = qiskit_to_mimiq(qc)
    ops = [type(instr.operation) for instr in c]
    assert ops == [mc.GateH, mc.GateCX, mc.GateT, mc.GateSWAP]


def test_parametric_gates():
    qc = QuantumCircuit(1)
    qc.rx(0.5, 0)
    qc.ry(math.pi / 3, 0)
    qc.rz(-1.0, 0)
    qc.u(0.1, 0.2, 0.3, 0)

    c = qiskit_to_mimiq(qc)
    instrs = list(c)
    assert isinstance(instrs[0].operation, mc.GateRX)
    assert isinstance(instrs[1].operation, mc.GateRY)
    assert isinstance(instrs[2].operation, mc.GateRZ)
    assert isinstance(instrs[3].operation, mc.GateU)


def test_measure_and_reset():
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.measure(0, 0)
    qc.reset(1)
    qc.measure(1, 1)

    c = qiskit_to_mimiq(qc)
    instrs = list(c)
    ops = [type(i.operation) for i in instrs]
    assert ops == [mc.GateH, mc.Measure, mc.Reset, mc.Measure]
    assert list(instrs[1].get_qubits()) == [0]
    assert list(instrs[1].get_bits()) == [0]
    assert list(instrs[3].get_qubits()) == [1]
    assert list(instrs[3].get_bits()) == [1]


def test_barrier():
    qc = QuantumCircuit(3)
    qc.barrier(0, 1, 2)
    c = qiskit_to_mimiq(qc)
    instrs = list(c)
    assert len(instrs) == 1
    assert isinstance(instrs[0].operation, mc.Barrier)
    assert list(instrs[0].get_qubits()) == [0, 1, 2]


def test_multi_register_qubit_indexing():
    qr0 = QuantumRegister(2, "a")
    qr1 = QuantumRegister(2, "b")
    cr = ClassicalRegister(4, "c")
    qc = QuantumCircuit(qr0, qr1, cr)
    qc.cx(qr0[1], qr1[0])  # global indices 1 and 2
    qc.measure(qr1[1], cr[3])  # global indices 3 and 3

    c = qiskit_to_mimiq(qc)
    instrs = list(c)
    assert list(instrs[0].get_qubits()) == [1, 2]
    assert list(instrs[1].get_qubits()) == [3]
    assert list(instrs[1].get_bits()) == [3]


def test_unbound_parameter_raises():
    from qiskit.circuit import Parameter

    qc = QuantumCircuit(1)
    qc.rx(Parameter("θ"), 0)
    with pytest.raises(UnsupportedGateError):
        qiskit_to_mimiq(qc)


def test_unknown_gate_with_no_definition_raises():
    """No mapping and nothing to decompose is the only forward dead end."""
    from qiskit.circuit import Gate

    qc = QuantumCircuit(2)
    qc.append(Gate(name="totally_made_up", num_qubits=2, params=[]), [0, 1])
    with pytest.raises(UnsupportedGateError, match="no definition"):
        qiskit_to_mimiq(qc)


def test_controlled_gate_becomes_a_native_control():
    """A closed control maps to MIMIQ's own Control, not a decomposition."""
    from qiskit.circuit.library import RXXGate

    qc = QuantumCircuit(3)
    qc.append(RXXGate(0.3).control(1), [0, 1, 2])

    instructions = list(qiskit_to_mimiq(qc))
    assert len(instructions) == 1
    control = instructions[0].operation
    # MIMIQ names its own controlled gates as Control subclasses (GateCX
    # is one), so the check is for the generic wrapper specifically.
    assert type(control) is mc.Control
    assert isinstance(control.get_operation(), mc.GateRXX)


def test_open_control_falls_back_to_the_definition():
    """An open control has no direct MIMIQ form, so it decomposes."""
    from qiskit.circuit.library import XGate

    qc = QuantumCircuit(2)
    qc.append(XGate().control(1, ctrl_state=0), [0, 1])

    # X-conjugated CX, rather than a single Control.
    ops = [type(i.operation) for i in qiskit_to_mimiq(qc)]
    assert ops == [mc.GateX, mc.GateCX, mc.GateX]


def test_unmapped_gate_decomposes_through_its_definition():
    block = QuantumCircuit(2, name="block")
    block.h(0)
    block.cx(0, 1)

    qc = QuantumCircuit(3)
    qc.append(block.to_gate(), [1, 2])

    instructions = list(qiskit_to_mimiq(qc))
    assert [type(i.operation) for i in instructions] == [mc.GateH, mc.GateCX]
    # The definition's wire 0 is the outer instruction's wire 0, so the
    # block's qubit 0 must land on the circuit's qubit 1.
    assert list(instructions[0].get_qubits()) == [1]
    assert list(instructions[1].get_qubits()) == [1, 2]


def test_delay_carries_its_duration_in_seconds():
    qc = QuantumCircuit(1)
    qc.delay(250, 0, unit="ns")

    delay = list(qiskit_to_mimiq(qc))[0].operation
    assert isinstance(delay, mc.Delay)
    assert pytest.approx(float(delay.t)) == 250e-9


def test_global_phase_gate_is_dropped():
    from qiskit.circuit.library import GlobalPhaseGate

    qc = QuantumCircuit(1)
    qc.h(0)
    qc.append(GlobalPhaseGate(0.8), [])

    assert [type(i.operation) for i in qiskit_to_mimiq(qc)] == [mc.GateH]


def test_reverse_falls_back_to_a_unitary_gate():
    """MIMIQ-only unitaries have no named Qiskit form, but do have a
    matrix, which beats refusing to convert them."""
    circuit = mc.Circuit()
    circuit.push(mc.GateSY(), 0)
    circuit.push(mc.Inverse(mc.GateU(0.1, 0.2, 0.3)), 0)
    circuit.push(mc.Parallel(2, mc.GateH()), 0, 1)

    names = [i.operation.name for i in mimiq_to_qiskit(circuit).data]
    assert names == ["unitary", "unitary", "unitary"]


def test_reverse_fallback_keeps_multiqubit_wire_order():
    """The matrix fallback must reverse wires, like the named path does.

    ``Control(1, GateRXX)`` puts its control on wire 0, and reversing the
    wires would move it to wire 2. RXX is symmetric in its two targets, so
    that is the only difference, and it is invisible to a round trip
    through both converters. Compared against Qiskit's own controlled RXX
    instead.
    """
    from qiskit.circuit.library import RXXGate
    from qiskit.quantum_info import Operator

    circuit = mc.Circuit()
    circuit.push(mc.Control(1, mc.GateRXX(0.3)), 0, 1, 2)

    back = mimiq_to_qiskit(circuit)
    # Control over a parametric two-qubit gate has no named Qiskit form,
    # so this exercises the matrix fallback.
    assert back.data[0].operation.name == "unitary"

    reference = QuantumCircuit(3)
    reference.append(RXXGate(0.3).control(1), [0, 1, 2])
    assert Operator(back).equiv(Operator(reference))


@pytest.mark.skipif(
    not hasattr(mc, "GateCustomDiagonal"),
    reason="GateCustomDiagonal arrived in mimiqcircuits 0.27",
)
def test_reverse_handles_a_diagonal_custom_gate():
    """``GateCustomDiagonal`` is 0.27's new gate, and the declared
    dependency range spans that minor, so the fallback has to cover it.

    It stores only the ``2**n`` diagonal entries rather than a full
    matrix, which is exactly the kind of new operation a matrix fallback
    picks up for free — this pins that it actually does, wire order and
    all.
    """
    from qiskit.quantum_info import Operator

    phases = np.exp(1j * np.array([0.0, 0.3, -0.7, 1.1]))
    circuit = mc.Circuit()
    circuit.push(mc.GateCustomDiagonal(phases), 0, 1)

    back = mimiq_to_qiskit(circuit)
    assert [i.operation.name for i in back.data] == ["unitary"]

    # MIMIQ indexes the diagonal big-endian and the converter applies it
    # on reversed wires, so the Qiskit operator carries the diagonal with
    # its two index bits transposed.
    want = np.diag(phases.reshape(2, 2).T.reshape(-1))
    assert np.allclose(np.asarray(Operator(back).data, dtype=complex), want)


@pytest.mark.skipif(
    not hasattr(mc, "fuse_circuit"),
    reason="fuse_circuit arrived in mimiqcircuits 0.26.2",
)
def test_reverse_handles_a_fused_circuit():
    """Fusion is the normal producer of block gates, so its output is
    what the reverse converter meets in practice."""
    circuit = mc.Circuit()
    for qubit in range(3):
        circuit.push(mc.GateRZ(0.2 * (qubit + 1)), qubit)
    circuit.push(mc.GateCP(0.4), 0, 1)

    fused = mc.fuse_circuit(circuit, max_support=3)
    names = [i.operation.name for i in mimiq_to_qiskit(fused).data]
    assert names, "fusion produced nothing to convert"
    assert all(name in {"unitary", "rz"} for name in names), names


def test_reverse_rejects_a_symbolic_gate():
    import symengine as se

    circuit = mc.Circuit()
    circuit.push(mc.Power(mc.GateRX(se.Symbol("x")), 2), 0)

    with pytest.raises(UnsupportedGateError, match="symbolic"):
        mimiq_to_qiskit(circuit)


def test_mimiq_to_qiskit_roundtrip_standard_gates():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.rz(0.7, 1)

    c = qiskit_to_mimiq(qc)
    qc2 = mimiq_to_qiskit(c)
    # Names should round-trip; parameter values preserved.
    names = [instr.operation.name for instr in qc2.data]
    assert names == ["h", "cx", "rz"]
    assert pytest.approx(float(qc2.data[2].operation.params[0])) == 0.7


def test_reverse_converter_produces_simulable_circuit():
    # The reverse converter must emit concrete gate classes (not opaque
    # named gates), so the result is operator-equivalent and transpilable.
    from qiskit.quantum_info import Operator

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.rz(0.7, 1)
    qc.swap(0, 1)

    back = mimiq_to_qiskit(qiskit_to_mimiq(qc))
    assert Operator(qc).equiv(Operator(back))


def test_unitary_gate_roundtrip():
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import Operator

    u = np.array([[0, 1j], [-1j, 0]], dtype=complex)
    qc = QuantumCircuit(1)
    qc.append(UnitaryGate(u), [0])

    c = qiskit_to_mimiq(qc)
    assert isinstance(list(c)[0].operation, mc.GateCustom)

    back = mimiq_to_qiskit(c)
    assert back.data[0].operation.name == "unitary"
    assert Operator(qc).equiv(Operator(back))


def test_multiqubit_unitary_matches_mimiq_qubit_order():
    """A 2+ qubit ``UnitaryGate`` must land on MIMIQ as the same operator.

    Qiskit indexes a matrix little-endian and MIMIQ big-endian, so a
    round-trip through both converters hides an ordering mistake; compare
    against MIMIQ's own equivalent gate instead. CX is asymmetric under
    the swap, so it fails loudly if the convention is dropped.
    """
    from qiskit.circuit.library import CCXGate, CXGate, UnitaryGate
    from qiskit.quantum_info import Operator

    for qiskit_gate, mimiq_gate, nq in (
        (CXGate(), mc.GateCX(), 2),
        (CCXGate(), mc.GateCCX(), 3),
    ):
        qc = QuantumCircuit(nq)
        qc.append(UnitaryGate(np.asarray(qiskit_gate.to_matrix())), range(nq))

        converted = qiskit_to_mimiq(qc)
        reference = mc.Circuit()
        reference.push(mimiq_gate, *range(nq))

        got = _to_complex(converted.instructions[0].matrix())
        want = _to_complex(reference.instructions[0].matrix())
        assert np.allclose(got, want), f"{type(mimiq_gate).__name__} mismatch"

        # And the reverse converter must undo it, not double-apply it.
        back = mimiq_to_qiskit(converted)
        assert Operator(qc).equiv(Operator(back))


def test_if_else_maps_to_ifstatement():
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.measure(0, 0)
    with qc.if_test((qc.clbits[0], 1)):
        qc.x(1)

    c = qiskit_to_mimiq(qc)
    ifs = [i for i in c if isinstance(i.operation, mc.IfStatement)]
    assert len(ifs) == 1
    assert isinstance(ifs[0].operation.op, mc.GateX)
    assert list(ifs[0].get_qubits()) == [1]


def test_if_else_register_condition():
    # A register==value condition becomes a BitString over the register's
    # bits, LSB first (creg[0] is the least significant bit).
    qc = QuantumCircuit(1, 2)
    with qc.if_test((qc.cregs[0], 2)):  # binary 10 → creg[1]=1
        qc.x(0)

    c = qiskit_to_mimiq(qc)
    ifs = [i for i in c if isinstance(i.operation, mc.IfStatement)][0]
    assert ifs.operation._bitstring.to01() == "01"  # bit0=0, bit1=1


def test_if_statement_round_trips_a_single_bit_condition():
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.measure(0, 0)
    with qc.if_test((qc.clbits[0], 1)):
        qc.x(1)

    back = mimiq_to_qiskit(qiskit_to_mimiq(qc))
    conditionals = [i for i in back.data if i.operation.name == "if_else"]
    assert len(conditionals) == 1

    body = conditionals[0].operation.blocks[0]
    assert [i.operation.name for i in body.data] == ["x"]
    # The condition must survive as the same clbit tested against 1.
    target, value = conditionals[0].operation.condition
    assert back.find_bit(target).index == 0
    assert value == 1

    # And converting forward again must land on the same MIMIQ shape.
    again = [i for i in qiskit_to_mimiq(back) if isinstance(i.operation, mc.IfStatement)]
    assert len(again) == 1
    assert isinstance(again[0].operation.get_operation(), mc.GateX)
    assert list(again[0].get_qubits()) == [1]


def test_if_statement_round_trips_a_register_condition():
    qc = QuantumCircuit(1, 2)
    with qc.if_test((qc.cregs[0], 2)):
        qc.x(0)

    back = mimiq_to_qiskit(qiskit_to_mimiq(qc))
    target, value = back.data[0].operation.condition
    assert list(target) == list(back.cregs[0])
    assert value == 2


def test_if_statement_with_a_matrix_only_body():
    """A conditional whose body has no named Qiskit gate still converts."""
    circuit = mc.Circuit()
    circuit.push(
        mc.IfStatement(mc.GateSY(), mc.BitString("1")), 0, 1
    )

    back = mimiq_to_qiskit(circuit)
    body = back.data[0].operation.blocks[0]
    assert [i.operation.name for i in body.data] == ["unitary"]


def test_if_else_with_else_branch_unsupported():
    qc = QuantumCircuit(1, 1)
    with qc.if_test((qc.clbits[0], 1)) as else_:
        qc.x(0)
    with else_:
        qc.z(0)
    with pytest.raises(UnsupportedGateError):
        qiskit_to_mimiq(qc)


def test_if_else_multi_gate_body_unsupported():
    qc = QuantumCircuit(2, 1)
    with qc.if_test((qc.clbits[0], 1)):
        qc.x(0)
        qc.x(1)
    with pytest.raises(UnsupportedGateError):
        qiskit_to_mimiq(qc)
