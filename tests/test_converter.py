"""Converter unit tests — no MIMIQ backend required."""

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


def test_unknown_gate_raises():
    qc = QuantumCircuit(2)
    # Fabricate an instruction Qiskit knows but we don't map.
    from qiskit.circuit import Gate

    fake = Gate(name="totally_made_up", num_qubits=2, params=[])
    qc.append(fake, [0, 1])
    with pytest.raises(UnsupportedGateError):
        qiskit_to_mimiq(qc)


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
