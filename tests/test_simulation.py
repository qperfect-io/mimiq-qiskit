"""End-to-end tests against a real MIMIQ simulator.

The other test modules use stub runners, which check plumbing but cannot
catch a semantic mistake: a gate mapped to the wrong MIMIQ counterpart, a
parameter passed in the wrong order, or a qubit-ordering convention
dropped all produce a perfectly well-formed result. Those only surface by
simulating and comparing against an independent reference.

``mimiq-exaqt`` is that simulator (a ``mimiqcircuits`` ``LocalBackend``,
so ``MimiqBackend`` wraps it like any other) and Qiskit's own
``Statevector`` is that reference.

The comparison rests on one alignment: :func:`qiskit_to_mimiq` maps Qiskit
qubit ``j`` to MIMIQ qubit ``j``, and ``BitString.fromint(n, i)`` puts bit
``j`` of ``i`` on MIMIQ qubit ``j``, which is also how Qiskit indexes a
``Statevector``. So amplitude ``i`` means the same basis state on both
sides and the two vectors compare entrywise.
"""

from __future__ import annotations

import math

import mimiqcircuits as mc
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import get_standard_gate_name_mapping
from qiskit.quantum_info import Operator, Statevector

from mimiq_qiskit import (
    MimiqBackend,
    MimiqEstimatorV2,
    MimiqSamplerV2,
    mimiq_to_qiskit,
    qiskit_to_mimiq,
)
from mimiq_qiskit.gate_map import QISKIT_TO_MIMIQ

exaqt = pytest.importorskip("exaqt", reason="mimiq-exaqt is not installed")


def _runner() -> "exaqt.ExaqtQCS":
    return exaqt.ExaqtQCS()


def _mimiq_statevector(qc: QuantumCircuit) -> np.ndarray:
    """Simulate ``qc`` through the converter and read back its amplitudes.

    Appends one MIMIQ ``Amplitude`` probe per basis state, each writing
    into its own z-variable, then runs the circuit on exaqt.
    """
    circuit = qiskit_to_mimiq(qc)
    nq = qc.num_qubits
    for i in range(2**nq):
        circuit.push(mc.Amplitude(mc.BitString.fromint(nq, i)), i)

    results = _runner().execute(circuit, nsamples=1, num_qubits=nq)
    return np.asarray(results.zstates[0], dtype=complex)


def _assert_same_state(qc: QuantumCircuit, *, atol: float = 1e-9) -> None:
    """Assert MIMIQ and Qiskit agree on the state ``qc`` prepares.

    Compared up to a global phase, which :func:`qiskit_to_mimiq` documents
    as not preserved.
    """
    got = _mimiq_statevector(qc)
    want = np.asarray(Statevector(qc).data, dtype=complex)

    overlap = np.vdot(want, got)
    assert abs(abs(overlap) - 1.0) < 1e-6, (
        f"states differ by more than a global phase: |<want|got>| = "
        f"{abs(overlap):.6f}"
    )
    phase = overlap / abs(overlap) if abs(overlap) > 0 else 1.0
    np.testing.assert_allclose(got, want * phase, atol=atol)


def _generic_state(qc: QuantumCircuit) -> None:
    """Rotate every qubit to a distinct generic state.

    A gate applied to |0…0⟩ hides mistakes: a swapped control and target,
    a permuted wire, or a mixed-up parameter can all leave the state
    untouched. Starting from a state with no symmetry between qubits makes
    those differences observable.
    """
    for q in range(qc.num_qubits):
        qc.u(0.7 + 0.31 * q, 0.29 + 0.43 * q, 0.13 + 0.57 * q, q)


# Gates the converter maps that Qiskit's standard-name table does not
# list: aliases for a gate under another name, and gates built by a
# method rather than exported under a name.
_EXTRA_GATES: dict[str, tuple[str, int]] = {
    "cnot": ("cx", 2),
    "toffoli": ("ccx", 3),
    "fredkin": ("cswap", 3),
    "c3x": ("c3x", 4),
    "iswap_dg": ("iswap_dg", 2),
}


def _gate_instance(name: str):
    """Build a concrete, fully bound instance of the named Qiskit gate."""
    standard = get_standard_gate_name_mapping()

    if name in _EXTRA_GATES:
        canonical, _ = _EXTRA_GATES[name]
        if canonical == "c3x":
            from qiskit.circuit.library import C3XGate

            return C3XGate()
        if canonical == "iswap_dg":
            from qiskit.circuit.library import iSwapGate

            return iSwapGate().inverse()
        gate = standard[canonical]
    else:
        gate = standard[name]

    if not gate.params:
        return gate
    # Distinct, non-special angles: equal or symmetric values would let a
    # transposed parameter pair pass. A ``Gate`` cannot rebind its own
    # parameters, so build a fresh instance from the same class.
    values = [0.37, 1.13, -0.61, 0.89][: len(gate.params)]
    return type(gate)(*values)


@pytest.mark.parametrize("name", sorted(QISKIT_TO_MIMIQ))
def test_gate_map_semantics_against_statevector(name):
    """Every mapped gate must act on MIMIQ exactly as it does on Qiskit.

    This is the test that pins the gate map down: it catches a gate
    pointed at the wrong MIMIQ class, parameters passed in the wrong
    order, and controls or targets swapped, none of which a conversion
    round trip or a stubbed backend can see.
    """
    gate = _gate_instance(name)
    qc = QuantumCircuit(gate.num_qubits)
    _generic_state(qc)
    qc.append(gate, range(gate.num_qubits))

    _assert_same_state(qc)


@pytest.mark.parametrize("name", sorted(QISKIT_TO_MIMIQ))
def test_gate_map_reverse_is_operator_faithful(name):
    """Converting back to Qiskit must preserve the operator."""
    gate = _gate_instance(name)
    qc = QuantumCircuit(gate.num_qubits)
    qc.append(gate, range(gate.num_qubits))

    back = mimiq_to_qiskit(qiskit_to_mimiq(qc))
    assert Operator(qc).equiv(Operator(back))


@pytest.mark.parametrize("nq", [2, 3])
def test_multiqubit_unitary_qubit_order(nq):
    """A ``UnitaryGate`` must keep its wiring across the endianness swap.

    Qiskit indexes a matrix little-endian and MIMIQ big-endian. A matrix
    handed over unchanged silently transposes control and target, and a
    conversion round trip cannot see it because both directions would be
    wrong the same way.
    """
    from qiskit.circuit.library import UnitaryGate

    reference = QuantumCircuit(nq)
    reference.mcx(list(range(nq - 1)), nq - 1)
    matrix = np.asarray(Operator(reference).data, dtype=complex)

    qc = QuantumCircuit(nq)
    _generic_state(qc)
    qc.append(UnitaryGate(matrix), range(nq))

    _assert_same_state(qc)


def test_controlled_gate_uses_native_control():
    """``ControlledGate`` should become one MIMIQ ``Control``.

    Qiskit synthesises a multi-controlled gate into dozens of primitives;
    MIMIQ carries the control natively. Converting through the definition
    would still be correct, just far larger, so this pins the shape as
    well as the result.
    """
    qc = QuantumCircuit(5)
    qc.mcx([0, 1, 2, 3], 4)

    converted = qiskit_to_mimiq(qc)
    instructions = list(converted)
    assert len(instructions) == 1
    # MIMIQ names its own controlled gates as Control subclasses, so this
    # asserts on the generic wrapper specifically.
    assert type(instructions[0].operation) is mc.Control

    checked = QuantumCircuit(5)
    _generic_state(checked)
    checked.mcx([0, 1, 2, 3], 4)
    _assert_same_state(checked)


def test_open_control_state_falls_back_to_definition():
    """An open control has no direct MIMIQ form but must still convert."""
    from qiskit.circuit.library import XGate

    qc = QuantumCircuit(2)
    _generic_state(qc)
    qc.append(XGate().control(1, ctrl_state=0), [0, 1])

    _assert_same_state(qc)


def test_composite_gate_decomposes_through_definition():
    """A gate with no mapping converts via its Qiskit ``definition``."""
    block = QuantumCircuit(2, name="block")
    block.h(0)
    block.cx(0, 1)
    block.rz(0.4, 1)

    qc = QuantumCircuit(2)
    _generic_state(qc)
    qc.append(block.to_gate(), [0, 1])

    _assert_same_state(qc)


def test_initialize_converts():
    """``initialize`` is a composite of reset and rotations, not a gate."""
    qc = QuantumCircuit(2)
    amplitudes = np.array([0.5, 0.5, 0.5, -0.5])
    qc.initialize(amplitudes, [0, 1])

    _assert_same_state(qc)


def test_delay_is_an_identity():
    """MIMIQ ``Delay`` must not disturb the state."""
    qc = QuantumCircuit(1)
    _generic_state(qc)
    qc.delay(120, 0, unit="ns")

    converted = qiskit_to_mimiq(qc)
    assert any(isinstance(i.operation, mc.Delay) for i in converted)
    _assert_same_state(qc)


def test_global_phase_gate_is_dropped():
    """A global phase affects nothing this bridge computes."""
    from qiskit.circuit.library import GlobalPhaseGate

    qc = QuantumCircuit(1)
    _generic_state(qc)
    qc.append(GlobalPhaseGate(0.8), [])

    _assert_same_state(qc)


def test_bell_state_counts():
    backend = MimiqBackend(_runner(), num_qubits=8)
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])

    counts = backend.run(qc, shots=2000, seed=1234).result().get_counts()
    assert set(counts) == {"00", "11"}
    assert sum(counts.values()) == 2000
    # Both outcomes at roughly half; a 5-sigma band on 2000 shots is ~±112.
    assert abs(counts["00"] - 1000) < 150


def test_ghz_state_counts():
    backend = MimiqBackend(_runner(), num_qubits=8)
    qc = QuantumCircuit(4, 4)
    qc.h(0)
    for q in range(3):
        qc.cx(q, q + 1)
    qc.measure(range(4), range(4))

    counts = backend.run(qc, shots=500, seed=7).result().get_counts()
    assert set(counts) == {"0000", "1111"}


def test_measurement_maps_qubits_to_the_requested_clbits():
    """A permuted measure map must land in the permuted clbits.

    Qiskit prints counts most-significant-clbit first, so measuring qubit
    0 into clbit 2 puts its value at the left of the string.
    """
    backend = MimiqBackend(_runner(), num_qubits=8)
    qc = QuantumCircuit(3, 3)
    qc.x(0)
    qc.measure(0, 2)
    qc.measure(1, 1)
    qc.measure(2, 0)

    counts = backend.run(qc, shots=64, seed=3).result().get_counts()
    assert counts == {"100": 64}


def test_mid_circuit_measurement_feedforward():
    """A conditional X, fed by a measured qubit, must actually fire."""
    backend = MimiqBackend(_runner(), num_qubits=8)
    qc = QuantumCircuit(2, 2)
    qc.x(0)
    qc.measure(0, 0)
    with qc.if_test((qc.clbits[0], 1)):
        qc.x(1)
    qc.measure(1, 1)

    counts = backend.run(qc, shots=64, seed=5).result().get_counts()
    assert counts == {"11": 64}


def test_reset_returns_qubit_to_ground():
    backend = MimiqBackend(_runner(), num_qubits=8)
    qc = QuantumCircuit(1, 1)
    qc.x(0)
    qc.reset(0)
    qc.measure(0, 0)

    assert backend.run(qc, shots=32, seed=2).result().get_counts() == {"0": 32}


def test_sampler_probabilities_match_statevector():
    """``MimiqSamplerV2`` bit arrays must follow the true distribution."""
    qc = QuantumCircuit(2, 2)
    qc.ry(2 * math.acos(math.sqrt(0.25)), 0)  # P(1) on qubit 0 = 0.75
    qc.measure([0, 1], [0, 1])

    sampler = MimiqSamplerV2(MimiqBackend(_runner(), num_qubits=8), seed=11)
    shots = 4000
    counts = sampler.run([qc], shots=shots).result()[0].data.c.get_counts()

    assert sum(counts.values()) == shots
    assert set(counts) <= {"00", "01"}
    assert abs(counts["01"] / shots - 0.75) < 0.05


def test_sampler_packs_each_register_separately():
    """Two classical registers must come back as two named bit arrays."""
    from qiskit import ClassicalRegister, QuantumRegister

    qr = QuantumRegister(2, "q")
    left = ClassicalRegister(1, "left")
    right = ClassicalRegister(1, "right")
    qc = QuantumCircuit(qr, left, right)
    qc.x(0)
    qc.measure(qr[0], left[0])
    qc.measure(qr[1], right[0])

    sampler = MimiqSamplerV2(MimiqBackend(_runner(), num_qubits=8), seed=13)
    data = sampler.run([qc], shots=32).result()[0].data

    assert data.left.get_counts() == {"1": 32}
    assert data.right.get_counts() == {"0": 32}


def test_estimator_matches_statevector_expectation():
    """The estimator is exact, so it must match the reference outright."""
    from qiskit.quantum_info import SparsePauliOp

    qc = QuantumCircuit(2)
    _generic_state(qc)
    qc.cx(0, 1)

    observable = SparsePauliOp(
        ["ZZ", "XX", "ZI", "IY"], [1.0, 0.5, -0.25, 0.75]
    )

    estimator = MimiqEstimatorV2(MimiqBackend(_runner(), num_qubits=8))
    got = estimator.run([(qc, observable)]).result()[0].data.evs
    want = np.real(Statevector(qc).expectation_value(observable))

    np.testing.assert_allclose(float(got), float(want), atol=1e-9)


def test_estimator_shots_mode_matches_the_exact_value():
    """Sampled estimation must land on the exact value within its own error.

    This is the path that emulates Qiskit's ``BackendEstimatorV2``: rotate
    into each measurement basis, measure, average the eigenvalues. It has to
    agree with the direct evaluation of the same observable, or one of the
    two conventions (label endianness, basis rotation, parity) is wrong.
    """
    from qiskit.quantum_info import SparsePauliOp

    qc = QuantumCircuit(3)
    _generic_state(qc)
    qc.cx(0, 1)
    qc.cx(1, 2)

    observable = SparsePauliOp(
        ["ZZI", "XXI", "IYZ", "III"], [1.0, 0.5, -0.75, 0.25]
    )
    backend = MimiqBackend(_runner(), num_qubits=8)

    exact = MimiqEstimatorV2(backend).run([(qc, observable)]).result()[0]
    sampled = MimiqEstimatorV2(backend, shots=20000, seed=17).run(
        [(qc, observable)]
    ).result()[0]

    error = float(sampled.data.stds)
    assert error > 0.0
    assert float(sampled.data.evs) == pytest.approx(
        float(exact.data.evs), abs=5 * error
    )
    assert sampled.metadata["shots"] == 20000


def test_estimator_averages_trajectories_to_the_ensemble_value():
    """A mid-circuit measurement leaves an ensemble, whose average is known.

    ``H`` then a measurement collapses qubit 0 to ``|0⟩`` or ``|1⟩`` with
    equal probability, and the trailing ``X`` flips it. So ``⟨Z⟩`` is -1 or
    +1 per trajectory and 0 over the ensemble, while any single trajectory
    is off by a full unit.
    """
    from qiskit.quantum_info import SparsePauliOp

    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    qc.x(0)

    backend = MimiqBackend(_runner(), num_qubits=8)
    result = MimiqEstimatorV2(backend, trajectories=4000, seed=23).run(
        [(qc, SparsePauliOp(["Z"]))]
    ).result()[0]

    error = float(result.data.stds)
    assert error == pytest.approx(1 / math.sqrt(4000), rel=0.2)
    assert float(result.data.evs) == pytest.approx(0.0, abs=5 * error)
    assert result.metadata["stochastic"] is True
    assert result.metadata["exact"] is False


def test_estimator_trajectories_and_shots_agree_on_a_noisy_circuit():
    """The two statistical methods must estimate the same quantity.

    Both target ``Tr(ρO)``. They differ only in variance, so with generous
    budgets they have to meet.
    """
    from qiskit.quantum_info import SparsePauliOp

    qc = QuantumCircuit(2, 1)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure(0, 0)
    qc.x(0)

    observable = SparsePauliOp(["ZZ", "IZ"], [1.0, 0.5])
    backend = MimiqBackend(_runner(), num_qubits=8)

    averaged = MimiqEstimatorV2(backend, trajectories=4000, seed=3).run(
        [(qc, observable)]
    ).result()[0]
    sampled = MimiqEstimatorV2(backend, shots=4000, seed=3).run(
        [(qc, observable)]
    ).result()[0]

    spread = math.hypot(float(averaged.data.stds), float(sampled.data.stds))
    assert float(averaged.data.evs) == pytest.approx(
        float(sampled.data.evs), abs=5 * spread
    )


def test_estimator_single_trajectory_is_far_from_the_average():
    """The behaviour the budget requirement exists to prevent.

    Forcing ``method="exact"`` on this circuit returns one branch, which is
    a full unit away from the ensemble value of 0. Averaging is not a
    refinement here; a single trajectory is simply a different number.
    """
    from qiskit.quantum_info import SparsePauliOp

    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    qc.x(0)

    backend = MimiqBackend(_runner(), num_qubits=8)
    with pytest.warns(UserWarning, match="single random trajectory"):
        one = MimiqEstimatorV2(backend, method="exact", seed=11).run(
            [(qc, SparsePauliOp(["Z"]))]
        ).result()[0]

    assert abs(float(one.data.evs)) == pytest.approx(1.0)
    assert one.metadata["exact"] is False


def test_estimator_reports_the_simulator_fidelity():
    """Averaging removes the statistical error, not the simulator's own."""
    from qiskit.quantum_info import SparsePauliOp

    qc = QuantumCircuit(2)
    _generic_state(qc)
    qc.cx(0, 1)

    result = MimiqEstimatorV2(MimiqBackend(_runner(), num_qubits=8)).run(
        [(qc, SparsePauliOp(["ZZ"]))]
    ).result()[0]

    assert result.metadata["min_fidelity"] == pytest.approx(1.0)


def test_estimator_pauli_label_order():
    """A Pauli label is qubit-0-right on Qiskit, qubit-0-left on MIMIQ.

    ``ZI`` reads Z on qubit 1, so on a state where only qubit 0 is
    excited it must come back as +1, not -1.
    """
    from qiskit.quantum_info import SparsePauliOp

    qc = QuantumCircuit(2)
    qc.x(0)

    estimator = MimiqEstimatorV2(MimiqBackend(_runner(), num_qubits=8))
    result = estimator.run(
        [(qc, SparsePauliOp(["ZI", "IZ"], [1.0, 1.0]))]
    ).result()

    # Z on qubit 1 (unexcited) = +1, Z on qubit 0 (excited) = -1.
    np.testing.assert_allclose(float(result[0].data.evs), 0.0, atol=1e-9)


def test_estimator_broadcasts_parameters():
    """A parametric circuit with several bindings returns one ev each."""
    from qiskit.circuit import Parameter
    from qiskit.quantum_info import SparsePauliOp

    theta = Parameter("θ")
    qc = QuantumCircuit(1)
    qc.ry(theta, 0)

    angles = [0.0, math.pi / 2, math.pi]
    estimator = MimiqEstimatorV2(MimiqBackend(_runner(), num_qubits=8))
    evs = estimator.run(
        [(qc, SparsePauliOp(["Z"]), [[a] for a in angles])]
    ).result()[0].data.evs

    np.testing.assert_allclose(evs, [math.cos(a) for a in angles], atol=1e-9)


def test_batch_of_circuits_keeps_result_order():
    backend = MimiqBackend(_runner(), num_qubits=8)
    circuits = []
    for value in (0, 1, 2):
        qc = QuantumCircuit(2, 2)
        if value & 1:
            qc.x(0)
        if value & 2:
            qc.x(1)
        qc.measure([0, 1], [0, 1])
        circuits.append(qc)

    result = backend.run(circuits, shots=16, seed=1).result()
    assert result.get_counts(0) == {"00": 16}
    assert result.get_counts(1) == {"01": 16}
    assert result.get_counts(2) == {"10": 16}


def test_transpile_against_backend_target_then_run():
    """The advertised Target must be enough for the transpiler.

    A gate outside the Target has to come out as something the converter
    accepts, which is the whole point of publishing one.
    """
    from qiskit import transpile
    from qiskit.circuit.library import EfficientSU2

    backend = MimiqBackend(_runner(), num_qubits=6)
    ansatz = EfficientSU2(3, reps=1).decompose()
    bound = ansatz.assign_parameters(
        np.linspace(0.1, 1.2, len(ansatz.parameters))
    )
    bound.measure_all()

    transpiled = transpile(bound, backend=backend, seed_transpiler=1)
    counts = backend.run(transpiled, shots=128, seed=4).result().get_counts()
    assert sum(counts.values()) == 128


def test_unsupported_run_option_names_the_backend():
    """A cloud-only option on a local backend must say so clearly."""
    backend = MimiqBackend(_runner(), num_qubits=4)
    qc = QuantumCircuit(1, 1)
    qc.measure(0, 0)

    with pytest.raises(ValueError, match="bonddim"):
        backend.run(qc, shots=4, bonddim=16).result()


def test_noise_channel_has_no_qiskit_form():
    """A MIMIQ noise channel is not unitary and must not be faked."""
    from mimiq_qiskit.converter import UnsupportedGateError

    circuit = mc.Circuit()
    circuit.push(mc.GateH(), 0)
    circuit.push(mc.Depolarizing1(0.1), 0)

    with pytest.raises(UnsupportedGateError, match="not unitary"):
        mimiq_to_qiskit(circuit)
