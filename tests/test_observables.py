"""Tests for the Pauli-observable helpers the estimators share.

These are the pieces where a convention mistake is invisible in the result:
Qiskit writes qubit 0 on the right of a Pauli label and MIMIQ writes it on
the left, a measured Y needs its two rotations in the right order, and a
grouped measurement only serves its members if they really do commute
qubit-wise. Each of those is checked here against an independent
computation rather than against another part of the same code.
"""

from __future__ import annotations

import math

import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

from mimiq_qiskit.observables import (
    MeasurementGroup,
    append_measurement,
    average_trajectories,
    combine_sampled_terms,
    measurement_groups,
    pauli_expectations,
    pauli_support,
    shots_for_precision,
    split_identity,
)


# ── support and endianness ───────────────────────────────────────────────


def test_pauli_support_is_little_endian_and_sorted():
    # "IXZ" is Z on qubit 0, X on qubit 1, identity on qubit 2.
    assert pauli_support("IXZ") == [(0, "Z"), (1, "X")]
    assert pauli_support("III") == []
    assert pauli_support("Y") == [(0, "Y")]


def test_pauli_support_agrees_with_qiskit_operator():
    """The support must name the qubits Qiskit's own operator acts on."""
    label = "IYIX"
    support = dict(pauli_support(label))
    # Build the same operator from single-qubit factors placed by index.
    qc = QuantumCircuit(len(label))
    for qubit, pauli in support.items():
        getattr(qc, pauli.lower())(qubit)
    assert Operator(qc) == Operator.from_label(label)


# ── identity split ───────────────────────────────────────────────────────


def test_split_identity_separates_and_sums():
    identity, terms = split_identity({"II": 2.0, "ZI": 1.5, "IZ": -0.5})
    assert identity == 2.0
    assert terms == {"ZI": 1.5, "IZ": -0.5}


def test_split_identity_tolerates_roundoff_but_rejects_complex():
    identity, terms = split_identity({"ZZ": 1.0 + 1e-16j})
    assert identity == 0.0
    assert terms == {"ZZ": pytest.approx(1.0)}

    with pytest.raises(ValueError, match="Hermitian"):
        split_identity({"ZZ": 1.0 + 0.5j})


# ── grouping ─────────────────────────────────────────────────────────────


def test_measurement_groups_commute_qubit_wise():
    groups = measurement_groups(["ZZ", "XX", "IZ", "XI", "ZI"])
    assert sum(len(g.labels) for g in groups) == 5
    for group in groups:
        # Every member must agree with the group basis wherever it acts, or
        # one measurement could not serve them all.
        for label in group.labels:
            for qubit, pauli in pauli_support(label):
                assert group.basis[qubit] == pauli


def test_measurement_groups_share_a_basis():
    """Terms over one basis cost one measurement, not one each."""
    groups = measurement_groups(["ZII", "IZI", "IIZ", "ZZZ"])
    assert len(groups) == 1
    assert groups[0].basis == {0: "Z", 1: "Z", 2: "Z"}
    assert groups[0].qubits == [0, 1, 2]


def test_measurement_groups_rejects_the_identity():
    with pytest.raises(ValueError, match="split_identity"):
        measurement_groups(["II"])


def test_measurement_groups_is_empty_for_no_terms():
    assert measurement_groups([]) == []


# ── measurement circuits ─────────────────────────────────────────────────


@pytest.mark.parametrize("pauli", ["X", "Y", "Z"])
def test_append_measurement_rotates_into_the_z_basis(pauli):
    """The rotation must map the Pauli's +1 eigenstate onto ``|0⟩``.

    That is the whole content of the basis change: after it, reading a 0
    means the +1 eigenvalue.
    """
    prep = QuantumCircuit(1)
    # Prepare the +1 eigenstate of the Pauli under test.
    if pauli == "X":
        prep.h(0)
    elif pauli == "Y":
        prep.h(0)
        prep.s(0)

    group = MeasurementGroup((pauli,), {0: pauli})
    measured, clbits = append_measurement(prep, group)
    assert clbits == [0]

    rotated = measured.remove_final_measurements(inplace=False)
    probabilities = Statevector(rotated).probabilities()
    assert probabilities[0] == pytest.approx(1.0, abs=1e-12)


def test_append_measurement_leaves_the_input_alone():
    prep = QuantumCircuit(2)
    prep.h(0)
    group = MeasurementGroup(("XI",), {1: "X"})
    append_measurement(prep, group)
    assert prep.num_clbits == 0
    assert len(prep.data) == 1


def test_append_measurement_maps_qubits_to_clbits_in_order():
    prep = QuantumCircuit(4)
    group = MeasurementGroup(("ZIZI",), {1: "Z", 3: "Z"})
    measured, clbits = append_measurement(prep, group)
    # Two measured qubits, so two classical bits, in ascending qubit order.
    assert clbits == [0, 1]
    measured_pairs = [
        (measured.find_bit(inst.qubits[0]).index,
         measured.find_bit(inst.clbits[0]).index)
        for inst in measured.data
        if inst.operation.name == "measure"
    ]
    assert measured_pairs == [(1, 0), (3, 1)]


def test_append_measurement_refuses_to_shadow_a_register():
    from qiskit.circuit import ClassicalRegister

    prep = QuantumCircuit(1)
    prep.add_register(ClassicalRegister(1, "_est"))
    with pytest.raises(ValueError, match="_est"):
        append_measurement(prep, MeasurementGroup(("Z",), {0: "Z"}))


# ── reading samples back ─────────────────────────────────────────────────


def test_pauli_expectations_counts_parity():
    group = MeasurementGroup(("ZZ", "IZ"), {0: "Z", 1: "Z"})
    # Bits are ordered like group.qubits == [0, 1].
    samples = [[0, 0], [1, 1], [1, 0], [0, 1]]
    got = pauli_expectations(group, samples)
    # "ZZ" has even parity on the first two shots, odd on the last two.
    assert got["ZZ"][0] == pytest.approx(0.0)
    # "IZ" is Z on qubit 0 alone: bits 0, 1, 1, 0.
    assert got["IZ"][0] == pytest.approx(0.0)

    # A term that always reads +1 has no variance left.
    got = pauli_expectations(group, [[0, 0]] * 8)
    assert got["ZZ"] == (pytest.approx(1.0), pytest.approx(0.0))


def test_pauli_expectations_needs_samples():
    with pytest.raises(ValueError, match="no samples"):
        pauli_expectations(MeasurementGroup(("Z",), {0: "Z"}), [])


def test_combine_sampled_terms_weights_and_scales_the_error():
    identity = 0.5
    terms = {"ZI": 2.0, "IZ": -1.0}
    expectations = {"ZI": (0.5, 0.75), "IZ": (-0.25, 0.9375)}
    value, error = combine_sampled_terms(identity, terms, expectations, 100)

    assert value == pytest.approx(0.5 + 2.0 * 0.5 + (-1.0) * (-0.25))
    expected = (2.0 * math.sqrt(0.75) + 1.0 * math.sqrt(0.9375)) / 10.0
    assert error == pytest.approx(expected)


# ── trajectory averaging ─────────────────────────────────────────────────


def test_average_trajectories_matches_the_sample_statistics():
    values = [1.0, -1.0, 1.0, 1.0]
    mean, error = average_trajectories(values)
    assert mean == pytest.approx(0.5)
    # Sample standard deviation with ddof=1 is 1.0, over sqrt(4).
    assert error == pytest.approx(0.5)


def test_average_trajectories_reports_no_error_for_one_value():
    """One trajectory leaves nothing to estimate a spread from."""
    assert average_trajectories([0.25]) == (0.25, 0.0)


def test_average_trajectories_needs_a_value():
    with pytest.raises(ValueError, match="no trajectories"):
        average_trajectories([])


# ── precision ────────────────────────────────────────────────────────────


def test_shots_for_precision_follows_qiskits_convention():
    assert shots_for_precision(0.5) == 4
    assert shots_for_precision(0.015625) == 4096
    assert shots_for_precision(0.03) == 1112  # ceil(1111.1)
    with pytest.raises(ValueError):
        shots_for_precision(0.0)
