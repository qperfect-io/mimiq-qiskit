"""Pauli-observable helpers shared by the MIMIQ Qiskit estimators.

Qiskit hands an ``EstimatorV2`` its observables as ``{pauli_label: coefficient}``
mappings, with qubit 0 on the **right** of each label. Everything here takes
labels in that convention and reports qubit indices, so no caller has to
reverse a label by hand.

Two estimation strategies need these helpers:

- The **direct** path evaluates each Pauli term on the simulator state. It
  needs the term's support (:func:`pauli_support`) to build one
  ``ExpectationValue`` operation per term (:func:`push_pauli_terms`), and
  reads the z-register back with :func:`read_pauli_terms`.
- The **sampling** path measures the state in rotated bases. It needs the
  terms collected into simultaneously measurable sets
  (:func:`measurement_groups`), one measurement circuit per set
  (:func:`append_measurement`), per-term averages back out of the shots
  (:func:`pauli_expectations`), and a final weighted sum
  (:func:`combine_sampled_terms`).

:func:`average_trajectories` serves the direct path when the circuit is
stochastic and every trajectory returns its own value.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from qiskit.circuit import ClassicalRegister, QuantumCircuit

__all__ = [
    "MeasurementGroup",
    "append_measurement",
    "average_trajectories",
    "combine_sampled_terms",
    "measurement_groups",
    "pauli_expectations",
    "pauli_support",
    "push_pauli_terms",
    "read_pauli_terms",
    "shots_for_precision",
    "split_identity",
]

# Name of the classical register the sampling path adds. Each measurement
# group gets its own copy of the circuit, so one fixed name never collides
# with another group, only with a user register of the same name.
MEASUREMENT_REGISTER = "_est"

# A Hermitian observable has real coefficients in the Pauli basis. Round-off
# from a symbolic or numerical construction is tolerated relative to the
# coefficient's own size; anything larger is a caller error, not noise.
_IMAG_TOLERANCE = 1e-10


def pauli_support(label: str) -> list[tuple[int, str]]:
    """Return ``(qubit, pauli)`` for each non-identity factor, qubit-ascending.

    Qiskit Pauli labels are little-endian: the leftmost character acts on the
    highest-index qubit, so character ``k`` of an ``n``-character label acts on
    qubit ``n - 1 - k``.

    Args:
        label: A Qiskit Pauli label such as ``"IXZ"``. Characters must be one
            of ``I``, ``X``, ``Y``, ``Z``.

    Returns:
        The non-identity factors as ``(qubit index, pauli character)`` pairs,
        sorted by qubit index. Empty for an all-identity label.

    Example::

        >>> pauli_support("IXZ")
        [(0, 'Z'), (1, 'X')]
    """
    n = len(label)
    return sorted(
        (n - 1 - k, ch) for k, ch in enumerate(label) if ch != "I"
    )


def split_identity(
    observable: Mapping[str, complex],
) -> tuple[float, dict[str, float]]:
    """Separate an observable's identity term from its measurable ones.

    The identity term contributes its coefficient to the expectation value
    outright, with no evolution and no error, so it is kept apart from the
    terms that have to be evaluated.

    Args:
        observable: ``{pauli_label: coefficient}``, as produced by
            :meth:`~qiskit.primitives.containers.ObservablesArray.coerce`.

    Returns:
        ``(identity_coefficient, terms)`` where ``terms`` maps each
        non-identity label to its real coefficient. Repeated labels are
        summed.

    Raises:
        ValueError: If a coefficient has an imaginary part too large to be
            round-off. ``EstimatorV2`` observables must be Hermitian, which in
            the Pauli basis means real coefficients, and silently discarding
            an imaginary part would hide the mistake.
    """
    identity = 0.0
    terms: dict[str, float] = {}
    for label, coeff in observable.items():
        value = complex(coeff)
        if abs(value.imag) > _IMAG_TOLERANCE * max(1.0, abs(value.real)):
            raise ValueError(
                f"observable term {label!r} has a complex coefficient "
                f"({value}); an estimator observable must be Hermitian, so "
                "its Pauli coefficients must be real"
            )
        if not pauli_support(label):
            identity += value.real
        else:
            terms[label] = terms.get(label, 0.0) + value.real
    return identity, terms


@dataclass(frozen=True)
class MeasurementGroup:
    """A set of Pauli terms that one measurement can serve.

    The members commute qubit-wise: on every qubit that two of them both act
    on, they act with the same Pauli. So rotating each qubit in ``basis`` into
    the Z basis and measuring it yields, in one pass, the eigenvalue of every
    member.

    Attributes:
        labels: The Qiskit Pauli labels in the group, in a deterministic
            order.
        basis: ``{qubit: pauli}`` over every qubit any member acts on.
    """

    labels: tuple[str, ...]
    basis: dict[int, str]

    @property
    def qubits(self) -> list[int]:
        """The measured qubits, ascending. Indexes the samples' bit order."""
        return sorted(self.basis)


def measurement_groups(labels: Iterable[str]) -> list[MeasurementGroup]:
    """Collect Pauli labels into qubit-wise commuting measurement groups.

    Grouping is what keeps the circuit count down: an observable with 50 terms
    over a shared basis costs one measurement circuit, not 50. The partition
    comes from Qiskit's own
    :meth:`~qiskit.quantum_info.PauliList.group_commuting`, so it matches what
    ``BackendEstimatorV2`` would do with the same observable.

    Args:
        labels: Non-identity Qiskit Pauli labels, all of the same width.

    Returns:
        One :class:`MeasurementGroup` per basis, ordered deterministically.
        Empty when ``labels`` is empty.

    Raises:
        ValueError: If a label is all-identity. Those carry no measurement;
            take them out with :func:`split_identity` first.
    """
    from qiskit.quantum_info import PauliList

    unique = sorted(set(labels))
    if not unique:
        return []
    for label in unique:
        if not pauli_support(label):
            raise ValueError(
                f"{label!r} is the identity and cannot be measured; separate "
                "it out with split_identity() first"
            )

    groups = []
    for commuting in PauliList(unique).group_commuting(qubit_wise=True):
        members = tuple(commuting.to_labels())
        basis: dict[int, str] = {}
        for label in members:
            basis.update(pauli_support(label))
        groups.append(MeasurementGroup(members, basis))
    return groups


def append_measurement(
    circuit: QuantumCircuit, group: MeasurementGroup
) -> tuple[QuantumCircuit, list[int]]:
    """Copy ``circuit`` and measure ``group``'s basis on the end of it.

    Each qubit in the group's basis is rotated into the Z basis (``H`` for an
    ``X`` factor, ``Sdg`` then ``H`` for a ``Y`` one, nothing for a ``Z``) and
    measured into a register added for the purpose. The rotations are the same
    ones Qiskit's ``BackendEstimatorV2`` uses.

    Args:
        circuit: The state-preparation circuit. Not modified.
        group: The terms to measure together.

    Returns:
        ``(measurement_circuit, clbits)``, where ``clbits`` holds the global
        classical-bit index of each measured qubit in ``group.qubits`` order.
        That is the order :func:`pauli_expectations` expects its samples in.

    Raises:
        ValueError: If ``circuit`` already has a register named ``_est``, so
            adding ours would be ambiguous. Rename yours.
    """
    if any(creg.name == MEASUREMENT_REGISTER for creg in circuit.cregs):
        raise ValueError(
            f"the circuit already has a classical register named "
            f"{MEASUREMENT_REGISTER!r}, which the estimator needs for its own "
            "measurements; rename it"
        )

    qubits = group.qubits
    out = circuit.copy()
    creg = ClassicalRegister(len(qubits), MEASUREMENT_REGISTER)
    out.add_register(creg)
    for slot, qubit in enumerate(qubits):
        pauli = group.basis[qubit]
        if pauli == "X":
            out.h(qubit)
        elif pauli == "Y":
            out.sdg(qubit)
            out.h(qubit)
        out.measure(qubit, creg[slot])

    return out, [out.find_bit(bit).index for bit in creg]


def pauli_expectations(
    group: MeasurementGroup, samples: Iterable[Sequence[int]]
) -> dict[str, tuple[float, float]]:
    """Average every term in ``group`` over one measurement's shots.

    A term's eigenvalue on a shot is ``(-1) ** parity`` of the measured bits on
    its support, the rotations having already turned each factor into a Z.

    Args:
        group: The group that was measured.
        samples: One sequence of bits per shot, each ordered like
            ``group.qubits``.

    Returns:
        ``{label: (expectation, variance)}``. The variance is the single-shot
        ``1 - ⟨P⟩²``; divide it by the shot count to get the squared standard
        error of the mean.

    Raises:
        ValueError: If ``samples`` is empty, since nothing can be averaged.
    """
    slots = {qubit: slot for slot, qubit in enumerate(group.qubits)}
    supports = {
        label: [slots[qubit] for qubit, _ in pauli_support(label)]
        for label in group.labels
    }
    totals = {label: 0 for label in group.labels}

    shots = 0
    for bits in samples:
        shots += 1
        for label, support in supports.items():
            parity = 0
            for slot in support:
                parity ^= bits[slot] & 1
            totals[label] += 1 - 2 * parity

    if shots == 0:
        raise ValueError("no samples to average")

    out = {}
    for label, total in totals.items():
        expectation = total / shots
        out[label] = (expectation, 1.0 - expectation**2)
    return out


def combine_sampled_terms(
    identity: float,
    terms: Mapping[str, float],
    expectations: Mapping[str, tuple[float, float]],
    shots: int,
) -> tuple[float, float]:
    """Weight sampled term averages into one expectation value and its error.

    Args:
        identity: The identity term's coefficient, from
            :func:`split_identity`.
        terms: ``{label: coefficient}`` for the measured terms.
        expectations: ``{label: (expectation, variance)}``, from
            :func:`pauli_expectations`.
        shots: Shots behind each term average.

    Returns:
        ``(expectation_value, standard_error)``. The error follows Qiskit's
        ``BackendEstimatorV2`` convention, ``Σ |cᵢ| √Var(Pᵢ) / √N``, which
        adds the per-term errors as if the terms were perfectly correlated and
        so is an upper bound on the true standard error.
    """
    value = identity
    error = 0.0
    for label, coeff in terms.items():
        expectation, variance = expectations[label]
        value += coeff * expectation
        error += abs(coeff) * math.sqrt(max(variance, 0.0))
    return value, error / math.sqrt(shots)


def push_pauli_terms(
    circuit, labels: Iterable[str], *, firstzvar: int = 0
) -> dict[str, int]:
    """Push one ``ExpectationValue`` operation per Pauli label onto a circuit.

    Each term is evaluated on the state itself rather than sampled, so it is
    exact at any weight, and only the term's own qubits are named: a weight-2
    term on a 500-qubit register stays a two-qubit operation.

    Labels are pushed once each, so several observables sharing a term at the
    same parameter binding can share its z-variable and its evaluation.

    Args:
        circuit: A ``mimiqcircuits.Circuit``, appended to in place.
        labels: Non-identity Qiskit Pauli labels, in the order they should
            take z-variables. Repeats are ignored.
        firstzvar: Index of the first z-variable to write into.

    Returns:
        ``{label: zvar}``, the z-variable each term's value will land in. Pass
        it to :func:`read_pauli_terms` along with the z-register that comes
        back.
    """
    import mimiqcircuits as mc

    zvar_of: dict[str, int] = {}
    for label in labels:
        if label in zvar_of:
            continue
        support = pauli_support(label)
        zvar = firstzvar + len(zvar_of)
        circuit.push(
            mc.ExpectationValue(mc.PauliString("".join(p for _, p in support))),
            *(qubit for qubit, _ in support),
            zvar,
        )
        zvar_of[label] = zvar
    return zvar_of


def read_pauli_terms(
    zstate: Sequence[complex],
    zvar_of: Mapping[str, int],
    identity: float,
    terms: Mapping[str, float],
) -> float:
    """Combine one z-register into an observable's expectation value.

    Args:
        zstate: One trajectory's z-register, as returned in
            ``QCSResults.zstates``.
        zvar_of: ``{label: zvar}``, from :func:`push_pauli_terms`.
        identity: The identity term's coefficient, from
            :func:`split_identity`.
        terms: ``{label: coefficient}`` for this observable's measured terms.
            Every label must appear in ``zvar_of``.

    Returns:
        ``identity + Σ cᵢ Re⟨Pᵢ⟩``. Each ``⟨Pᵢ⟩`` is real for a Pauli string
        on a normalised state, so taking the real part discards only
        round-off.
    """
    value = identity
    for label, coeff in terms.items():
        value += coeff * complex(zstate[zvar_of[label]]).real
    return value


def average_trajectories(values: Sequence[float]) -> tuple[float, float]:
    """Mean and standard error of one expectation value per trajectory.

    The trajectory average is an unbiased estimator of the density-matrix
    expectation value: a branch sampled with probability :math:`\\|K_k\\psi\\|^2`
    and renormalised gives
    :math:`\\mathbb{E}[\\langle\\psi_k|O|\\psi_k\\rangle]
    = \\sum_k \\langle\\psi|K_k^\\dagger O K_k|\\psi\\rangle
    = \\mathrm{Tr}(\\rho' O)`, and the same holds for measurement branches. So
    averaging is correct, and reading a single trajectory is one draw from a
    distribution whose spread can be as wide as the observable's own range.

    Args:
        values: One value per trajectory.

    Returns:
        ``(mean, standard_error)``. A single trajectory reports an error of
        zero, there being nothing to estimate a spread from; that zero means
        "unknown", not "exact", so check the run's metadata before trusting
        it.

    Raises:
        ValueError: If ``values`` is empty.
    """
    count = len(values)
    if count == 0:
        raise ValueError("no trajectories to average")
    mean = math.fsum(values) / count
    if count == 1:
        return mean, 0.0
    variance = math.fsum((v - mean) ** 2 for v in values) / (count - 1)
    return mean, math.sqrt(variance / count)


def shots_for_precision(precision: float) -> int:
    """Shots needed for a target precision, ``ceil(1 / precision²)``.

    Qiskit's convention, so a pub's ``precision`` sizes a MIMIQ run the same
    way it would size a hardware one.

    Args:
        precision: Target standard error. Must be positive.

    Returns:
        The shot count, at least 1.

    Raises:
        ValueError: If ``precision`` is not positive.
    """
    if precision <= 0:
        raise ValueError(f"precision must be positive, got {precision}")
    return max(1, math.ceil(1.0 / precision**2))
