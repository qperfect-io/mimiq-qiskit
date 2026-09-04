"""The estimation engine behind the MIMIQ Qiskit estimators.

:class:`MimiqEstimatorV2` runs this against a :class:`~mimiq_qiskit.MimiqBackend`,
and provider-specific estimators (such as TensorWeaver's) run it against their
own simulator, so that every one of them resolves methods, averages
trajectories, and reports metadata the same way. Feed
:func:`estimate_pub` an :class:`~qiskit.primitives.containers.EstimatorPub`,
an :class:`EstimatorConfig`, and a ``run`` callable that submits MIMIQ
circuits.

Three ways to get an expectation value
--------------------------------------

``exact``
    One evolution, and each Pauli term read straight off the state. No
    statistical error at all: the only error is the simulator's own (for an
    MPS engine, the truncation). Only meaningful for a circuit that ends in a
    single definite state.

``trajectories``
    A circuit carrying a mid-circuit measurement, a reset, a noise channel, or
    qubit loss does not end in one state; it ends in an ensemble. The engine
    re-evolves it once per shot, and each trajectory reports its own
    expectation value. Averaging them estimates the density-matrix value
    :math:`\\mathrm{Tr}(\\rho O)` without bias, so that is what this does.
    Reading one trajectory instead, as a single-shot run must, returns a
    random draw whose spread can cover the observable's whole range.

``shots``
    What hardware and Qiskit's own ``BackendEstimatorV2`` do: rotate into each
    measurement basis, measure, and average the ±1 eigenvalues. Correct for
    every circuit, and the way to compare against a shot-based reference, but
    for the same budget it is strictly noisier than averaging trajectories,
    since it samples the observable on top of sampling the ensemble.

The default, ``"auto"``, evaluates exactly when the circuit is deterministic
and averages trajectories when it is not. It refuses to guess a budget: a
stochastic circuit with no ``trajectories``, ``shots``, or ``precision`` set
raises rather than returning one trajectory dressed up as an exact number.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from qiskit.primitives.containers import DataBin, PubResult
from qiskit.primitives.containers.estimator_pub import EstimatorPub

from mimiq_qiskit.converter import qiskit_to_mimiq
from mimiq_qiskit.observables import (
    append_measurement,
    average_trajectories,
    combine_sampled_terms,
    measurement_groups,
    pauli_expectations,
    push_pauli_terms,
    read_pauli_terms,
    shots_for_precision,
    split_identity,
)

__all__ = ["EstimatorConfig", "estimate_pub"]

METHODS = ("auto", "exact", "trajectories", "shots")


@dataclass(frozen=True)
class EstimatorConfig:
    """How an estimator should turn circuits into expectation values.

    Args:
        method: ``"auto"`` (the default) evaluates exactly when the circuit is
            deterministic and averages trajectories when it is not.
            ``"exact"``, ``"trajectories"``, and ``"shots"`` force one of the
            three methods described in this module.
        trajectories: Trajectories to average. ``None`` sizes it from the
            pub's ``precision``.
        shots: Shots per measurement basis for the sampled method. ``None``
            sizes it from the pub's ``precision``. Giving this selects
            ``"shots"`` under ``method="auto"``.
        emulate_shot_noise: Add Gaussian noise of width ``precision`` to a
            value that came out exact, the way Qiskit's
            ``StatevectorEstimator`` does. A cheap stand-in for shot noise
            that costs one evolution instead of a shot budget. Ignored when
            the value carries real statistical error already.
        default_precision: Precision for pubs that do not carry their own.
        seed: Seeds the ``emulate_shot_noise`` generator. The simulator's own
            seed belongs to the ``run`` callable.
        assume_stochastic: Treat every circuit as stochastic even when its
            instructions look deterministic. Set this when the simulator adds
            noise the circuit does not show, such as a server-side
            ``noisemodel`` run option.

    Raises:
        ValueError: If ``method`` is unknown, a count is not positive, or both
            ``trajectories`` and ``shots`` are set, which would ask for two
            methods at once.
    """

    method: str = "auto"
    trajectories: int | None = None
    shots: int | None = None
    emulate_shot_noise: bool = False
    default_precision: float = 0.0
    seed: int | None = None
    assume_stochastic: bool = False

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(
                f"method must be one of {', '.join(METHODS)}; "
                f"got {self.method!r}"
            )
        if self.trajectories is not None and self.shots is not None:
            raise ValueError(
                "set trajectories= to average exact per-trajectory values, or "
                "shots= to estimate from measurements, not both"
            )
        for name in ("trajectories", "shots"):
            count = getattr(self, name)
            if count is not None and count < 1:
                raise ValueError(f"{name} must be at least 1, got {count}")
        if self.method == "shots" and self.trajectories is not None:
            raise ValueError(
                "method='shots' does not average trajectories; use shots= to "
                "set its budget"
            )
        if self.method in ("exact", "trajectories") and self.shots is not None:
            raise ValueError(
                f"method={self.method!r} does not sample; use trajectories= "
                "to set its budget"
            )


def estimate_pub(
    pub: EstimatorPub,
    config: EstimatorConfig,
    run: Callable[[list, int], Sequence],
) -> PubResult:
    """Estimate every observable in ``pub`` and package the result.

    Args:
        pub: The pub to estimate. Its observables and parameter bindings
            broadcast against each other under Qiskit's rules, and the result
            arrays take the broadcast shape.
        config: Which method to use and what to spend on it.
        run: ``run(circuits, nsamples) -> list[QCSResults]``, submitting a
            list of MIMIQ circuits and returning one result per circuit in the
            same order. Everything backend-specific (the connection, the
            simulator options, the seed) lives behind this callable.

    Returns:
        A :class:`~qiskit.primitives.containers.PubResult` whose ``evs`` and
        ``stds`` take the pub's shape. Its metadata reports ``method``,
        ``exact``, ``stochastic``, the budget actually spent, the
        ``target_precision``, and ``min_fidelity`` where the backend reports
        one.

    Raises:
        ValueError: If the circuit is stochastic and no budget was given (see
            this module's docstring), or if the backend returns a result the
            requested method cannot be read out of.
    """
    precision = pub.precision if pub.precision is not None else 0.0
    if not precision:
        precision = config.default_precision

    bindings, bc_param, bc_obs = _broadcast(pub)
    stochastic = config.assume_stochastic or any(
        _needs_trajectories(qiskit_to_mimiq(circuit))
        for circuit in bindings.values()
    )
    method, budget = _resolve(config, precision, stochastic)

    if method == "shots":
        evs, stds, fidelity, spent = _estimate_sampled(
            bindings, bc_param, bc_obs, budget, run
        )
    else:
        evs, stds, fidelity, spent = _estimate_direct(
            bindings, bc_param, bc_obs, budget, run
        )

    exact = method != "shots" and not stochastic
    if exact and config.emulate_shot_noise and precision > 0:
        rng = np.random.default_rng(config.seed)
        # `normal` returns a scalar for a 0-d input, so put the shape back.
        evs = np.asarray(rng.normal(evs, precision), dtype=float).reshape(
            evs.shape
        )
        stds = np.full(evs.shape, float(precision))
        exact = False

    metadata = {
        "target_precision": precision,
        "method": method,
        "exact": exact,
        "stochastic": stochastic,
    }
    metadata["shots" if method == "shots" else "trajectories"] = spent
    if fidelity is not None:
        metadata["min_fidelity"] = fidelity

    return PubResult(
        DataBin(evs=evs, stds=stds, shape=evs.shape), metadata=metadata
    )


# ── method resolution ────────────────────────────────────────────────────


def _resolve(
    config: EstimatorConfig, precision: float, stochastic: bool
) -> tuple[str, int]:
    """Pick the method and its budget. See this module's docstring."""
    method = config.method
    if method == "auto":
        if config.shots is not None:
            method = "shots"
        elif config.trajectories is not None or stochastic:
            method = "trajectories"
        else:
            method = "exact"

    if method == "exact":
        if stochastic:
            warnings.warn(
                "method='exact' on a circuit that needs one evolution per "
                "shot returns a single random trajectory, not the ensemble "
                "average; its expectation value is only exact for that one "
                "branch",
                UserWarning,
                stacklevel=4,
            )
        return "exact", 1

    if method == "shots":
        if config.shots is not None:
            return "shots", config.shots
        if precision > 0:
            return "shots", shots_for_precision(precision)
        raise ValueError(
            "method='shots' needs a budget: set shots=N, or pass a positive "
            "precision to size it as ceil(1/precision**2)"
        )

    if config.trajectories is not None:
        # A deterministic circuit has one answer, so re-evolving it would buy
        # nothing but time.
        return "trajectories", config.trajectories if stochastic else 1
    if not stochastic:
        return "trajectories", 1
    if precision > 0:
        return "trajectories", shots_for_precision(precision)
    raise ValueError(
        "this circuit needs one evolution per shot (a mid-circuit "
        "measurement, a reset, a noise channel, or qubit loss), so its "
        "expectation value is an average over the ensemble it produces, not "
        "an exact number. Set trajectories=N to average N of them, shots=N "
        "to estimate from measurements the way hardware does, or pass a "
        "positive precision to size either automatically. method='exact' "
        "reads a single trajectory, which is unbiased but as noisy as the "
        "observable's own range."
    )


def _needs_trajectories(circuit) -> bool:
    """Whether MIMIQ has to re-evolve ``circuit`` once per shot."""
    try:
        from mimiqcircuits.backends.measure_analysis import needs_trajectories
    except ImportError:  # pragma: no cover - guards a mimiqcircuits reshuffle
        # The same predicate: an operation that touches qubits and is not
        # unitary cannot be applied once and then sampled.
        return any(
            inst.operation.num_qubits != 0 and not inst.operation.isunitary()
            for inst in circuit.instructions
        )
    return needs_trajectories(circuit)


# ── broadcasting ─────────────────────────────────────────────────────────


def _broadcast(pub: EstimatorPub):
    """Line the pub's parameter bindings up against its observables.

    Returns ``(bindings, bc_param, bc_obs)``: the bound circuits keyed by
    parameter index, and the parameter indices and observables broadcast to
    the pub's shape. Circuits are bound once per parameter index, not once per
    output element, so observables sharing a binding share one circuit.
    """
    parameter_values = pub.parameter_values
    param_shape = parameter_values.shape
    param_indices = np.fromiter(
        np.ndindex(param_shape), dtype=object
    ).reshape(param_shape)

    observables = np.empty(pub.observables.shape, dtype=object)
    for index in np.ndindex(observables.shape):
        observables[index] = pub.observables[index]

    bc_param, bc_obs = np.broadcast_arrays(param_indices, observables)

    bindings = {}
    for index in np.ndindex(*bc_param.shape):
        param_index = bc_param[index]
        if param_index not in bindings:
            circuit = parameter_values.bind(pub.circuit, param_index)
            # An estimator pub carries state preparation, not readout. Qiskit
            # says it should hold no measurements at all; drop any trailing
            # ones rather than letting them collapse the state we are about to
            # measure. Mid-circuit measurements are part of the state
            # preparation and stay.
            bindings[param_index] = circuit.remove_final_measurements(
                inplace=False
            )
    return bindings, bc_param, bc_obs


def _by_binding(bc_param, bc_obs):
    """Group output indices by parameter index, with each one's observable."""
    grouped: dict[tuple, list] = {}
    for index in np.ndindex(*bc_param.shape):
        identity, terms = split_identity(bc_obs[index])
        grouped.setdefault(bc_param[index], []).append(
            (index, identity, terms)
        )
    return grouped


def _min_fidelity(results: Sequence) -> float | None:
    """Lowest fidelity any submission reported, or ``None`` if none did.

    On an MPS engine this is the truncation fidelity, the error that survives
    when the statistical one is averaged away.
    """
    seen = [
        float(f)
        for result in results
        for f in (getattr(result, "fidelities", None) or ())
    ]
    return min(seen) if seen else None


# ── direct evaluation ────────────────────────────────────────────────────


def _estimate_direct(bindings, bc_param, bc_obs, trajectories, run):
    """Read each Pauli term off the state, averaging over trajectories."""
    circuits: list = []
    plans: list = []
    for param_index, entries in _by_binding(bc_param, bc_obs).items():
        labels = list(
            dict.fromkeys(
                label for _, _, terms in entries for label in terms
            )
        )
        slot = None
        zvar_of: dict[str, int] = {}
        if labels:
            circuit = qiskit_to_mimiq(bindings[param_index])
            zvar_of = push_pauli_terms(circuit, labels)
            slot = len(circuits)
            circuits.append(circuit)
        for index, identity, terms in entries:
            plans.append((index, identity, terms, zvar_of, slot))

    results = run(circuits, trajectories) if circuits else []
    if len(results) != len(circuits):
        raise ValueError(
            f"the backend returned {len(results)} results for "
            f"{len(circuits)} circuits"
        )

    evs = np.zeros(bc_param.shape, dtype=float)
    stds = np.zeros(bc_param.shape, dtype=float)
    spent = 1
    for index, identity, terms, zvar_of, slot in plans:
        if slot is None:
            # An identity-only observable needs no evolution at all.
            evs[index] = identity
            continue
        zstates = getattr(results[slot], "zstates", None)
        if not zstates:
            raise ValueError(
                "the backend returned no z-register, so the expectation "
                "values it was asked for cannot be read back"
            )
        # One z-register per trajectory, or exactly one when the circuit is
        # deterministic. Averaging covers both.
        values = [
            read_pauli_terms(zstate, zvar_of, identity, terms)
            for zstate in zstates
        ]
        spent = max(spent, len(values))
        evs[index], stds[index] = average_trajectories(values)

    return evs, stds, _min_fidelity(results), spent


# ── sampled estimation ───────────────────────────────────────────────────


def _estimate_sampled(bindings, bc_param, bc_obs, shots, run):
    """Estimate from measurements in rotated bases, as hardware does."""
    circuits: list = []
    submissions: list = []
    grouped = _by_binding(bc_param, bc_obs)
    for param_index, entries in grouped.items():
        labels = {label for _, _, terms in entries for label in terms}
        for group in measurement_groups(labels):
            circuit, clbits = append_measurement(
                bindings[param_index], group
            )
            submissions.append((param_index, group, clbits))
            circuits.append(qiskit_to_mimiq(circuit))

    results = run(circuits, shots) if circuits else []
    if len(results) != len(circuits):
        raise ValueError(
            f"the backend returned {len(results)} results for "
            f"{len(circuits)} circuits"
        )

    expectations: dict[tuple, tuple[float, float]] = {}
    for (param_index, group, clbits), result in zip(submissions, results):
        cstates = getattr(result, "cstates", None) or ()
        if len(cstates) < shots:
            raise ValueError(
                f"the backend returned {len(cstates)} samples but {shots} "
                "shots were requested, so the estimate would rest on a "
                "smaller budget than its error bar claims"
            )
        samples = [
            [
                1 if bit < len(cstate) and cstate[bit] else 0
                for bit in clbits
            ]
            for cstate in cstates[:shots]
        ]
        for label, value in pauli_expectations(group, samples).items():
            expectations[param_index, label] = value

    evs = np.zeros(bc_param.shape, dtype=float)
    stds = np.zeros(bc_param.shape, dtype=float)
    for param_index, entries in grouped.items():
        for index, identity, terms in entries:
            evs[index], stds[index] = combine_sampled_terms(
                identity,
                terms,
                {
                    label: expectations[param_index, label]
                    for label in terms
                },
                shots,
            )

    return evs, stds, _min_fidelity(results), shots
