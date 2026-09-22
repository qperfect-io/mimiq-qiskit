"""Tests for the native MIMIQ Sampler/Estimator V2 primitives.

These use stub runners that fabricate ``QCSResults`` so the tests cover
the primitive plumbing (bitstring packing, register/observable
broadcasting, result shapes) without a MIMIQ server, including the shapes
a real backend would never produce.

``tests/test_simulation.py`` covers the same primitives against a real
simulator, where sampled distributions and expectation values are checked
against Qiskit's own reference.
"""

from __future__ import annotations

from inspect import signature

import pytest
from bitarray import bitarray
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp

from mimiqcircuits import QCSResults

from qiskit.primitives.containers.estimator_pub import EstimatorPub

from mimiq_qiskit import MimiqBackend, MimiqEstimatorV2, MimiqSamplerV2
from mimiq_qiskit.estimation import EstimatorConfig, TermValues, estimate_pub


class _SamplerStub:
    """Every shot reports clbit 0 high, the rest low (MIMIQ bit order)."""

    def __call__(self, circuit, *, nsamples, seed):
        nb = max(circuit.num_bits(), 1)
        cstates = [bitarray("1" + "0" * (nb - 1)) for _ in range(nsamples)]
        return QCSResults(
            simulator="stub", version="0", cstates=cstates, timings={}
        )


class _EstimatorStub:
    """Answers z[i] = i + 1 and records every circuit it was given.

    Distinct values per z-variable make the mapping from observable to
    z-variable observable, and the recorded circuits let a test inspect the
    ``ExpectationValue`` operations the estimator pushed.
    """

    def __init__(self, trajectories: int = 1, fidelity: float = 1.0):
        self.circuits = []
        self._trajectories = trajectories
        self._fidelity = fidelity

    def __call__(self, circuit, *, nsamples, seed):
        self.circuits.append(circuit)
        nz = circuit.num_zvars()
        count = min(nsamples, self._trajectories)
        return QCSResults(
            simulator="stub",
            version="0",
            cstates=[],
            zstates=[
                [complex(i + 1) for i in range(nz)] for _ in range(count)
            ],
            fidelities=[self._fidelity] * max(count, 1),
            timings={},
        )


class _TrajectoryStub:
    """Answers z[0] = each of ``values`` in turn, one trajectory each."""

    def __init__(self, values):
        self._values = list(values)

    def __call__(self, circuit, *, nsamples, seed):
        return QCSResults(
            simulator="stub",
            version="0",
            cstates=[],
            zstates=[[complex(v)] for v in self._values[:nsamples]],
            fidelities=[1.0] * len(self._values[:nsamples]),
            timings={},
        )


class _ShotStub:
    """Reports a fixed bit pattern on every measured clbit, every shot."""

    def __init__(self, bits: str):
        self._bits = bits

    def __call__(self, circuit, *, nsamples, seed):
        nb = max(circuit.num_bits(), len(self._bits))
        pattern = self._bits.ljust(nb, "0")
        return QCSResults(
            simulator="stub",
            version="0",
            cstates=[bitarray(pattern) for _ in range(nsamples)],
            timings={},
        )


# ── Sampler ────────────────────────────────────────────────────────────


def test_sampler_counts_and_bit_order():
    sampler = MimiqSamplerV2(MimiqBackend(_SamplerStub()))
    qc = QuantumCircuit(2, 2)
    qc.measure([0, 1], [0, 1])

    pub_result = sampler.run([qc], shots=64).result()[0]
    counts = pub_result.data.c.get_counts()
    # clbit 0 high, clbit 1 low → Qiskit key "01" (clbit 0 is the LSB).
    assert counts == {"01": 64}


def test_sampler_multiple_registers():
    from qiskit import ClassicalRegister, QuantumRegister

    qr = QuantumRegister(2, "q")
    a = ClassicalRegister(1, "a")
    b = ClassicalRegister(1, "b")
    qc = QuantumCircuit(qr, a, b)
    qc.measure(qr[0], a[0])  # global clbit 0 → high
    qc.measure(qr[1], b[0])  # global clbit 1 → low

    sampler = MimiqSamplerV2(MimiqBackend(_SamplerStub()))
    data = sampler.run([qc], shots=10).result()[0].data
    assert data.a.get_counts() == {"1": 10}
    assert data.b.get_counts() == {"0": 10}


def test_sampler_parameter_broadcasting():
    t = Parameter("t")
    qc = QuantumCircuit(1, 1)
    qc.rx(t, 0)
    qc.measure(0, 0)

    sampler = MimiqSamplerV2(MimiqBackend(_SamplerStub()))
    pub_result = sampler.run(
        [(qc, [[0.1], [0.2], [0.3]])], shots=8
    ).result()[0]
    assert pub_result.data.c.shape == (3,)
    assert pub_result.data.c.num_shots == 8


def test_sampler_default_shots():
    sampler = MimiqSamplerV2(MimiqBackend(_SamplerStub()), default_shots=32)
    qc = QuantumCircuit(1, 1)
    qc.measure(0, 0)
    pub_result = sampler.run([qc]).result()[0]
    assert pub_result.data.c.num_shots == 32


# ── Estimator ────────────────────────────────────────────────────────────


def test_estimator_single_observable():
    estimator = MimiqEstimatorV2(MimiqBackend(_EstimatorStub()))
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    pub_result = estimator.run([(qc, "ZZ")]).result()[0]
    assert pub_result.data.evs.shape == ()
    # The single term reads z[0], which the stub sets to 1.
    assert float(pub_result.data.evs) == 1.0
    assert float(pub_result.data.stds) == 0.0  # exact: no shot noise
    assert pub_result.metadata["method"] == "exact"
    assert pub_result.metadata["exact"] is True
    assert pub_result.metadata["stochastic"] is False
    assert pub_result.metadata["trajectories"] == 1
    assert pub_result.metadata["min_fidelity"] == 1.0


def test_estimator_shares_one_submission_across_observables():
    """Observables asked for at one binding cost one evolution, not one each.

    They are read from the same state, so re-evolving it per observable
    would be wasted work. Each observable still gets its own z-variable.
    """
    stub = _EstimatorStub()
    estimator = MimiqEstimatorV2(MimiqBackend(stub))
    qc = QuantumCircuit(2)
    qc.h(0)

    evs = estimator.run([(qc, ["ZZ", "XX", "YY"])]).result()[0].data.evs
    assert evs.shape == (3,)
    assert len(stub.circuits) == 1
    # z[0], z[1], z[2] in the order the observables were given.
    assert list(evs) == [1.0, 2.0, 3.0]


def test_estimator_reuses_one_z_variable_for_a_repeated_term():
    """A term two observables share is evaluated once."""
    stub = _EstimatorStub()
    estimator = MimiqEstimatorV2(MimiqBackend(stub))
    qc = QuantumCircuit(2)
    qc.h(0)

    obs = [{"ZZ": 1.0}, {"ZZ": 2.0, "XX": 1.0}]
    evs = estimator.run([(qc, obs)]).result()[0].data.evs
    assert stub.circuits[0].num_zvars() == 2  # ZZ and XX, not three
    # ZZ is z[0] = 1 and XX is z[1] = 2.
    assert list(evs) == [1.0, 2.0 * 1.0 + 1.0 * 2.0]


def test_estimator_pushes_only_a_terms_own_qubits():
    """A narrow term stays narrow, and lands on the qubits Qiskit meant.

    Qiskit's "IZ" is Z on qubit 0, so the pushed operation must be a
    one-qubit ``⟨Z⟩`` on qubit 0, not a two-qubit string padded with an
    identity.
    """
    import mimiqcircuits as mc

    stub = _EstimatorStub()
    estimator = MimiqEstimatorV2(MimiqBackend(stub))
    qc = QuantumCircuit(3)
    qc.h(0)

    estimator.run([(qc, "IIZ")]).result()
    pushed = [
        inst
        for inst in stub.circuits[0].instructions
        if isinstance(inst.operation, mc.ExpectationValue)
    ]
    assert len(pushed) == 1
    assert str(pushed[0].operation.op.pauli) == "Z"
    assert list(pushed[0].qubits) == [0]


def test_estimator_identity_observable_needs_no_run():
    """An identity term is its own coefficient: nothing to evolve."""
    stub = _EstimatorStub()
    estimator = MimiqEstimatorV2(MimiqBackend(stub))
    qc = QuantumCircuit(2)
    qc.h(0)

    result = estimator.run([(qc, SparsePauliOp(["II"], [3.5]))]).result()[0]
    assert float(result.data.evs) == 3.5
    assert float(result.data.stds) == 0.0
    assert stub.circuits == []


def test_estimator_parameter_broadcasting():
    t = Parameter("t")
    qc = QuantumCircuit(1)
    qc.rx(t, 0)

    stub = _EstimatorStub()
    estimator = MimiqEstimatorV2(MimiqBackend(stub))
    pub = (qc, "Z", [[0.0], [0.5], [1.0]])
    evs = estimator.run([pub]).result()[0].data.evs
    assert evs.shape == (3,)
    # One binding per submission: they are different circuits.
    assert len(stub.circuits) == 3


def test_estimator_converts_each_binding_once(monkeypatch):
    """Guards against converting every binding twice per pub.

    Deciding whether the circuit is stochastic needs a MIMIQ circuit, and so
    does reading the Pauli terms off the state. Converting separately for
    each doubled the Qiskit-side cost of a pub, which a parameter sweep pays
    once per row and which dominates the run well before the simulator does.
    """
    import mimiq_qiskit.estimation as estimation

    calls = []
    original = estimation.qiskit_to_mimiq

    def counting(circuit):
        calls.append(circuit)
        return original(circuit)

    monkeypatch.setattr(estimation, "qiskit_to_mimiq", counting)

    t = Parameter("t")
    qc = QuantumCircuit(1)
    qc.rx(t, 0)

    estimator = MimiqEstimatorV2(MimiqBackend(_EstimatorStub()))
    bindings = [[0.0], [0.25], [0.5], [0.75], [1.0]]
    # Two observables per binding, to pin that the count follows the number
    # of distinct bindings rather than the number of output elements.
    observables = [["Z"], ["X"]]
    pub = (qc, observables, [bindings])
    evs = estimator.run([pub]).result()[0].data.evs

    assert evs.shape == (2, len(bindings))
    assert len(calls) == len(bindings)


def test_estimator_converts_nothing_it_cannot_use(monkeypatch):
    """The stochastic predicate keeps both of its short-circuits.

    ``assume_stochastic`` settles the question without looking at a circuit,
    and ``any`` stops at the first binding that needs trajectories. Neither
    conversion would be reused: the sampled path appends measurements on the
    Qiskit side and converts its own copies, so a binding converted for the
    predicate is work thrown away.
    """
    import mimiq_qiskit.estimation as estimation

    calls = []
    original = estimation.qiskit_to_mimiq
    monkeypatch.setattr(
        estimation,
        "qiskit_to_mimiq",
        lambda circuit: (calls.append(circuit), original(circuit))[1],
    )

    t = Parameter("t")
    deterministic = QuantumCircuit(1)
    deterministic.rx(t, 0)
    # A reset makes the circuit end in an ensemble, so the predicate is true
    # on the first binding it looks at.
    stochastic = deterministic.copy()
    stochastic.reset(0)

    bindings = [[0.0], [0.25], [0.5], [0.75], [1.0]]
    n = len(bindings)

    # A noise model is invisible in the circuit, so `assume_stochastic` is set
    # and the predicate never runs: only the sampled path converts.
    calls.clear()
    MimiqEstimatorV2(
        MimiqBackend(_ShotStub("0")),
        shots=8,
        run_options={"noisemodel": object()},
    ).run([(deterministic, [["Z"]], [bindings])]).result()
    assert len(calls) == n

    # Without the flag the predicate runs, but stops at the first binding.
    calls.clear()
    MimiqEstimatorV2(MimiqBackend(_ShotStub("0")), shots=8).run(
        [(stochastic, [["Z"]], [bindings])]
    ).result()
    assert len(calls) == 1 + n


# ── stochastic circuits ──────────────────────────────────────────────────


def _mid_circuit_measurement() -> QuantumCircuit:
    """A circuit whose measurement is state preparation, not readout.

    The trailing ``x`` keeps the measurement from being final, so it is not
    removed and the circuit really does end in an ensemble.
    """
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    qc.x(0)
    return qc


def test_estimator_stochastic_circuit_without_a_budget_raises():
    """One trajectory is not the ensemble average, so it is not offered."""
    estimator = MimiqEstimatorV2(MimiqBackend(_EstimatorStub()))
    with pytest.raises(ValueError, match="one evolution per shot"):
        estimator.run([(_mid_circuit_measurement(), "Z")]).result()


def test_estimator_averages_trajectories_with_a_real_error_bar():
    values = [1.0, -1.0, 1.0, 1.0]
    estimator = MimiqEstimatorV2(
        MimiqBackend(_TrajectoryStub(values)), trajectories=len(values)
    )
    result = estimator.run([(_mid_circuit_measurement(), "Z")]).result()[0]

    assert float(result.data.evs) == pytest.approx(0.5)
    assert float(result.data.stds) == pytest.approx(0.5)
    assert result.metadata["method"] == "trajectories"
    assert result.metadata["exact"] is False
    assert result.metadata["stochastic"] is True
    assert result.metadata["trajectories"] == 4


def test_estimator_precision_sizes_the_trajectory_budget():
    stub = _TrajectoryStub([1.0] * 400)
    estimator = MimiqEstimatorV2(MimiqBackend(stub))
    result = estimator.run(
        [(_mid_circuit_measurement(), "Z")], precision=0.05
    ).result()[0]
    assert result.metadata["trajectories"] == 400  # ceil(1 / 0.05**2)


def test_estimator_method_exact_warns_on_a_stochastic_circuit():
    """Forcing ``exact`` is allowed, but the label does not apply."""
    estimator = MimiqEstimatorV2(
        MimiqBackend(_TrajectoryStub([1.0])), method="exact"
    )
    with pytest.warns(UserWarning, match="single random trajectory"):
        result = estimator.run(
            [(_mid_circuit_measurement(), "Z")]
        ).result()[0]
    assert result.metadata["method"] == "exact"
    assert result.metadata["exact"] is False
    assert result.metadata["stochastic"] is True


def test_estimator_treats_a_noise_model_as_stochastic():
    """A run option's noise never appears in the circuit, so it is declared.

    Without this the estimator would call a noisy run deterministic and
    report one trajectory as exact.
    """
    estimator = MimiqEstimatorV2(
        MimiqBackend(_EstimatorStub()), run_options={"noisemodel": object()}
    )
    qc = QuantumCircuit(1)
    qc.h(0)
    with pytest.raises(ValueError, match="one evolution per shot"):
        estimator.run([(qc, "Z")]).result()


def test_estimator_ignores_trajectories_on_a_deterministic_circuit():
    """A circuit with one answer is not evolved again for a second look."""
    stub = _EstimatorStub(trajectories=8)
    estimator = MimiqEstimatorV2(MimiqBackend(stub), trajectories=8)
    qc = QuantumCircuit(1)
    qc.h(0)

    result = estimator.run([(qc, "Z")]).result()[0]
    assert result.metadata["trajectories"] == 1
    assert result.metadata["exact"] is True


# ── sampled estimation ───────────────────────────────────────────────────


def test_estimator_shots_mode_reads_parities():
    """Every shot reads clbit 0 high, so ⟨Z⟩ on qubit 0 must come back -1."""
    estimator = MimiqEstimatorV2(MimiqBackend(_ShotStub("1")), shots=100)
    qc = QuantumCircuit(1)
    qc.h(0)

    result = estimator.run([(qc, SparsePauliOp(["Z"], [2.0]))]).result()[0]
    assert float(result.data.evs) == pytest.approx(-2.0)
    # A deterministic ±1 leaves no variance, so the error bar closes.
    assert float(result.data.stds) == pytest.approx(0.0)
    assert result.metadata["method"] == "shots"
    assert result.metadata["shots"] == 100
    assert result.metadata["exact"] is False


def test_estimator_shots_mode_groups_commuting_terms():
    """Terms over one basis share a measurement circuit."""
    stub = _EstimatorStub()  # records circuits; cstates are empty
    estimator = MimiqEstimatorV2(MimiqBackend(stub), shots=8)
    qc = QuantumCircuit(2)
    qc.h(0)
    with pytest.raises(ValueError, match="samples"):
        estimator.run([(qc, ["ZZ", "ZI", "IZ"])]).result()
    # One Z-basis measurement covers all three terms.
    assert len(stub.circuits) == 1


def test_estimator_shots_mode_needs_the_requested_shots():
    """A short result would leave the error bar claiming a budget it lacks."""
    stub = _ShotStub("1")

    class Short:
        def __call__(self, circuit, *, nsamples, seed):
            return stub(circuit, nsamples=2, seed=seed)

    estimator = MimiqEstimatorV2(MimiqBackend(Short()), shots=8)
    qc = QuantumCircuit(1)
    qc.h(0)
    with pytest.raises(ValueError, match="8 shots were requested"):
        estimator.run([(qc, "Z")]).result()


# ── configuration ────────────────────────────────────────────────────────


def test_estimator_rejects_contradictory_budgets():
    backend = MimiqBackend(_EstimatorStub())
    with pytest.raises(ValueError, match="not both"):
        MimiqEstimatorV2(backend, trajectories=4, shots=4)
    with pytest.raises(ValueError, match="method must be one of"):
        MimiqEstimatorV2(backend, method="sampled")
    with pytest.raises(ValueError, match="at least 1"):
        MimiqEstimatorV2(backend, shots=0)
    with pytest.raises(ValueError, match="does not sample"):
        MimiqEstimatorV2(backend, method="trajectories", shots=4)


def test_estimator_emulated_shot_noise_perturbs_an_exact_value():
    """Gaussian noise at ``precision``, for one evolution's worth of work."""
    qc = QuantumCircuit(1)
    qc.h(0)

    backend = MimiqBackend(_EstimatorStub())
    exact = MimiqEstimatorV2(backend).run([(qc, "Z")], precision=0.1)
    noisy = MimiqEstimatorV2(
        backend, emulate_shot_noise=True, seed=7
    ).run([(qc, "Z")], precision=0.1)

    exact_result = exact.result()[0]
    noisy_result = noisy.result()[0]
    assert float(exact_result.data.evs) == 1.0
    assert float(noisy_result.data.evs) != 1.0
    assert float(noisy_result.data.evs) == pytest.approx(1.0, abs=0.5)
    assert float(noisy_result.data.stds) == 0.1
    assert noisy_result.metadata["exact"] is False


def test_sampler_accepts_raw_runner():
    # A non-MimiqBackend argument is wrapped automatically.
    sampler = MimiqSamplerV2(_SamplerStub())
    qc = QuantumCircuit(1, 1)
    qc.measure(0, 0)
    counts = sampler.run([qc], shots=4).result()[0].data.c.get_counts()
    assert counts == {"1": 4}

def test_sampler_rejects_a_short_result():
    """A BitArray has a fixed shot axis, so a short result cannot be
    packed into it; the sampler must say so rather than read off the end
    of the sample list."""

    def short_runner(circuit, *, nsamples, seed):
        return QCSResults(
            simulator="stub",
            version="0",
            cstates=[bitarray("0" * max(circuit.num_bits(), 1))
                     for _ in range(nsamples // 2)],
            timings={},
        )

    qc = QuantumCircuit(2, 2)
    qc.measure([0, 1], [0, 1])

    sampler = MimiqSamplerV2(MimiqBackend(short_runner))
    with pytest.raises(ValueError, match="returned 5 samples"):
        sampler.run([qc], shots=10).result()


# ── direct term evaluation (the optional `evaluate` hook) ──────────────


class _ListEstimatorStub:
    """`_EstimatorStub` at the `estimate_pub` seam: a list in, a list out."""

    def __init__(self, fidelity: float = 1.0):
        self.calls = 0

    def __call__(self, circuits, nsamples):
        self.calls += 1
        return [
            QCSResults(
                simulator="stub",
                version="0",
                cstates=[],
                zstates=[
                    [complex(i + 1) for i in range(c.num_zvars())]
                    for _ in range(max(nsamples, 1))
                ],
                fidelities=[1.0],
                timings={},
            )
            for c in circuits
        ]


class _EvaluateStub:
    """Answers the k-th label of each circuit with ``k + 1``.

    Mirrors `_EstimatorStub`, which fills z[k] with k + 1, so the same pub
    run through either path must come out the same.
    """

    def __init__(
        self,
        fidelity: float = 1.0,
        drop: str | None = None,
        imag: float = 0.0,
    ):
        self.requests = []
        self._fidelity = fidelity
        self._drop = drop
        self._imag = imag

    def __call__(self, requests):
        self.requests.extend(requests)
        return [
            TermValues(
                {
                    label: complex(k + 1, self._imag) if self._imag else float(k + 1)
                    for k, label in enumerate(labels)
                    if label != self._drop
                },
                (self._fidelity,),
            )
            for _circuit, labels in requests
        ]


def _refuse_run(circuits, nsamples):
    raise AssertionError("run must not be called when evaluate serves the pub")


def _estimate(pub, config=None, run=_refuse_run, evaluate=None):
    return estimate_pub(
        EstimatorPub.coerce(pub, 0.0), config or EstimatorConfig(), run, evaluate
    )


def test_evaluate_serves_a_deterministic_pub():
    """The terms go beside the circuit, not onto it, and `run` is untouched."""
    ev = _EvaluateStub(fidelity=0.75)
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    result = _estimate((qc, [{"ZZ": 2.0, "XX": 1.0}]), evaluate=ev)

    # ZZ is the first label (2.0 * 1) and XX the second (1.0 * 2).
    assert list(result.data.evs) == [4.0]
    assert list(result.data.stds) == [0.0]
    assert result.metadata["min_fidelity"] == 0.75
    circuit, labels = ev.requests[0]
    assert labels == ["ZZ", "XX"]
    assert circuit.num_zvars() == 0  # no ExpectationValue was pushed


def test_evaluate_and_pushed_terms_agree():
    """Both routes read the same terms, so they must report the same value."""
    qc = QuantumCircuit(2)
    qc.h(0)
    obs = [{"ZZ": 1.0}, {"ZZ": 2.0, "XX": 1.0, "II": 0.5}]

    pushed = _estimate((qc, obs), run=_ListEstimatorStub())
    direct = _estimate((qc, obs), evaluate=_EvaluateStub())

    assert list(direct.data.evs) == list(pushed.data.evs)


def test_evaluate_is_skipped_for_a_stochastic_circuit():
    """One evolution is not the ensemble, so the hook does not serve it."""
    ev = _EvaluateStub()
    run = _ListEstimatorStub()
    config = EstimatorConfig(trajectories=3)

    result = _estimate(
        (_mid_circuit_measurement(), "Z"), config=config, run=run, evaluate=ev
    )

    assert ev.requests == []
    assert run.calls == 1
    assert result.metadata["stochastic"] is True


def test_evaluate_missing_a_label_raises():
    ev = _EvaluateStub(drop="XX")
    qc = QuantumCircuit(2)
    qc.h(0)

    with pytest.raises(ValueError, match="no value for Pauli term 'XX'"):
        _estimate((qc, [{"ZZ": 1.0, "XX": 1.0}]), evaluate=ev)


def test_evaluate_may_return_complex_values():
    """`evs` is a float array, so the real part is taken here, not trusted.

    A Pauli string on a normalised state has a real expectation value, but a
    backend is free to hand back the complex number it computed; the operation
    route takes `.real` off the z-register for the same reason.
    """
    qc = QuantumCircuit(2)
    qc.h(0)
    obs = [{"ZZ": 2.0, "XX": 1.0}]

    real = _estimate((qc, obs), evaluate=_EvaluateStub())
    complexy = _estimate((qc, obs), evaluate=_EvaluateStub(imag=1e-16))

    assert list(complexy.data.evs) == list(real.data.evs)
    assert complexy.data.evs.dtype.kind == "f"


# ── the site map behind a direct read ──────────────────────────────────


def test_site_map_reads_a_permutation_base_off_its_values():
    """0- and 1-based permutations both arrive; the values say which.

    `apply_passes` composes as 1-based while every pass that ships today
    emits 0-based, so the convention is read rather than assumed.
    """
    from mimiq_qiskit.local_terms import _site_map

    zero_based = _site_map([[2, 0, 1]], 3)
    assert [zero_based(q) for q in range(3)] == [2, 0, 1]

    one_based = _site_map([[3, 1, 2]], 3)
    assert [one_based(q) for q in range(3)] == [2, 0, 1]

    assert [_site_map([None, None], 3)(q) for q in range(3)] == [0, 1, 2]


def test_site_map_composes_the_passes_with_the_compile():
    """A backend may relabel in both places, and the two have to compose.

    exaqt reorders in the pass pipeline and tensorweaver inside `compile`,
    so the map is built from whichever steps happened, in order.
    """
    from mimiq_qiskit.local_terms import _site_map

    # 0 -> 1 -> 2, 1 -> 2 -> 0, 2 -> 0 -> 1.
    site = _site_map([[1, 2, 0], [1, 2, 0]], 3)
    assert [site(q) for q in range(3)] == [2, 0, 1]

    # A step that changed nothing drops out.
    assert [_site_map([None, [1, 2, 0]], 3)(q) for q in range(3)] == [1, 2, 0]


def test_site_map_rejects_a_non_permutation():
    from mimiq_qiskit.local_terms import _site_map

    with pytest.raises(ValueError, match="not a permutation"):
        _site_map([[0, 0, 2]], 3)


# ── preparation knobs come from `execute`, not from a copy ─────────────


def test_prep_kwargs_read_the_defaults_off_execute():
    """A default lives on `execute`; this route reads it rather than repeating it.

    Copying the literals would let the two routes prepare circuits
    differently the day one of those defaults moves, with nothing to say so.
    """
    from mimiq_qiskit.local_terms import _PREP_KNOBS, _prep_kwargs

    class _Backend:
        def execute(
            self,
            circuit,
            *,
            nsamples=1000,
            seed=None,
            fuse=True,
            fuse_threshold=7,
            canonicaldecompose=True,
            reorderqubits="greedy",
            remove_swaps=True,
        ):
            raise AssertionError("not called")

    assert _prep_kwargs(_Backend(), {}) == {
        "fuse": True,
        "fuse_threshold": 7,
        "canonicaldecompose": True,
        "reorderqubits": "greedy",
        "remove_swaps": True,
    }
    # An option the caller set wins over the signature's default.
    assert _prep_kwargs(_Backend(), {"fuse": False})["fuse"] is False
    assert set(_prep_kwargs(_Backend(), {})) == set(_PREP_KNOBS)


def test_prep_kwargs_fall_back_to_the_base_class_for_a_knob_execute_omits():
    """A backend may narrow `execute` to the knobs it implements.

    exaqt names none of the preparation knobs. A knob its `execute` does
    not take is one it cannot set either, so the value that stands in for
    the submission is what `LocalBackend.execute` would have resolved with,
    not a TypeError.
    """
    from mimiqcircuits.backends import LocalBackend

    from mimiq_qiskit.local_terms import _PREP_KNOBS, _prep_kwargs

    class _Backend:
        def execute(self, circuit, *, nsamples=1000, seed=None):
            raise AssertionError("not called")

    base = signature(LocalBackend.execute).parameters
    resolved = _prep_kwargs(_Backend(), {})
    assert set(resolved) == set(_PREP_KNOBS)
    assert resolved == {knob: base[knob].default for knob in _PREP_KNOBS}
    # An option the caller set still wins.
    assert _prep_kwargs(_Backend(), {"fuse": False})["fuse"] is False


def test_prep_kwargs_reject_a_knob_with_no_default():
    """Nothing to prepare with, so say which knob rather than guess one."""
    from mimiq_qiskit.local_terms import _prep_kwargs

    class _Backend:
        def execute(self, circuit, *, fuse, **kwargs):
            raise AssertionError("not called")

    with pytest.raises(TypeError, match="no default for 'fuse'"):
        _prep_kwargs(_Backend(), {})
