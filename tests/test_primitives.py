"""Tests for the native MIMIQ Sampler/Estimator V2 primitives.

These use stub runners that fabricate ``QCSResults`` so the tests cover
the primitive plumbing (bitstring packing, register/observable
broadcasting, result shapes) without a MIMIQ server. The numerical
physics (sampling distributions, expectation values) is MIMIQ's own,
exercised by its test suite; here we verify the conversion is faithful.
"""

from __future__ import annotations

from bitarray import bitarray
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp

from mimiqcircuits import QCSResults

from mimiq_qiskit import MimiqBackend, MimiqEstimatorV2, MimiqSamplerV2


class _SamplerStub:
    """Every shot reports clbit 0 high, the rest low (MIMIQ bit order)."""

    def __call__(self, circuit, *, nsamples, seed):
        nb = max(circuit.num_bits(), 1)
        cstates = [bitarray("1" + "0" * (nb - 1)) for _ in range(nsamples)]
        return QCSResults(
            simulator="stub", version="0", cstates=cstates, timings={}
        )


class _EstimatorStub:
    """Returns z[0] = (call index) so result ordering is observable."""

    def __init__(self):
        self.calls = 0

    def __call__(self, circuit, *, nsamples, seed):
        z = [[complex(self.calls, 0.0)]]
        self.calls += 1
        return QCSResults(
            simulator="stub", version="0", cstates=[], zstates=z, timings={}
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
    assert float(pub_result.data.evs) == 0.0
    assert float(pub_result.data.stds) == 0.0  # exact: no shot noise


def test_estimator_observable_array_shape_and_order():
    estimator = MimiqEstimatorV2(MimiqBackend(_EstimatorStub()))
    qc = QuantumCircuit(2)
    qc.h(0)

    obs = ["ZZ", "XX", "YY"]
    evs = estimator.run([(qc, obs)]).result()[0].data.evs
    assert evs.shape == (3,)
    # The stub returns z[0] = submission index, in np.ndindex order.
    assert list(evs) == [0.0, 1.0, 2.0]


def test_estimator_parameter_broadcasting():
    t = Parameter("t")
    qc = QuantumCircuit(1)
    qc.rx(t, 0)

    estimator = MimiqEstimatorV2(MimiqBackend(_EstimatorStub()))
    pub = (qc, "Z", [[0.0], [0.5], [1.0]])
    evs = estimator.run([pub]).result()[0].data.evs
    assert evs.shape == (3,)


def test_observable_to_hamiltonian_endianness_and_coeffs():
    # Qiskit "IZ" is Z on qubit 0 → MIMIQ "ZI" (char 0 acts on qubit 0).
    from qiskit.primitives.containers.observables_array import ObservablesArray

    from mimiq_qiskit.primitives import _observable_to_hamiltonian

    spo = SparsePauliOp(["IZ", "XX"], [1.5, 0.5])
    obs = ObservablesArray.coerce(spo)[()]
    ham = _observable_to_hamiltonian(obs, 2)
    terms = {str(t.pauli): t.get_coefficient() for t in ham}
    assert terms == {"ZI": 1.5, "XX": 0.5}


def test_sampler_accepts_raw_runner():
    # A non-MimiqBackend argument is wrapped automatically.
    sampler = MimiqSamplerV2(_SamplerStub())
    qc = QuantumCircuit(1, 1)
    qc.measure(0, 0)
    counts = sampler.run([qc], shots=4).result()[0].data.c.get_counts()
    assert counts == {"1": 4}
