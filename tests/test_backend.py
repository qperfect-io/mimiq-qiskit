"""Backend end-to-end test using a stub runner.

Avoids touching the real MIMIQ cloud — the stub fabricates a
``QCSResults`` so we can verify the ``backend.run -> job.result ->
counts`` path without a server round-trip.
"""

from __future__ import annotations

from bitarray import bitarray
from qiskit import QuantumCircuit
from qiskit.providers import JobStatus

from mimiqcircuits import QCSResults

from mimiq_qiskit import MimiqBackend, MimiqProvider


def _stub_runner(circuit, *, nsamples, seed):
    """Return a deterministic QCSResults — half '00', half '11'."""
    nq = max(circuit.num_qubits(), 1)
    nb = max(circuit.num_bits(), nq)
    half = nsamples // 2
    other = nsamples - half
    zeros = bitarray("0" * nb)
    ones = bitarray("1" * nb)
    cstates = [zeros for _ in range(half)] + [ones for _ in range(other)]
    return QCSResults(
        simulator="stub",
        version="0.0",
        cstates=cstates,
        fidelities=[1.0],
        avggateerrors=[0.0],
        timings={"total": 0.001},
    )


def test_run_returns_counts():
    backend = MimiqBackend(_stub_runner, num_qubits=4)
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])

    job = backend.run(qc, shots=1000)
    result = job.result()

    assert job.status() == JobStatus.DONE
    counts = result.get_counts()
    # Stub maps half to '00' (hex 0x0) and half to '11' (hex 0x3).
    assert sum(counts.values()) == 1000
    assert set(counts.keys()) <= {"00", "11"}
    assert counts.get("00", 0) + counts.get("11", 0) == 1000


def test_run_propagates_metadata():
    backend = MimiqBackend(_stub_runner)
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)

    result = backend.run(qc, shots=10).result()
    meta = result.results[0].metadata
    assert meta["simulator"] == "stub"
    assert meta["fidelities"] == [1.0]
    assert "total" in meta["timings"]


def test_provider_get_backend():
    provider = MimiqProvider(_stub_runner)
    backend = provider.get_backend("mimiq")
    assert isinstance(backend, MimiqBackend)
    assert backend.name == "mimiq"


def test_multiple_circuits():
    backend = MimiqBackend(_stub_runner)
    qc_a = QuantumCircuit(1, 1)
    qc_a.h(0)
    qc_a.measure(0, 0)
    qc_b = QuantumCircuit(2, 2)
    qc_b.x(0)
    qc_b.measure([0, 1], [0, 1])

    result = backend.run([qc_a, qc_b], shots=20).result()
    assert len(result.results) == 2
    assert sum(result.get_counts(0).values()) == 20
    assert sum(result.get_counts(1).values()) == 20


def test_run_emits_per_shot_memory():
    # SamplerV2 reconstructs its bit arrays from `memory`, not `counts`,
    # so the result must carry one hex record per shot.
    backend = MimiqBackend(_stub_runner)
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])

    result = backend.run(qc, shots=10).result()
    memory = result.get_memory(0)
    assert len(memory) == 10
    assert set(memory) <= {"00", "11"}


def test_sampler_v2_roundtrip():
    from qiskit.primitives import BackendSamplerV2

    backend = MimiqBackend(_stub_runner)
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    sampler = BackendSamplerV2(backend=backend)
    counts = sampler.run([qc], shots=100).result()[0].data.meas.get_counts()
    assert sum(counts.values()) == 100
    assert set(counts) <= {"00", "11"}


class _RecordingBackend:
    """MIMIQ-backend-shaped stub: ``execute`` takes a list, returns a list,
    and records the call so we can assert one-job batching."""

    def __init__(self):
        self.calls = []

    def execute(self, circuits, *, nsamples, seed, **kwargs):
        self.calls.append((len(circuits), nsamples, seed, kwargs))
        out = []
        for c in circuits:
            nb = max(c.num_bits(), 1)
            out.append(
                QCSResults(
                    simulator="rec",
                    version="0",
                    cstates=[bitarray("0" * nb) for _ in range(nsamples)],
                    timings={},
                )
            )
        return out


def test_batch_submitted_as_single_job():
    rec = _RecordingBackend()
    backend = MimiqBackend(rec)
    qc_a = QuantumCircuit(1, 1)
    qc_a.measure(0, 0)
    qc_b = QuantumCircuit(2, 2)
    qc_b.measure([0, 1], [0, 1])

    result = backend.run([qc_a, qc_b], shots=7).result()
    # One execute call carrying both circuits — not one job per circuit.
    assert len(rec.calls) == 1
    assert rec.calls[0][0] == 2
    assert rec.calls[0][1] == 7
    assert len(result.results) == 2


def test_run_options_forwarded_to_mimiq():
    rec = _RecordingBackend()
    backend = MimiqBackend(rec)
    qc = QuantumCircuit(1, 1)
    qc.measure(0, 0)

    backend.run(qc, shots=5, bonddim=16, timelimit=30).result()
    kwargs = rec.calls[0][3]
    assert kwargs["bonddim"] == 16
    assert kwargs["timelimit"] == 30
    # Unset options are not forwarded.
    assert "noisemodel" not in kwargs


def test_backend_target_lists_standard_gates():
    backend = MimiqBackend(_stub_runner, num_qubits=5)
    op_names = set(backend.target.operation_names)
    # Spot-check a representative subset.
    for name in ("h", "cx", "rz", "u", "swap", "ccx", "measure", "reset"):
        assert name in op_names, f"missing {name} in target operations"
