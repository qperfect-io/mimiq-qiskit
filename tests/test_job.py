"""Job lifecycle and result shaping.

``MimiqJob`` runs on a background thread, so its failure paths (an
exception raised inside the worker, a result awaited before submission, a
timeout) are the ones most likely to go untested and most annoying when
wrong: a swallowed exception turns a server error into an empty result.
"""

from __future__ import annotations

import threading

import pytest
from bitarray import bitarray
from mimiqcircuits import QCSResults
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.providers import JobStatus

from mimiq_qiskit import MimiqBackend, MimiqProvider
from mimiq_qiskit.job import MimiqJob
from mimiq_qiskit.result import qcsresults_to_qiskit_result


def _results(cstates: list[str]) -> QCSResults:
    return QCSResults(
        simulator="stub",
        version="0.0",
        cstates=[bitarray(s) for s in cstates],
        timings={"total": 0.0},
    )


def _measured_circuit(nq: int = 1) -> QuantumCircuit:
    qc = QuantumCircuit(nq, nq)
    qc.measure(range(nq), range(nq))
    return qc


# ── failure propagation ────────────────────────────────────────────────


def test_worker_exception_surfaces_from_result():
    """An error inside the worker must reach the caller, not vanish."""

    class Boom(RuntimeError):
        pass

    def failing_runner(circuit, *, nsamples, seed):
        raise Boom("the server said no")

    backend = MimiqBackend(failing_runner)
    job = backend.run(_measured_circuit(), shots=4)

    with pytest.raises(Boom, match="the server said no"):
        job.result()
    assert job.status() == JobStatus.ERROR


def test_result_before_submit_raises():
    job = MimiqJob(
        backend=MimiqBackend(lambda c, *, nsamples, seed: _results(["0"])),
        qiskit_circuits=[_measured_circuit()],
        work=lambda: [],
        shots=1,
    )
    with pytest.raises(RuntimeError, match="not been submitted"):
        job.result()


def test_double_submit_raises():
    backend = MimiqBackend(lambda c, *, nsamples, seed: _results(["0"]))
    job = backend.run(_measured_circuit(), shots=1)
    job.result()
    with pytest.raises(RuntimeError, match="already been submitted"):
        job.submit()


def test_result_timeout_raises_and_leaves_job_running():
    """A timeout must not be reported as a finished-but-empty result."""
    release = threading.Event()

    def slow_runner(circuit, *, nsamples, seed):
        release.wait(timeout=10)
        return _results(["0"] * nsamples)

    backend = MimiqBackend(slow_runner)
    job = backend.run(_measured_circuit(), shots=2)
    try:
        with pytest.raises(TimeoutError):
            job.result(timeout=0.05)
        assert job.status() == JobStatus.RUNNING
    finally:
        release.set()

    # Once the worker finishes, the same job hands back its result.
    assert sum(job.result(timeout=10).get_counts().values()) == 2


def test_cancel_is_refused_rather_than_faked():
    backend = MimiqBackend(lambda c, *, nsamples, seed: _results(["0"]))
    job = backend.run(_measured_circuit(), shots=1)
    with pytest.raises(NotImplementedError, match="stopExecution"):
        job.cancel()


def test_job_ids_are_distinct():
    backend = MimiqBackend(lambda c, *, nsamples, seed: _results(["0"]))
    ids = {backend.run(_measured_circuit(), shots=1).job_id() for _ in range(5)}
    assert len(ids) == 5


# ── result shaping ─────────────────────────────────────────────────────


def test_cstate_bit_order_is_lsb_first():
    """MIMIQ cstate bit ``i`` is clbit ``i``; Qiskit prints clbits
    most-significant first, so a cstate of ``10`` reads as ``01``."""
    result = qcsresults_to_qiskit_result(
        [_results(["10"])],
        qiskit_circuits=[_measured_circuit(2)],
        backend_name="stub",
        backend_version="0",
        job_id="j",
        shots=1,
    )
    assert result.get_counts() == {"01": 1}


def test_result_reports_register_layout():
    """``creg_sizes`` is what lets Qiskit split counts per register."""
    qr = QuantumRegister(3, "q")
    first = ClassicalRegister(2, "first")
    second = ClassicalRegister(1, "second")
    qc = QuantumCircuit(qr, first, second)
    qc.measure(qr[0], first[0])
    qc.measure(qr[1], first[1])
    qc.measure(qr[2], second[0])

    result = qcsresults_to_qiskit_result(
        [_results(["110"])],
        qiskit_circuits=[qc],
        backend_name="stub",
        backend_version="0",
        job_id="j",
        shots=1,
    )
    header = result.results[0].header
    assert header["memory_slots"] == 3
    assert [list(entry) for entry in header["creg_sizes"]] == [
        ["first", 2],
        ["second", 1],
    ]
    # cstate "110" is clbit0=1, clbit1=1, clbit2=0, so first="11" and
    # second="0". Qiskit renders one space-separated group per register,
    # last register first.
    assert result.get_counts() == {"0 11": 1}


def test_counts_and_memory_agree():
    cstates = ["00", "10", "10", "11"]
    result = qcsresults_to_qiskit_result(
        [_results(cstates)],
        qiskit_circuits=[_measured_circuit(2)],
        backend_name="stub",
        backend_version="0",
        job_id="j",
        shots=len(cstates),
    )
    counts = result.get_counts()
    memory = result.get_memory()
    assert len(memory) == len(cstates)
    assert counts == {"00": 1, "01": 2, "11": 1}
    for key, count in counts.items():
        assert memory.count(key) == count


def test_result_carries_backend_identity():
    backend = MimiqBackend(
        lambda c, *, nsamples, seed: _results(["0"] * nsamples),
        name="my-mimiq",
    )
    result = backend.run(_measured_circuit(), shots=2).result()
    assert result.backend_name == "my-mimiq"
    assert result.backend_version == backend.backend_version


# ── provider ───────────────────────────────────────────────────────────


def test_provider_lists_and_filters_backends():
    provider = MimiqProvider(lambda c, *, nsamples, seed: _results(["0"]))
    assert [b.name for b in provider.backends()] == ["mimiq"]
    assert [b.name for b in provider.backends("mimiq")] == ["mimiq"]
    assert provider.backends("nope") == []


def test_provider_unknown_backend_names_the_alternatives():
    provider = MimiqProvider(lambda c, *, nsamples, seed: _results(["0"]))
    with pytest.raises(ValueError, match="available: \\['mimiq'\\]"):
        provider.get_backend("nope")


def test_provider_forwards_num_qubits():
    provider = MimiqProvider(
        lambda c, *, nsamples, seed: _results(["0"]), num_qubits=7
    )
    backend = provider.get_backend()
    assert backend.num_qubits == 7
    assert backend.target.num_qubits == 7


def test_backend_rejects_an_unusable_runner():
    with pytest.raises(TypeError, match="runner must be"):
        MimiqBackend(object())
