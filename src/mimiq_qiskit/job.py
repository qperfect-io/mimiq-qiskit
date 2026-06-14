"""Qiskit ``JobV1`` returned by :meth:`MimiqBackend.run`.

The job executes in a background thread so ``backend.run`` returns
immediately and ``job.result()`` is the blocking call, matching the
Qiskit Aer and IBM provider conventions. Cancellation is not supported:
the MIMIQ client does not yet expose a ``stopExecution`` endpoint, so
``cancel`` raises rather than pretending to abort the remote job.
"""

from __future__ import annotations

import threading
import uuid
from typing import Callable

from qiskit.providers import JobStatus, JobV1
from qiskit.result import Result

from mimiq_qiskit.result import qcsresults_to_qiskit_result


class MimiqJob(JobV1):
    """Background-thread job. One thread per ``run()`` call.

    ``work`` is a zero-argument callable returning the list of
    ``QCSResults`` (one per submitted circuit); the whole batch is one
    MIMIQ job.
    """

    def __init__(
        self,
        backend,
        *,
        qiskit_circuits,
        work: Callable,
        shots: int,
        job_id: str | None = None,
    ):
        super().__init__(backend, job_id or str(uuid.uuid4()))
        self._qiskit_circuits = qiskit_circuits
        self._work = work
        self._shots = shots
        self._qcs_results = None
        self._exc: BaseException | None = None
        self._thread: threading.Thread | None = None
        self._status = JobStatus.INITIALIZING

    # ── lifecycle ──────────────────────────────────────────────────────

    def submit(self) -> None:
        if self._thread is not None:
            raise RuntimeError("MimiqJob has already been submitted")
        self._status = JobStatus.RUNNING
        self._thread = threading.Thread(
            target=self._target, name=f"mimiq-job-{self.job_id()}", daemon=True
        )
        self._thread.start()

    def _target(self) -> None:
        try:
            self._qcs_results = self._work()
            self._status = JobStatus.DONE
        except BaseException as exc:
            self._exc = exc
            self._status = JobStatus.ERROR

    # ── status ─────────────────────────────────────────────────────────

    def status(self) -> JobStatus:
        return self._status

    def cancel(self) -> None:
        raise NotImplementedError(
            "MimiqJob does not support cancellation; the MIMIQ client "
            "exposes no stopExecution endpoint yet"
        )

    # ── result ─────────────────────────────────────────────────────────

    def result(self, timeout: float | None = None) -> Result:
        if self._thread is None:
            raise RuntimeError("MimiqJob has not been submitted")
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            raise TimeoutError(
                f"MimiqJob {self.job_id()} did not finish within {timeout}s"
            )
        if self._exc is not None:
            raise self._exc
        return qcsresults_to_qiskit_result(
            self._qcs_results,
            qiskit_circuits=self._qiskit_circuits,
            backend_name=self.backend().name,
            backend_version=self.backend().backend_version,
            job_id=self.job_id(),
            shots=self._shots,
        )
