"""Qiskit ``BackendV2`` that runs Qiskit circuits on MIMIQ.

:class:`MimiqBackend` accepts any of:

- a :class:`mimiqcircuits.RemoteConnection` / ``MimiqConnection`` — the
  cloud path. Wrapped in :class:`mimiqcircuits.backends.MimiqRemoteBackend`
  so its capabilities, limits, and execute semantics carry over.
- a :class:`mimiqcircuits.backends.Backend` instance — any local or
  remote MIMIQ simulator. The backend is used as-is.
- a plain callable ``(circuit, *, nsamples, seed) -> QCSResults`` —
  escape hatch for custom executors and tests.

A whole batch of circuits is submitted to MIMIQ in a **single** job (MIMIQ
natively accepts a list of circuits and returns a list of results), so
``backend.run([qc1, qc2, ...])`` is one round-trip rather than one per
circuit.

The ``Target`` it advertises lists the gates the converter understands;
``transpile(qc, backend=mimiq_backend)`` will decompose unsupported gates
into that set before they reach the converter.
"""

from __future__ import annotations

from typing import Any, Callable

from qiskit.circuit import Measure, Reset
from qiskit.circuit.library import (
    CCXGate,
    CHGate,
    CPhaseGate,
    CRXGate,
    CRYGate,
    CRZGate,
    CSGate,
    CSdgGate,
    CSwapGate,
    CSXGate,
    CUGate,
    CXGate,
    CYGate,
    CZGate,
    DCXGate,
    ECRGate,
    HGate,
    IGate,
    PhaseGate,
    RXGate,
    RXXGate,
    RYGate,
    RYYGate,
    RZGate,
    RZXGate,
    RZZGate,
    SdgGate,
    SGate,
    SwapGate,
    SXdgGate,
    SXGate,
    TdgGate,
    TGate,
    UGate,
    XGate,
    YGate,
    ZGate,
    iSwapGate,
)
from qiskit.circuit import Parameter
from qiskit.providers import BackendV2, Options
from qiskit.transpiler import Target

from mimiq_qiskit.__version__ import __version__
from mimiq_qiskit.converter import qiskit_to_mimiq
from mimiq_qiskit.job import MimiqJob


# Default qubit count advertised to Qiskit. The MIMIQ cloud accepts much
# bigger circuits, but Qiskit's ``Target`` needs a concrete number for
# coupling-map-free (all-to-all) backends. ``run`` itself never rejects a
# larger circuit; this only bounds ``transpile(qc, backend=...)``. Override
# with ``num_qubits=``.
_DEFAULT_NUM_QUBITS = 64

# Execution knobs forwarded to MIMIQ ``execute`` when set (non-``None``).
# These are MIMIQ-specific and have no Qiskit equivalent.
_RUN_OPTION_KEYS = (
    "bonddim",
    "entdim",
    "mpscutoff",
    "mpsmethod",
    "mpotraversal",
    "timelimit",
    "noisemodel",
    "label",
)


def _build_target(num_qubits: int) -> Target:
    """Register supported gates on an all-to-all Target.

    Passing ``properties=None`` for each ``add_instruction`` tells the
    transpiler the gate is available on every qubit / qubit pair with
    no calibration data — sufficient for a software-simulator backend.
    """
    target = Target(num_qubits=num_qubits, description="MIMIQ simulator")

    theta = Parameter("theta")
    phi = Parameter("phi")
    lam = Parameter("lam")
    gam = Parameter("gam")

    one_q: list = [
        IGate(), XGate(), YGate(), ZGate(), HGate(),
        SGate(), SdgGate(), TGate(), TdgGate(),
        SXGate(), SXdgGate(),
        RXGate(theta), RYGate(theta), RZGate(theta),
        PhaseGate(theta), UGate(theta, phi, lam),
    ]
    two_q: list = [
        CXGate(), CYGate(), CZGate(), CHGate(),
        SwapGate(), iSwapGate(),
        CSGate(), CSdgGate(), CSXGate(),
        ECRGate(), DCXGate(),
        CPhaseGate(theta),
        CRXGate(theta), CRYGate(theta), CRZGate(theta),
        CUGate(theta, phi, lam, gam),
        RXXGate(theta), RYYGate(theta), RZZGate(theta), RZXGate(theta),
    ]
    three_q: list = [CCXGate(), CSwapGate()]

    for gate in one_q + two_q + three_q:
        target.add_instruction(gate, properties=None)

    target.add_instruction(Measure(), properties=None)
    target.add_instruction(Reset(), properties=None)
    # Barriers are passed through the transpiler regardless of Target —
    # no need to register them here.

    return target


def _coerce_runner(runner: Any) -> Callable:
    """Normalise the constructor argument into a batch runner
    ``(circuits, *, nsamples, seed, **opts) -> list[QCSResults]``.

    Order of detection matters: ``RemoteConnection`` instances also
    expose ``execute`` (deprecated alias for ``submit``), so we check
    for the connection shape first and route those through
    ``MimiqRemoteBackend`` to get the polling + result-typing logic
    that's already there.
    """
    # 1. RemoteConnection-shaped: has submit + get_results.
    if callable(getattr(runner, "submit", None)) and callable(
        getattr(runner, "get_results", None)
    ):
        from mimiqcircuits.backends.remote import MimiqRemoteBackend

        return _backend_runner(MimiqRemoteBackend(runner))

    # 2. ``Backend`` instance: has execute returning QCSResults.
    if callable(getattr(runner, "execute", None)):
        return _backend_runner(runner)

    # 3. Plain callable, invoked once per circuit. MIMIQ-specific run
    #    options have no meaning here and are ignored.
    if callable(runner):
        def call(circuits, *, nsamples, seed, **_opts):
            return [
                runner(c, nsamples=nsamples, seed=seed) for c in circuits
            ]

        return call

    raise TypeError(
        f"runner must be a mimiqcircuits Backend, a RemoteConnection, "
        f"or a callable (circuit, *, nsamples, seed) -> QCSResults; "
        f"got {type(runner).__name__}"
    )


def _backend_runner(backend) -> Callable:
    """Wrap a MIMIQ backend so a list of circuits is submitted as one job.

    MIMIQ ``execute`` mirrors its input shape, so we always pass a list
    and normalise the result back to a list of ``QCSResults``.
    """
    def call(circuits, *, nsamples, seed, **opts):
        kwargs = {k: v for k, v in opts.items() if v is not None}
        results = backend.execute(
            list(circuits), nsamples=nsamples, seed=seed, **kwargs
        )
        return results if isinstance(results, list) else [results]

    return call


class MimiqBackend(BackendV2):
    """Qiskit BackendV2 powered by MIMIQ.

    Args:
        runner: A MIMIQ connection, a MIMIQ ``Backend`` instance, or a
            ``(circuit, *, nsamples, seed) -> QCSResults`` callable.
        name: Backend name reported to Qiskit. Defaults to ``"mimiq"``.
        num_qubits: Qubit count advertised on the ``Target``. The MIMIQ
            cloud handles much more — bump this when transpiling wide
            circuits against the backend.
        description: Human-readable backend description.

    Beyond ``shots`` and ``seed``, ``run`` accepts MIMIQ-specific options
    (``bonddim``, ``entdim``, ``mpscutoff``, ``mpsmethod``,
    ``mpotraversal``, ``timelimit``, ``noisemodel``, ``label``) which are
    forwarded to MIMIQ when set.
    """

    def __init__(
        self,
        runner: Any,
        *,
        name: str = "mimiq",
        num_qubits: int = _DEFAULT_NUM_QUBITS,
        description: str | None = None,
        provider=None,
    ):
        super().__init__(
            provider=provider,
            name=name,
            description=description or "MIMIQ cloud / local simulator",
            backend_version=__version__,
        )
        self._runner = _coerce_runner(runner)
        self._num_qubits = num_qubits
        self._target = _build_target(num_qubits)

    # ── BackendV2 surface ──────────────────────────────────────────────

    @property
    def target(self) -> Target:
        return self._target

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    @property
    def max_circuits(self) -> int | None:
        return None

    @classmethod
    def _default_options(cls) -> Options:
        opts = Options(shots=1024, seed=None)
        opts.update_options(**{k: None for k in _RUN_OPTION_KEYS})
        return opts

    # ── execution helpers (shared with the primitives) ─────────────────

    def _execute_batch(self, mimiq_circuits, *, shots, seed, **run_opts):
        """Submit a list of MIMIQ circuits as one job, blocking for the
        list of ``QCSResults``. Used by ``run`` and by the native
        Sampler/Estimator primitives.
        """
        return self._runner(
            list(mimiq_circuits), nsamples=shots, seed=seed, **run_opts
        )

    def _run_options(self, options: dict) -> dict:
        """Pick out the MIMIQ-specific run options that are set."""
        merged = {k: getattr(self.options, k, None) for k in _RUN_OPTION_KEYS}
        merged.update(
            {k: options[k] for k in _RUN_OPTION_KEYS if k in options}
        )
        return {k: v for k, v in merged.items() if v is not None}

    # ── run ────────────────────────────────────────────────────────────

    def run(self, run_input, **options) -> MimiqJob:
        from qiskit import QuantumCircuit  # local import keeps import cost low

        if isinstance(run_input, QuantumCircuit):
            qcs = [run_input]
        else:
            qcs = list(run_input)

        mimiq_circuits = [qiskit_to_mimiq(qc) for qc in qcs]

        shots = int(options.get("shots", self.options.shots))
        seed = options.get("seed", self.options.seed)
        run_opts = self._run_options(options)

        def work():
            return self._execute_batch(
                mimiq_circuits, shots=shots, seed=seed, **run_opts
            )

        job = MimiqJob(
            backend=self,
            qiskit_circuits=qcs,
            work=work,
            shots=shots,
        )
        job.submit()
        return job
