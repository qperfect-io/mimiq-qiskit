"""Native Qiskit V2 primitives backed by MIMIQ.

Qiskit ships generic ``BackendSamplerV2`` / ``BackendEstimatorV2`` that
wrap any ``BackendV2``. Its own documentation recommends a native
primitive when the provider can do better, which MIMIQ can:

- :class:`MimiqSamplerV2` reads MIMIQ's sampled bitstrings (``cstates``)
  directly into per-register :class:`~qiskit.primitives.containers.BitArray`,
  skipping the counts/hex round trip the generic sampler needs.
- :class:`MimiqEstimatorV2` reads each Pauli term straight off the
  simulator state instead of estimating it from measurement samples, so a
  deterministic circuit carries no shot noise at any term weight. A
  circuit that ends in an ensemble rather than a state (mid-circuit
  measurement, reset, noise) is averaged over trajectories, and a
  ``shots`` budget switches it to hardware-style sampled estimation. See
  :mod:`mimiq_qiskit.estimation` for the three methods.

Both batch every circuit in a pub into a single MIMIQ submission, and the
estimator shares one submission between observables that were asked for at
the same parameter binding.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

import mimiqcircuits as mc
from qiskit.primitives import (
    BaseEstimatorV2,
    BaseSamplerV2,
    PrimitiveJob,
    PrimitiveResult,
    SamplerPubResult,
)
from qiskit.primitives.containers import BitArray, DataBin
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.primitives.containers.sampler_pub import SamplerPub

from mimiq_qiskit.backend import MimiqBackend
from mimiq_qiskit.converter import qiskit_to_mimiq
from mimiq_qiskit.estimation import EstimatorConfig, estimate_pub


def _as_backend(backend) -> MimiqBackend:
    """Accept a ``MimiqBackend`` or anything ``MimiqBackend`` can wrap."""
    if isinstance(backend, MimiqBackend):
        return backend
    return MimiqBackend(backend)


def _register_bitarray(qcs_list, shape, clbit_indices, reg_size, shots):
    """Pack one classical register's samples into a ``BitArray``.

    ``qcs_list`` holds one ``QCSResults`` per flattened parameter binding
    (in ``np.ndindex(shape)`` order). ``clbit_indices[j]`` is the global
    MIMIQ classical-bit index feeding bit ``j`` of the register (bit 0 is
    the least significant), matching Qiskit's LSB-first register packing.
    """
    num_bytes = (reg_size + 7) // 8
    arr = np.zeros(tuple(shape) + (shots, num_bytes), dtype=np.uint8)
    flat = arr.reshape((-1, shots, num_bytes))

    for i, qcs in enumerate(qcs_list):
        cstates = qcs.cstates
        # A BitArray has a fixed shot axis, so a runner that returns a
        # different number of samples than asked for cannot be packed into
        # it. Say so, rather than reading off the end of the list.
        if len(cstates) < shots:
            raise ValueError(
                f"MIMIQ returned {len(cstates)} samples for circuit {i} "
                f"but {shots} shots were requested; the sampler cannot "
                "pack a short result"
            )
        for s in range(shots):
            cstate = cstates[s]
            value = 0
            for j, g in enumerate(clbit_indices):
                if g < len(cstate) and cstate[g]:
                    value |= 1 << j
            flat[i, s, :] = list(value.to_bytes(num_bytes, "big"))

    return BitArray(arr, reg_size)


class MimiqSamplerV2(BaseSamplerV2):
    """``BaseSamplerV2`` that samples bitstrings on MIMIQ.

    Args:
        backend: A :class:`MimiqBackend`, or a connection / MIMIQ backend
            / runner that :class:`MimiqBackend` can wrap.
        default_shots: Shots used for pubs that don't specify their own.
        seed: Seed forwarded to MIMIQ. Defaults to the backend's.
        run_options: Extra MIMIQ run options (``noisemodel``, ``bonddim``,
            …) merged over the backend's.
    """

    def __init__(
        self, backend, *, default_shots: int = 1024, seed=None, run_options=None
    ):
        self._backend = _as_backend(backend)
        self._default_shots = default_shots
        self._seed = seed if seed is not None else self._backend.options.seed
        self._run_opts = {
            **self._backend._run_options({}),
            **(run_options or {}),
        }

    @property
    def default_shots(self) -> int:
        return self._default_shots

    def run(self, pubs: Iterable, *, shots: int | None = None) -> PrimitiveJob:
        coerced = [
            SamplerPub.coerce(pub, shots or self._default_shots)
            for pub in pubs
        ]
        job = PrimitiveJob(self._run, coerced)
        job._submit()
        return job

    def _run(self, pubs) -> PrimitiveResult:
        return PrimitiveResult(
            [self._run_pub(pub) for pub in pubs],
            metadata={"version": 2},
        )

    def _run_pub(self, pub: SamplerPub) -> SamplerPubResult:
        circuit = pub.circuit
        shots = pub.shots
        shape = pub.shape

        bound = pub.parameter_values.bind_all(circuit)
        indices = list(np.ndindex(shape))
        mimiq_circuits = [qiskit_to_mimiq(bound[idx]) for idx in indices]

        qcs_list = self._backend._execute_batch(
            mimiq_circuits, shots=shots, seed=self._seed, **self._run_opts
        )

        data = {}
        for reg in circuit.cregs:
            clbit_indices = [circuit.find_bit(b).index for b in reg]
            data[reg.name] = _register_bitarray(
                qcs_list, shape, clbit_indices, reg.size, shots
            )

        databin = DataBin(**data, shape=shape)
        return SamplerPubResult(databin, metadata={"shots": shots})


class MimiqEstimatorV2(BaseEstimatorV2):
    """``BaseEstimatorV2`` that evaluates observables on MIMIQ.

    Each Pauli term is evaluated on the simulator state rather than sampled,
    so a circuit that ends in a definite state gives an exact value at any
    term weight and reports a standard error of zero.

    A circuit that ends in an *ensemble* has no single exact value: a
    mid-circuit measurement, a reset, a noise channel, or a server-side
    ``noisemodel`` makes MIMIQ re-evolve the circuit once per shot, and each
    trajectory has its own expectation value. The average over trajectories
    is the density-matrix value, so that is what this reports, with the
    sample standard error in ``stds``. Reading one trajectory would be
    unbiased but as noisy as the observable's range, so a stochastic circuit
    with no budget raises rather than returning it. A ``shots`` budget
    switches to hardware-style estimation from measurements in rotated
    bases.

    :mod:`mimiq_qiskit.estimation` documents the three methods and how
    ``method="auto"`` chooses between them.

    Args:
        backend: A :class:`MimiqBackend`, or anything it can wrap.
        method: ``"auto"`` (default), ``"exact"``, ``"trajectories"``, or
            ``"shots"``.
        trajectories: Trajectories to average for a stochastic circuit.
            ``None`` sizes it from ``precision``. Ignored for a deterministic
            circuit, which has one answer.
        shots: Shots per measurement basis. Giving this selects sampled
            estimation; ``None`` sizes it from ``precision`` when
            ``method="shots"``.
        emulate_shot_noise: Add Gaussian noise of width ``precision`` to an
            otherwise exact value, as Qiskit's ``StatevectorEstimator`` does.
            One evolution instead of a shot budget, for code that wants to
            see plausible shot noise without paying for it.
        default_precision: Precision for ``run`` calls and pubs that do not
            carry their own. ``0.0``, meaning "as exact as the simulator
            gets".
        seed: Seed forwarded to MIMIQ, and to the ``emulate_shot_noise``
            generator. Defaults to the backend's.
        run_options: Extra MIMIQ run options merged over the backend's.

    Result metadata reports ``method``, whether the value is ``exact``,
    whether the run was ``stochastic``, the ``trajectories`` or ``shots``
    spent, the ``target_precision``, and ``min_fidelity``: the lowest
    simulator fidelity behind the pub, which on an MPS backend is the
    truncation error that averaging cannot remove.
    """

    def __init__(
        self,
        backend,
        *,
        method: str = "auto",
        trajectories: int | None = None,
        shots: int | None = None,
        emulate_shot_noise: bool = False,
        default_precision: float = 0.0,
        seed=None,
        run_options=None,
    ):
        self._backend = _as_backend(backend)
        self._seed = seed if seed is not None else self._backend.options.seed
        self._run_opts = {
            **self._backend._run_options({}),
            **(run_options or {}),
        }
        self._default_precision = default_precision
        self._config = EstimatorConfig(
            method=method,
            trajectories=trajectories,
            shots=shots,
            emulate_shot_noise=emulate_shot_noise,
            default_precision=default_precision,
            seed=self._seed,
            # A noise model applied by the runner never appears in the
            # circuit, so nothing else can tell that the run is stochastic.
            assume_stochastic=self._run_opts.get("noisemodel") is not None,
        )

    @property
    def default_precision(self) -> float:
        """Precision used when ``run`` is called without one."""
        return self._default_precision

    def run(self, pubs: Iterable, *, precision: float | None = None) -> PrimitiveJob:
        """Estimate every pub and return the job carrying the results.

        ``precision`` overrides :attr:`default_precision` for this call. A
        positive value sizes the trajectory or shot budget as
        ``ceil(1/precision**2)`` where one is needed.
        """
        if precision is None:
            precision = self._default_precision
        coerced = [EstimatorPub.coerce(pub, precision) for pub in pubs]
        job = PrimitiveJob(self._run, coerced)
        job._submit()
        return job

    def _run(self, pubs) -> PrimitiveResult:
        return PrimitiveResult(
            [estimate_pub(pub, self._config, self._submit) for pub in pubs],
            metadata={"version": 2},
        )

    def _submit(self, mimiq_circuits, nsamples):
        """Run one batch of MIMIQ circuits for the shared estimation engine."""
        return self._backend._execute_batch(
            mimiq_circuits, shots=nsamples, seed=self._seed, **self._run_opts
        )
