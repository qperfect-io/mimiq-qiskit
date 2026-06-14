"""Native Qiskit V2 primitives backed by MIMIQ.

Qiskit ships generic ``BackendSamplerV2`` / ``BackendEstimatorV2`` that
wrap any ``BackendV2``. Its own documentation recommends a native
primitive when the provider can do better, which MIMIQ can:

- :class:`MimiqSamplerV2` reads MIMIQ's sampled bitstrings (``cstates``)
  directly into per-register :class:`~qiskit.primitives.containers.BitArray`,
  skipping the counts/hex round trip the generic sampler needs.
- :class:`MimiqEstimatorV2` evaluates observables exactly with MIMIQ's
  expectation-value engine (``Circuit.push_expval``) instead of
  estimating them from measurement samples. This avoids shot noise and
  scales to large circuits on the MPS backend.

Both batch every circuit in a pub (one per parameter binding) into a
single MIMIQ submission.
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
    PubResult,
    SamplerPubResult,
)
from qiskit.primitives.containers import BitArray, DataBin
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.primitives.containers.sampler_pub import SamplerPub

from mimiq_qiskit.backend import MimiqBackend
from mimiq_qiskit.converter import qiskit_to_mimiq


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
        for s in range(shots):
            cstate = cstates[s]
            value = 0
            for j, g in enumerate(clbit_indices):
                if g < len(cstate) and cstate[g]:
                    value |= 1 << j
            flat[i, s, :] = list(value.to_bytes(num_bytes, "big"))

    return BitArray(arr, reg_size)


def _observable_to_hamiltonian(obs: dict, num_qubits: int) -> mc.Hamiltonian:
    """Build a MIMIQ ``Hamiltonian`` from a Qiskit observable mapping.

    ``obs`` is ``{pauli_label: coefficient}`` as produced by
    :meth:`ObservablesArray.coerce`. Qiskit orders Pauli labels with
    qubit 0 on the right, so the label is reversed to MIMIQ's qubit-0-left
    convention before being attached to qubits ``0..num_qubits-1``.
    """
    ham = mc.Hamiltonian()
    for label, coeff in obs.items():
        mimiq_label = label[::-1]
        ham.push(
            float(np.real(coeff)),
            mc.PauliString(mimiq_label),
            *range(num_qubits),
        )
    return ham


def _obs_object_array(observables) -> np.ndarray:
    """Materialise an ``ObservablesArray`` as an object ndarray of dicts,
    so it broadcasts against the array of bound circuits with numpy."""
    out = np.empty(observables.shape, dtype=object)
    for idx in np.ndindex(observables.shape):
        out[idx] = observables[idx]
    return out


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
    """``BaseEstimatorV2`` that evaluates observables exactly on MIMIQ.

    Each observable is converted to a MIMIQ ``Hamiltonian`` and its
    expectation value is computed directly (no measurement sampling), so
    results carry no shot noise and standard deviations are reported as
    zero.

    Args:
        backend: A :class:`MimiqBackend`, or anything it can wrap.
        seed: Seed forwarded to MIMIQ. Defaults to the backend's.
        run_options: Extra MIMIQ run options merged over the backend's.
    """

    def __init__(self, backend, *, seed=None, run_options=None):
        self._backend = _as_backend(backend)
        self._seed = seed if seed is not None else self._backend.options.seed
        self._run_opts = {
            **self._backend._run_options({}),
            **(run_options or {}),
        }

    @property
    def precision(self) -> float:
        return 0.0

    def run(self, pubs: Iterable, *, precision: float | None = None) -> PrimitiveJob:
        coerced = [EstimatorPub.coerce(pub, precision) for pub in pubs]
        job = PrimitiveJob(self._run, coerced)
        job._submit()
        return job

    def _run(self, pubs) -> PrimitiveResult:
        return PrimitiveResult(
            [self._run_pub(pub) for pub in pubs],
            metadata={"version": 2},
        )

    def _run_pub(self, pub: EstimatorPub) -> PubResult:
        circuit = pub.circuit
        shape = pub.shape
        num_qubits = circuit.num_qubits

        bound = pub.parameter_values.bind_all(circuit)
        obs_arr = _obs_object_array(pub.observables)
        bound_b, obs_b = np.broadcast_arrays(bound, obs_arr)

        indices = list(np.ndindex(shape))
        mimiq_circuits = []
        for idx in indices:
            mc_circ = qiskit_to_mimiq(bound_b[idx])
            ham = _observable_to_hamiltonian(obs_b[idx], num_qubits)
            # push_expval sums the coefficient-scaled terms into the first
            # z-register, so the full observable lands in z[0] of an
            # otherwise z-register-free circuit, read back below.
            mc_circ.push_expval(ham, *range(num_qubits))
            mimiq_circuits.append(mc_circ)

        qcs_list = self._backend._execute_batch(
            mimiq_circuits, shots=1, seed=self._seed, **self._run_opts
        )

        evs = np.zeros(shape, dtype=float)
        for k, idx in enumerate(indices):
            zstates = qcs_list[k].zstates
            value = zstates[0][0] if zstates and zstates[0] else 0.0
            evs[idx] = float(np.real(value))

        databin = DataBin(evs=evs, stds=np.zeros(shape), shape=shape)
        return PubResult(databin, metadata={"precision": 0.0, "exact": True})
