"""Conversion from :class:`mimiqcircuits.QCSResults` to a Qiskit
:class:`qiskit.result.Result`.

The remote MIMIQ service returns a list of ``QCSResults``, one per
submitted circuit. The mapping is:

- ``QCSResults.cstates`` (a list of ``bitarray``, one per shot) becomes
  the Qiskit ``counts`` histogram (``{hex: int}``).
- Each ``cstate`` is indexed by classical bit position: index ``i`` is
  the value of clbit ``i``, matching Qiskit's LSB-first convention for
  the hex integer key.
- Timings and fidelities are forwarded as ``metadata``, reachable via
  ``result.results[0].metadata``.
"""

from __future__ import annotations

from typing import Iterable

from qiskit.result import Result


def _cstate_to_hex(cstate: Iterable[int]) -> str:
    """Pack a MIMIQ classical state (bit i = clbit i) into a Qiskit hex
    key (bit i = bit i of the integer)."""
    value = 0
    for i, bit in enumerate(cstate):
        if bit:
            value |= 1 << i
    return hex(value)


def _histogram(cstates: Iterable[Iterable[int]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for cstate in cstates:
        key = _cstate_to_hex(cstate)
        counts[key] = counts.get(key, 0) + 1
    return counts


def qcsresults_to_qiskit_result(
    qcs_results,
    *,
    qiskit_circuits,
    backend_name: str,
    backend_version: str,
    job_id: str,
    shots: int,
) -> Result:
    """Bundle a list of ``QCSResults`` into a single Qiskit ``Result``.

    ``qiskit_circuits`` is the parallel list of source ``QuantumCircuit``
    instances. They supply ``memory_slots`` and ``creg_sizes`` so that
    ``result.get_counts(circuit)`` lookups work.
    """
    results = []
    for qcs, qc in zip(qcs_results, qiskit_circuits):
        counts = _histogram(qcs.cstates)
        # Per-shot record, in execution order. Sampler primitives
        # (``BackendSamplerV2``) reconstruct their bit arrays from ``memory``,
        # not ``counts``, so it must be present for them to work.
        memory = [_cstate_to_hex(cs) for cs in qcs.cstates]
        creg_sizes = [[reg.name, reg.size] for reg in qc.cregs]
        results.append({
            "shots": shots,
            "success": True,
            "header": {
                "name": qc.name,
                "memory_slots": qc.num_clbits,
                "creg_sizes": creg_sizes,
                "qreg_sizes": [[reg.name, reg.size] for reg in qc.qregs],
            },
            "data": {"counts": counts, "memory": memory},
            "metadata": {
                "simulator": getattr(qcs, "simulator", None),
                "simulator_version": getattr(qcs, "version", None),
                "timings": dict(getattr(qcs, "timings", {}) or {}),
                "fidelities": list(getattr(qcs, "fidelities", []) or []),
                "avggateerrors": list(
                    getattr(qcs, "avggateerrors", []) or []
                ),
            },
        })

    return Result.from_dict({
        "backend_name": backend_name,
        "backend_version": backend_version,
        "qobj_id": job_id,
        "job_id": job_id,
        "success": True,
        "results": results,
    })
