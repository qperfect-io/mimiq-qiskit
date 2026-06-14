"""Qiskit BackendV2 bridge for MIMIQ.

Public surface:

- :class:`MimiqBackend` — Qiskit ``BackendV2`` that runs circuits on a
  MIMIQ cloud connection or any local ``mimiqcircuits`` backend.
- :class:`MimiqJob` — Qiskit ``JobV1`` returned by ``backend.run``.
- :class:`MimiqProvider` — convenience entry point that lists the
  available MIMIQ backends for a given connection.
- :class:`MimiqSamplerV2`, :class:`MimiqEstimatorV2` — native Qiskit V2
  primitives that sample / estimate directly on MIMIQ.
- :func:`qiskit_to_mimiq`, :func:`mimiq_to_qiskit` — low-level circuit
  converters, exposed for users who want to bypass the backend wrapper.
"""

from mimiq_qiskit.__version__ import __version__
from mimiq_qiskit.backend import MimiqBackend
from mimiq_qiskit.converter import mimiq_to_qiskit, qiskit_to_mimiq
from mimiq_qiskit.job import MimiqJob
from mimiq_qiskit.primitives import MimiqEstimatorV2, MimiqSamplerV2
from mimiq_qiskit.provider import MimiqProvider

__all__ = [
    "MimiqBackend",
    "MimiqEstimatorV2",
    "MimiqJob",
    "MimiqProvider",
    "MimiqSamplerV2",
    "__version__",
    "mimiq_to_qiskit",
    "qiskit_to_mimiq",
]
