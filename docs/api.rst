API reference
=============

.. currentmodule:: mimiq_qiskit

Backend
-------

.. autoclass:: MimiqBackend
   :members:

.. autoclass:: MimiqJob
   :members:

.. autoclass:: MimiqProvider
   :members:

Primitives
----------

Native Qiskit V2 primitives backed by MIMIQ. Prefer these over Qiskit's
generic ``BackendSamplerV2`` / ``BackendEstimatorV2``: the estimator
evaluates observables exactly (no shot noise) and both batch a pub's
circuits into a single MIMIQ submission.

.. autoclass:: MimiqSamplerV2
   :members:

.. autoclass:: MimiqEstimatorV2
   :members:

Converters
----------

See :doc:`conversion` for what each direction accepts.

.. autofunction:: qiskit_to_mimiq

.. autofunction:: mimiq_to_qiskit

.. autoexception:: mimiq_qiskit.converter.UnsupportedGateError

.. autofunction:: mimiq_qiskit.gate_map.supported_qiskit_names
