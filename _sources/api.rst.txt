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
generic ``BackendSamplerV2`` / ``BackendEstimatorV2``: the estimator reads
observables off the state instead of sampling them, and both batch a pub's
circuits into a single MIMIQ submission. See :ref:`estimation-methods`
for how the estimator handles a circuit that ends in an ensemble.

.. autoclass:: MimiqSamplerV2
   :members:

.. autoclass:: MimiqEstimatorV2
   :members:

Converters
----------

.. currentmodule:: mimiq_qiskit

See :doc:`conversion` for what each direction accepts.

.. autofunction:: qiskit_to_mimiq

.. autofunction:: mimiq_to_qiskit

.. autoexception:: mimiq_qiskit.converter.UnsupportedGateError

.. autofunction:: mimiq_qiskit.gate_map.supported_qiskit_names

Estimation internals
--------------------

The engine behind :class:`MimiqEstimatorV2`, and the Pauli-observable
helpers it rests on. These are the shared surface a provider-specific
estimator builds on, so that every MIMIQ estimator resolves methods,
averages trajectories, and reports metadata identically.

.. automodule:: mimiq_qiskit.estimation
   :members:

.. automodule:: mimiq_qiskit.observables
   :members:
