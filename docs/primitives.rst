Primitives
==========

Qiskit's V2 primitives are the standard interface for sampling and
expectation-value workloads. ``mimiq-qiskit`` ships native
implementations that talk to MIMIQ directly, rather than relying on
Qiskit's generic ``BackendSamplerV2`` / ``BackendEstimatorV2`` wrappers.

The native primitives differ from the generic wrappers in two ways:

- :class:`~mimiq_qiskit.MimiqEstimatorV2` computes expectation values
  exactly using MIMIQ's expectation-value engine. The generic estimator
  estimates them by sampling in rotated measurement bases; MIMIQ
  evaluates :math:`\langle \psi | H | \psi \rangle` directly, so there is
  no shot noise and the reported standard deviation is zero.
- Both primitives submit every circuit in a pub (one per parameter
  binding) as a single MIMIQ job, instead of one round trip per circuit.

Sampling
--------

.. code-block:: python

   from qiskit import QuantumCircuit
   from mimiqlink import MimiqConnection
   from mimiq_qiskit import MimiqBackend, MimiqSamplerV2

   conn = MimiqConnection(); conn.connect()
   sampler = MimiqSamplerV2(MimiqBackend(conn))

   qc = QuantumCircuit(2)
   qc.h(0)
   qc.cx(0, 1)
   qc.measure_all()

   result = sampler.run([qc], shots=1000).result()
   # Per-register bit arrays; ``measure_all`` writes the "meas" register.
   print(result[0].data.meas.get_counts())

Estimating expectation values
-----------------------------

.. code-block:: python

   from qiskit import QuantumCircuit
   from qiskit.quantum_info import SparsePauliOp
   from mimiq_qiskit import MimiqBackend, MimiqEstimatorV2

   estimator = MimiqEstimatorV2(MimiqBackend(conn))

   qc = QuantumCircuit(2)
   qc.h(0)
   qc.cx(0, 1)

   observable = SparsePauliOp(["ZZ", "XX"], [1.0, 0.5])
   result = estimator.run([(qc, observable)]).result()
   print(result[0].data.evs)   # exact expectation value

Parameterised circuits broadcast the usual way: pass an array of bindings
as the third pub element, and the result's ``evs`` / bit arrays take the
broadcast shape.
