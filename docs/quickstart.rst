Quickstart
==========

Cloud execution
---------------

Authenticate against the MIMIQ cloud, wrap the connection in a
:class:`~mimiq_qiskit.MimiqBackend`, and run a familiar
``QuantumCircuit``:

.. code-block:: python

   from qiskit import QuantumCircuit
   from mimiqlink import MimiqConnection
   from mimiq_qiskit import MimiqBackend

   conn = MimiqConnection()
   conn.connect()

   backend = MimiqBackend(conn)

   qc = QuantumCircuit(2, 2)
   qc.h(0)
   qc.cx(0, 1)
   qc.measure([0, 1], [0, 1])

   job = backend.run(qc, shots=1000)
   counts = job.result().get_counts()

The job is executed in a background thread, so ``backend.run`` returns
immediately. Block for completion by calling ``job.result()``.

Provider helper
---------------

When you want a Qiskit-style provider entry point:

.. code-block:: python

   from mimiq_qiskit import MimiqProvider

   provider = MimiqProvider(conn)
   backend = provider.get_backend("mimiq")

Batching
--------

Pass a list of circuits to run them in one submission:

.. code-block:: python

   result = backend.run([qc_a, qc_b, qc_c], shots=2000).result()
   counts_a = result.get_counts(0)
