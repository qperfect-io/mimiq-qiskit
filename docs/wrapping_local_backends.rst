Wrapping any MIMIQ backend
==========================

``MimiqBackend`` wraps anything that produces a MIMIQ ``QCSResults``.
There are three accepted shapes for the constructor argument:

1. **A MIMIQ ``RemoteConnection``** (e.g. ``MimiqConnection``). Wrapped
   internally in ``mimiqcircuits.backends.MimiqRemoteBackend`` so the
   polling and result post-processing already in mimiqcircuits is
   reused.

2. **A MIMIQ ``Backend`` instance**, any class inheriting from
   ``mimiqcircuits.backends.Backend``. Used as is via its ``execute``
   method.

3. **A plain callable** ``(circuit, *, nsamples, seed) -> QCSResults``,
   an escape hatch for custom executors or tests.

Example: wrapping a local in-process simulator
----------------------------------------------

.. code-block:: python

   from mimiq_qiskit import MimiqBackend
   from somewhere import MyLocalBackend  # subclass of mimiqcircuits.backends.Backend

   backend = MimiqBackend(MyLocalBackend(), name="my-local")

Example: a stub for unit tests
------------------------------

.. code-block:: python

   from bitarray import bitarray
   from mimiqcircuits import QCSResults
   from mimiq_qiskit import MimiqBackend

   def runner(circuit, *, nsamples, seed):
       return QCSResults(cstates=[bitarray("0" * circuit.num_bits())] * nsamples)

   backend = MimiqBackend(runner)

The Target advertised to Qiskit lists every standard gate that the
converter recognises. ``transpile(qc, backend=backend)`` will decompose
unsupported gates into that set before they reach the converter.
