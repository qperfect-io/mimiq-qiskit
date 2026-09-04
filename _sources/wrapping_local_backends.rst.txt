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

``mimiq-exaqt`` is a state-vector simulator on PyPI whose ``ExaqtQCS`` is
a ``mimiqcircuits`` ``LocalBackend``, so it needs no cloud credentials:

.. code-block:: python

   from exaqt import ExaqtQCS
   from mimiq_qiskit import MimiqBackend

   backend = MimiqBackend(ExaqtQCS(), name="exaqt", num_qubits=24)

Any other class inheriting from ``mimiqcircuits.backends.Backend`` wraps
the same way. :doc:`local_simulation` works this example through the whole
API — counts, both primitives, mid-circuit measurement, transpilation, and
a check against Qiskit's own reference.

Run options are not portable across backends
--------------------------------------------

MIMIQ backends do not all take the same options. The cloud accepts every
knob; a local simulator names only what it implements, so
``backend.run(qc, bonddim=64)`` against ``ExaqtQCS`` raises a
``ValueError`` naming the backend and the option rather than failing
somewhere inside mimiqcircuits. Set MPS and job options only on backends
that have them.

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
