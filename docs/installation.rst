Installation
============

From PyPI:

.. code-block:: bash

   pip install mimiq-qiskit

From source:

.. code-block:: bash

   git clone https://github.com/qperfect-io/mimiq-qiskit.git
   cd mimiq-qiskit
   uv sync

Runtime dependencies:

- Python >= 3.10
- `mimiqcircuits <https://pypi.org/project/mimiqcircuits/>`_ >= 0.26.7, < 0.28
- `qiskit <https://www.ibm.com/quantum/qiskit>`_ >= 2.0, < 3
- numpy >= 1.26, < 3

Each upper bound sits at the next version semantic versioning allows a
breaking change in, so an incompatible release fails to install rather
than failing at runtime. For a ``0.y.z`` dependency such as
``mimiqcircuits`` that boundary is the minor, since 0.x promises nothing
across minors — and it has been a real break: 0.26.7 changed
``GateCustom.matrix`` from an attribute back into a method in a *patch*
release.

The ``mimiqcircuits`` range covers two minors because the suite runs
against both. 0.27's one new operation, ``GateCustomDiagonal``, needs
nothing here: it is a unitary, so :func:`~mimiq_qiskit.mimiq_to_qiskit`
picks it up through the matrix fallback described in :doc:`conversion`.

Bounds widen only once the test suite has run against the new version, so
a release that raises one says so in the changelog. To try an unreleased
``mimiqcircuits`` against this package, override the constraint locally:

.. code-block:: bash

   uv add --editable ../mimiqcircuits-python

Running circuits locally
------------------------

The MIMIQ cloud is one execution target; the other is any local
``mimiqcircuits`` backend. `mimiq-exaqt
<https://pypi.org/project/mimiq-exaqt/>`_ is a state-vector simulator
published on PyPI, so it needs no credentials:

.. code-block:: bash

   pip install mimiq-exaqt

It is also what this project's own test suite simulates against, and is
installed by ``uv sync`` as part of the dev dependency group. Tests that
need it skip themselves when it is missing.
