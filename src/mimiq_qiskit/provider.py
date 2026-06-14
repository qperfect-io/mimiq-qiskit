"""Lightweight provider that enumerates :class:`MimiqBackend` instances
for a given MIMIQ connection or local runner.

Qiskit 2.x removed the abstract ``Provider`` class; a provider is now a
plain Python object exposing ``backends()`` and ``get_backend(name)``.
:class:`MimiqProvider` follows that pattern.
"""

from __future__ import annotations

from typing import Any

from mimiq_qiskit.backend import MimiqBackend


class MimiqProvider:
    """Expose the MIMIQ backend for a single connection or runner.

    Example::

        from mimiqlink import MimiqConnection
        from mimiq_qiskit import MimiqProvider

        conn = MimiqConnection(); conn.connect()
        provider = MimiqProvider(conn)
        backend = provider.get_backend("mimiq")
    """

    def __init__(self, runner: Any, *, num_qubits: int = 64):
        self._runner = runner
        self._backends = {
            "mimiq": MimiqBackend(
                runner, name="mimiq", num_qubits=num_qubits, provider=self
            ),
        }

    def backends(self, name: str | None = None) -> list[MimiqBackend]:
        if name is None:
            return list(self._backends.values())
        return [b for n, b in self._backends.items() if n == name]

    def get_backend(self, name: str = "mimiq") -> MimiqBackend:
        try:
            return self._backends[name]
        except KeyError as exc:
            raise ValueError(
                f"unknown backend {name!r}; available: "
                f"{sorted(self._backends)}"
            ) from exc
