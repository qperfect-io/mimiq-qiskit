"""Run the manual's worked example, so it cannot drift from the code.

``docs/local_simulation.rst`` is a page of snippets a reader is expected to
paste and run, which makes it exactly the kind of prose that rots quietly
when an API moves. Executing it in order is the cheapest guard: if a
snippet stops working, this fails.

Only the code is checked here, not the ``code-block:: text`` outputs. Those
are real transcripts, but sample counts depend on the simulator's RNG and
pinning them would break on any upstream change to seeding.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("exaqt", reason="mimiq-exaqt is not installed")

PAGE = Path(__file__).resolve().parents[1] / "docs" / "local_simulation.rst"


def _python_blocks(rst: str) -> list[str]:
    """Return the dedented body of each ``code-block:: python`` in order."""
    blocks: list[str] = []
    lines = rst.split("\n")
    i = 0
    while i < len(lines):
        header = re.match(r"^(\s*)\.\. code-block:: (\w+)\s*$", lines[i])
        i += 1
        if not header:
            continue
        indent, language = header.group(1), header.group(2)

        body: list[str] = []
        while i < len(lines):
            line = lines[i]
            if not line.strip():
                body.append("")
                i += 1
            elif len(line) - len(line.lstrip()) > len(indent):
                body.append(line)
                i += 1
            else:
                break

        if language == "python":
            widths = [len(l) - len(l.lstrip()) for l in body if l.strip()]
            pad = min(widths) if widths else 0
            blocks.append(
                "\n".join(l[pad:] if l.strip() else "" for l in body)
            )
    return blocks


def test_worked_example_runs():
    """Every python snippet on the page, executed in order, in one scope.

    Later snippets build on names the earlier ones bind (``backend`` above
    all), which is how the page reads, so they share one namespace.
    """
    blocks = _python_blocks(PAGE.read_text())

    # The closing section switches to the cloud, which needs credentials.
    runnable = [b for b in blocks if "MimiqConnection" not in b]
    assert len(runnable) >= 6, "the page lost its code blocks"
    assert len(runnable) < len(blocks), "the cloud snippet went missing"

    scope: dict = {"__name__": "docs_local_simulation"}
    for number, block in enumerate(runnable, start=1):
        try:
            exec(compile(block, f"{PAGE.name}[block {number}]", "exec"), scope)
        except Exception as exc:
            pytest.fail(
                f"{PAGE.name} block {number} failed with "
                f"{type(exc).__name__}: {exc}\n\n{block}"
            )


def test_page_is_in_the_toctree():
    """A page missing from the toctree is a page nobody reads."""
    index = (PAGE.parent / "index.rst").read_text()
    assert PAGE.stem in index
