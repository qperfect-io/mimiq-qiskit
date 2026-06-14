"""Sphinx configuration for mimiq-qiskit."""

from __future__ import annotations

import os
import sys
from datetime import date

# Make the package importable without installing it.
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")),
)

project = "mimiq-qiskit"
author = "QPerfect"
copyright = f"{date.today().year}, QPerfect"

from mimiq_qiskit import __version__  # noqa: E402

release = __version__
version = ".".join(__version__.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
]

autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True
typehints_fully_qualified = False
always_document_param_types = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "qiskit": ("https://quantum.cloud.ibm.com/docs/api/qiskit", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = f"mimiq-qiskit {release}"
html_static_path = ["_static"] if os.path.isdir("_static") else []
