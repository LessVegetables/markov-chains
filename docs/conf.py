"""Sphinx configuration for markov_chain API documentation."""

from __future__ import annotations

import os
import sys

# Paths: project root and package directory (same layout as ``python src/markov_chain/main.py``).
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_pkg = os.path.join(_root, "src", "markov_chain")
sys.path.insert(0, _pkg)

project = "markov_chain"
copyright = "2026, markov-chains contributors"
author = "markov-chains contributors"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "alabaster"
html_static_path = []

autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
