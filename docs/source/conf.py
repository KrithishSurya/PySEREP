# docs/source/conf.py
# Sphinx documentation configuration for pyserep

import os
import sys
sys.path.insert(0, os.path.abspath("../.."))

project   = "pyserep"
copyright = "2025, Krithish Surya"
author    = "Krithish Surya"
release   = "3.0.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",       # NumPy / Google docstrings
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",        # LaTeX math in docs
    "sphinx.ext.intersphinx",
    "numpydoc",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy":  ("https://numpy.org/doc/stable", None),
    "scipy":  ("https://docs.scipy.org/doc/scipy", None),
}

autodoc_default_options = {
    "members":         True,
    "undoc-members":   False,
    "show-inheritance":True,
}

numpydoc_show_class_members = False

templates_path   = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
}

html_static_path   = ["_static"]
html_title         = "pyserep documentation"
html_short_title   = "pyserep"
