# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import importlib.metadata
import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "hydrantic"
copyright = "2026, Sander Hummerich"
author = "Sander Hummerich"

# Do not `from importlib.metadata import version` here: that name would shadow Sphinx's own
# `version` config value with a function, which breaks the inventory dump.
try:
    release = importlib.metadata.version("hydrantic")
except importlib.metadata.PackageNotFoundError:  # docs built without installing the package
    release = "0.0.0+unknown"

version = release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# No napoleon: every docstring in this package is reST `:param:`/`:return:` style.
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
]

# The docs link out to Lightning/Hydra/pydantic rather than restating them.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://docs.pytorch.org/docs/stable", None),
    "lightning": ("https://lightning.ai/docs/pytorch/stable", None),
}

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_typehints = "description"
autodoc_member_order = "bysource"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
