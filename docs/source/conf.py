# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

     

import sys
import os

# Ensure package root is importable for autodoc when building locally and on CI
DOCS_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(DOCS_DIR, '..', '..'))
PKG_ROOT = os.path.join(REPO_ROOT, 'wlclusters')
for path in [REPO_ROOT, PKG_ROOT]:
    if path not in sys.path:
        sys.path.insert(0, path)


project = 'wlclusters'
copyright = '2024, Loris Chappuis'
author = 'Loris Chappuis'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "numpydoc",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

try:
    import furo  # noqa: F401
    html_theme = 'furo'
except Exception:
    html_theme = 'alabaster'
html_static_path = ['_static']

# Mock heavy optional dependencies during autodoc to avoid import failures on CI
autodoc_mock_imports = [
    'pymc',
    'numpy',
    'scipy',
    'astropy',
    'tqdm',
]
