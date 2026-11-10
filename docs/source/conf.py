"""Sphinx configuration for the CRYPTEX documentation site.

https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import os
import sys
from datetime import datetime

# Read the version from the source tree; no install needed.
sys.path.insert(0, os.path.abspath("../../src"))

# -- Project information -----------------------------------------------------

project = "CRYPTEX"
author = "Basak Eraslan"
copyright = f"{datetime.now():%Y}, {author}"

try:  # keep the docs version in lockstep with the package
    from cryptex import __version__ as release
except Exception:  # pragma: no cover - docs can build without the deps
    release = "0.1.0.dev0"
version = release

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_nb",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "**/.ipynb_checkpoints", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

# -- MyST --------------------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "amsmath",
    "linkify",
    "substitution",
]
myst_heading_anchors = 3

# Pages are rendered as written; building the docs never needs screen data.
nb_execution_mode = "off"

# -- HTML output -------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = f"{project} {version}"
html_show_sourcelink = False

html_theme_options = {
    "github_url": "https://github.com/EraslanBas/CRYPTEX",
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_align": "left",
    "show_toc_level": 2,
    "show_nav_level": 1,
    "header_links_before_dropdown": 6,
    "use_edit_page_button": True,
    "show_prev_next": False,
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
}

html_context = {
    "github_user": "EraslanBas",
    "github_repo": "CRYPTEX",
    "github_version": "main",
    "doc_path": "docs/source",
    "default_mode": "auto",
}

# One page: no left navigation, the page's own contents on the right.
html_sidebars = {"**": []}
