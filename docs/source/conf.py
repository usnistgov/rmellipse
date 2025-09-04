# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
from pathlib import Path
from importlib.metadata import version as get_version
import os
# The full version, including alpha/beta/rc tags

project = 'Rocky Mountain Ellipse'
copyright = '2024, National Institute of Standards and Technology'
author = 'Daniel C. Gray, Zenn C. Roberts, Aaron M. Hagerstrom'
release = get_version('rmellipse')


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'autoapi.extension',
    'sphinx_gallery.gen_gallery',
    'numpydoc',
    "sphinx_click",
    "sphinx.ext.githubpages",
    "sphinx_multiversion",
    "sphinx.ext.viewcode"
]

# napoleon settings
# napoleon_include_init_with_doc = True
# napoleon_include_private_with_doc = True
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']


autoapi_dirs = ['../../src']
autoapi_ignore = [
    '*migrations*',
    '*_archive*',

]
# numpydoc_validation_checks = {"all","GL08"}
numpydoc_validation_exclude = set([
    r'\.undocumented_method$',
    r'\.__repr__$',
    r'\.__call_$',
])
autoapi_python_class_content = 'both'

autoapi_options = [
    'members',
    'undoc-members',
    # 'private-members',
    'show-inheritance',
    'show-module-summary',
    # 'special-members',
    'imported-members'
]

# -- SPHINX GALLERY OPTIONS --
sphinx_gallery_conf = {
    'examples_dirs': '../examples',   # path to your example scripts
    'gallery_dirs': 'auto_examples',  # path to where to save gallery generated output
    'within_subsection_order': 'FileNameSortKey',
    'ignore_pattern': '/_*',
    'run_stale_examples': True
}
import shutil


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'nature'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Whitelist pattern for tags (set to None to ignore all tags)
smv_tag_whitelist = r'^v\d+\.\d+\.\d+$'

# Whitelist pattern for branches (set to None to ignore all branches)
smv_branch_whitelist = r'^stable$|^development$'

# Whitelist pattern for remotes (set to None to use local branches only)
smv_remote_whitelist = None

# Pattern for released versions
smv_released_pattern = r'^tags/.*$'

# Format for versioned output directories inside the build directory
smv_outputdir_format = '{ref.name}'

# Determines whether remote or local git branches/tags are preferred if their output dirs conflict
smv_prefer_remote_refs = False

html_sidebars = {
   '**': ['globaltoc.html', 'sourcelink.html', 'searchbox.html','versioning.html'],
   'using/windows': ['windows-sidebar.html', 'searchbox.html'],
}