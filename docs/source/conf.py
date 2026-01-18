# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
from importlib.metadata import version as get_version
import subprocess
import itertools
import os


def get_all_git_tags():
    """
    Fetches all tags from the remote and lists all local tags in the current repository.
    """
    try:
        # First, fetch all tags from the remote to ensure local list is complete
        # This assumes a remote named 'origin' exists
        subprocess.run(
            ['git', 'fetch', '--tags'], check=True, capture_output=True, text=True
        )

        # Now, list all locally available tags
        result = subprocess.run(
            ['git', 'tag', '-l'], check=True, capture_output=True, text=True
        )

        # Split the output into a list of tags
        tags = result.stdout.strip().split('\n')

        # Filter out any empty strings that might result from the split
        return [tag for tag in tags if tag]

    except subprocess.CalledProcessError as e:
        print(f'An error occurred while running Git command: {e.stderr}')
        return []
    except FileNotFoundError:
        print(
            "The 'git' command was not found. Please ensure Git is installed and in your system's PATH."
        )
        return []


class VersionedTag:
    def __init__(self, tag: str):
        if tag[0] != 'v':
            raise ValueError
        self.tag = tag
        split = tag[1:].split('.')
        self.major = int(split[0])
        self.minor = int(split[1])
        self.patch = int(split[2])

    def __repr__(self):
        return f'VersionTag({self.tag})'

    def __str__(self):
        return self.tag


def get_latest_minor_versions(version_list):
    parsed_versions = []
    for tstr in version_list:
        try:
            parsed_versions.append(VersionedTag(tstr))
        # raises if its a development release
        except ValueError:
            pass
    # parse and sort all versions
    parsed_versions.sort(key=lambda x: [x.major, x.minor, x.patch])

    major_groups = itertools.groupby(parsed_versions, key=lambda v: v.major)

    latest_minors = []
    for major, versions_of_major in major_groups:
        # Sort again within the major group (already sorted overall, but good practice)
        sorted_minors = sorted(
            list(versions_of_major), key=lambda v: (v.minor, v.patch)
        )

        # 3. Group by minor version
        minor_groups = itertools.groupby(sorted_minors, key=lambda v: v.minor)

        for minor, versions_of_minor in minor_groups:
            # 4. Select the highest version (which is the last one in the sorted group)
            latest_patch_in_minor = list(versions_of_minor)[-1]
            latest_minors.append(str(latest_patch_in_minor))

    return latest_minors


# The full version, including alpha/beta/rc tags

project = 'Rocky Mountain Ellipse'
copyright = '2024, National Institute of Standards and Technology'
author = 'Daniel C. Gray, Zenn C. Roberts, Aaron M. Hagerstrom'

## check for warn on example fail env variable
# used for multiversioned documentation building


multiversioned = os.getenv('SPHINX_MULTIVERSIONED')

# default settings for a development build
release = get_version('rmellipse')
version = '.'.join(release.split('.')[:-1])
plot_gallery = True
latest_minors = get_latest_minor_versions(get_all_git_tags())

# multiversioned documentation then use the
# most up to date release tag
if multiversioned is not None:
    release = latest_minors[-1]
    version = '.'.join(release.split('.')[:-1])
    plot_gallery = False


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'autoapi.extension',
    'sphinx_gallery.gen_gallery',
    'numpydoc',
    'sphinx_click',
    'sphinx.ext.githubpages',
]
if multiversioned is not None:
    extensions.append('sphinx_multiversion')

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
numpydoc_validation_exclude = set(
    [
        r'\.undocumented_method$',
        r'\.__repr__$',
        r'\.__call_$',
    ]
)
autoapi_python_class_content = 'both'

autoapi_options = [
    'members',
    'undoc-members',
    # 'private-members',
    'show-inheritance',
    'show-module-summary',
    # 'special-members',
    'imported-members',
]


# -- SPHINX GALLERY OPTIONS --
sphinx_gallery_conf = {
    'examples_dirs': '../examples',  # path to your example scripts
    'gallery_dirs': 'auto_examples',  # path to where to save gallery generated output
    'within_subsection_order': 'FileNameSortKey',
    'ignore_pattern': '/_*',
    'run_stale_examples': True,
    'recommender': {'enable': True, 'n_examples': 5, 'min_df': 3, 'max_df': 0.9},
    # only execute examples on development builds for CI jobs
    'plot_gallery': plot_gallery,
}


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


spelling_word_list_filename = ['docs/source/spelling_wordlist.txt']
# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Whitelist pattern for tags (set to None to ignore all tags)
# white list the most recent minor tags we found earlier
smv_tag_whitelist = '|'.join([t.replace('.', '\.') for t in latest_minors])


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
    '**': ['globaltoc.html', 'sourcelink.html', 'searchbox.html', 'versioning.html'],
    'using/windows': ['windows-sidebar.html', 'searchbox.html'],
}
