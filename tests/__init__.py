# This file makes `tests/` a Python package so that absolute imports of the
# form `from tests.validation.shared import ...` resolve correctly on all
# platforms (Linux, macOS, Windows).  Without it, `ModuleNotFoundError:
# No module named 'tests.validation.shared'` is raised on Windows because
# strict module resolution requires every component of a dotted path to be
# a package.
