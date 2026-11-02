"""
Directory anchors: PACKAGE_ROOT is reconstruction/ (code and NEURON models),
REPO_ROOT the repository above it (simulate/, RESULTS/, data/).

Entry points run as scripts put their own directory on sys.path, so each one
first inserts reconstruction/ itself before importing recon_core.
"""

import os
import sys

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO_ROOT = os.path.dirname(PACKAGE_ROOT)


def ensure_package_on_path():
    """Idempotently put `reconstruction/` on `sys.path`."""
    if PACKAGE_ROOT not in sys.path:
        sys.path.insert(0, PACKAGE_ROOT)
    return PACKAGE_ROOT
