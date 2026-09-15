"""
The two directory anchors everything else is built from.

PACKAGE_ROOT is this reconstruction/ folder, holding the reconstruction code
and the NEURON models it loads. REPO_ROOT is the repository above it, which
also holds simulate/, RESULTS/ and data/.

This module cannot do the sys.path setup itself: running an entry point as
python reconstruction/ABR_reconstruction/main_abr.py puts that directory on
sys.path, not reconstruction/, so import recon_core fails first. Every entry
point therefore starts with

    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

Running via python -m or through reconstruction/main.py needs no bootstrap.
"""

import os
import sys

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(PACKAGE_ROOT)


def ensure_package_on_path():
    """Idempotently put `reconstruction/` on `sys.path`."""
    if PACKAGE_ROOT not in sys.path:
        sys.path.insert(0, PACKAGE_ROOT)
    return PACKAGE_ROOT
