"""
Live bridge to the NEST network definition.

simulate/models/BrainstemModel/params.py defines the spiking model the
reconstruction is built on: cell counts, convergences, pathway delays and
reversal potentials. This module reads it instead of duplicating the numbers.

The import needs care on two counts. params.py starts with an absolute import
(from utils.cochlea_utils import ...), so simulate/ itself has to be on
sys.path, not just the repo root. And utils.cochlea_utils pulls in brian2,
which costs a couple of seconds, so the instance is built once and cached.

simulate/ is read-only from here.
"""

import functools
import os
import sys

from recon_core.bootstrap import REPO_ROOT

# simulate/ is a sibling of reconstruction/, at the repository root.
SIMULATE_DIR = os.path.join(REPO_ROOT, 'simulate')
PARAMS_FILE = os.path.join(SIMULATE_DIR, 'models', 'BrainstemModel', 'params.py')


@functools.lru_cache(maxsize=1)
def parameters():
    """The NEST Parameters dataclass instance, with defaults as committed.

    Cached, so importing it from several modules costs one load.
    """
    if SIMULATE_DIR not in sys.path:
        sys.path.insert(0, SIMULATE_DIR)
    try:
        from models.BrainstemModel.params import Parameters
    except ImportError as exc:                              # pragma: no cover
        raise ImportError(
            f'cannot import the NEST network parameters from {PARAMS_FILE}.\n'
            'The reconstruction reads its population sizes, convergences and '
            'synaptic constants from there. Activate the sl_env environment '
            '(it needs brian2) and run from the repository root.'
        ) from exc
    return Parameters()


def population_sizes():
    """`{population name: neurons per side}` for the whole brainstem model."""
    pop = parameters().POP_NUM
    return {'SBC': pop.n_SBCs, 'GBC': pop.n_GBCs, 'MNTBC': pop.n_MNTBCs,
            'LNTBC': pop.n_LNTBCs, 'LSO': pop.n_LSOs, 'MSO': pop.n_MSOs,
            'SPN': pop.n_SPNs, 'ANF': parameters().n_ANFs}
