"""
MPI and NEURON mechanism boilerplate.

Every pipeline runs the same way under mpiexec: rank 0 extracts spikes from the
.pic (a slow, single-writer step), broadcasts the resulting metadata, and all
ranks then simulate their share of the population. That handshake and the
repeated-mechanism-load handling live here instead of in every entry point.
"""

import neuron
from mpi4py import MPI

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()


def is_root():
    return RANK == 0


def load_mechanisms(*directories):
    """Load compiled NEURON mechanisms, tolerating repeated loads.

    Importing two model packages that share a mechanism (MNTB reuses the MSO
    klt/kht/ih) makes NEURON raise on the second load. That specific error is
    benign and is the only one swallowed here.
    """
    for directory in directories:
        try:
            neuron.load_mechanisms(directory)
        except RuntimeError as exc:
            if 'already exists' not in str(exc):
                raise


def set_temperature(celsius):
    """Set NEURON's global simulation temperature.

    Call it wherever mechanisms are loaded: temperature-scaled mechanisms read
    celsius when their rates are computed, so it has to be right before the
    first finitialize. NEURON defaults to 6.3 degC, which is not a body
    temperature and not what these channels were fitted at.
    """
    neuron.h.celsius = float(celsius)
    return neuron.h.celsius


def broadcast_from_root(compute):
    """Run compute() on rank 0 only, broadcast its result, then barrier.

    Used for spike extraction: the .pic is hundreds of MB and the GDF files are
    a shared cache, so exactly one rank may produce them.
    """
    payload = compute() if is_root() else None
    payload = COMM.bcast(payload, root=0)
    COMM.Barrier()
    return payload


def reduce_sum(local, root=0):
    """Element-wise sum of an array across ranks; non-root ranks get None.

    The population dipole is a sum over cells split across ranks, so the total
    only exists after this reduction.
    """
    import numpy as np

    local = np.ascontiguousarray(local, dtype=float)
    total = np.zeros_like(local) if RANK == root else None
    COMM.Reduce(local, total, op=MPI.SUM, root=root)
    return total
