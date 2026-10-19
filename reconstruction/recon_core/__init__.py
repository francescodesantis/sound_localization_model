"""
Shared framework for the LFP and ABR reconstruction pipelines.

LFP_reconstruction/ (near-field probe potentials) and ABR_reconstruction/
(far-field scalp potentials) are two views of the same simulated populations,
so their parameters, spike and geometry bookkeeping, HDF5 layout and signal
processing live here. The two packages hold only what is specific to them.

Import as a package from the repository root:

    from recon_core import params, paths, signal_utils

Entry-point scripts stay runnable directly; they put the repository root on
sys.path first, see recon_core.bootstrap.
"""
