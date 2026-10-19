"""
Base class shared by every nucleus's hybridLFPy population.

All five populations (MSO, LSO, AVCN GBC/SBC, MNTB and the calyx) answer the
same three questions:

get_all_SpCells      which presynaptic cells drive this postsynaptic cell?
insert_all_synapses  where on the morphology do their synapses land?
draw_rand_pos        where in the nucleus does each cell sit?

Only the answers differ, so only the answers are overridden. Subclasses
provide N_POST_TOTAL, a synapse placement rule and a nucleus geometry.

The order of random draws is part of the contract. Cell positions and synapse
placement come from the global NumPy generator, so draw order decides the
result. rejection_sample_ellipse therefore takes an explicit sample_order: the
LSO samples x, z, y while the others sample x, y, z.
"""

import numpy as np
from hybridLFPy.population import Population


class ReconstructionPopulation(Population):
    """hybridLFPy Population with the pipeline's shared wiring and placement."""

    #: Synapse parameters per presynaptic population name (from recon_core.params).
    PER_POP_SYN = {}

    #: Cells per side in the NEST model, the tonotopic axis this population maps onto.
    N_POST_TOTAL = None

    #: Skip layers that received no synapses. The MSO alone processes them, because
    #: its non-SBC inputs use hybridLFPy's own layer indices unmodified.
    SKIP_EMPTY_LAYERS = True

    #: Synapses per presynaptic population when n_syn_per_pop does not name one.
    DEFAULT_N_SRC = 1

    def __init__(self, n_syn_per_pop=None, per_pop_syn=None, **kwargs):
        self.n_syn_per_pop = n_syn_per_pop or {}
        # Instance-level override so one class can serve two cell types that
        # differ only in their synapses (the AVCN globular and spherical cells).
        self.per_pop_syn = per_pop_syn if per_pop_syn is not None else self.PER_POP_SYN
        super().__init__(**kwargs)

    # -- presynaptic assignment ---------------------------------------------
    def post_index(self, cellindex):
        """Tonotopic index in the full NEST population of simulated cell i.

        A run with --n-cells 200 stands in for all N_POST_TOTAL cells, so cell i
        of the sample represents the cell at this position along the tonotopic
        axis and must receive the inputs that cell would have received.
        """
        n_sim = self.POPULATION_SIZE
        if n_sim <= 1:
            return 0
        return int(round(cellindex * (self.N_POST_TOTAL - 1) / (n_sim - 1)))

    def presynaptic_span(self):
        """Denominator mapping a tonotopic index onto the presynaptic population."""
        return self.N_POST_TOTAL

    def get_all_SpCells(self):
        """Tonotopic presynaptic assignment, mirroring the NEST x_to_one connector.

        Each postsynaptic cell draws its inputs from a contiguous window of the
        presynaptic population, positioned by its own tonotopic index, so a cell
        tuned to 1 kHz is driven by 1 kHz fibres as in the NEST run being
        replayed.
        """
        span = max(self.presynaptic_span() - 1, 1)
        SpCells = {}

        for cellindex in self.RANK_CELLINDICES:
            post_idx = self.post_index(cellindex)
            SpCells[cellindex] = {}

            for X in self.X:
                nodes = self.networkSim.nodes[X]
                n_pre = len(nodes)
                n_src = self.n_syn_per_pop.get(X, self.DEFAULT_N_SRC)

                start = min(int(round(post_idx * (n_pre - n_src) / span)), n_pre - n_src)
                window = nodes[start:start + n_src]

                # Hand the window out layer by layer, matching the synIdx structure.
                per_layer, used = [], 0
                for compartments in self.synIdx[cellindex][X]:
                    size = len(compartments)
                    per_layer.append(window[used:used + size].astype('int32')
                                     if size else np.array([], dtype='int32'))
                    used += size
                SpCells[cellindex][X] = per_layer

        return SpCells

    # -- synapse placement --------------------------------------------------
    def select_synapse_idx(self, cell, pop_type, idx, layer):
        """Where this input's synapses go on the morphology.

        Default: hybridLFPy's own layer-based indices, unchanged. Subclasses
        override to place by section name instead, which lets a nucleus whose
        somas spread along the depth axis still target dendrites and somata
        correctly (a fixed depth band would miss most cells).
        """
        return idx

    def insert_all_synapses(self, cellindex, cell):
        for X in self.X:
            pop_type = X.rsplit('_', 1)[0]
            for layer in range(len(self.synIdx[cellindex][X])):
                idx = self.synIdx[cellindex][X][layer]
                if self.SKIP_EMPTY_LAYERS and len(idx) == 0:
                    continue
                syn_delays = (self.synDelays[cellindex][X][layer]
                              if self.synDelays is not None else None)
                self.insert_synapses(
                    cell=cell,
                    cellindex=cellindex,
                    synParams=self.per_pop_syn[pop_type].copy(),
                    idx=self.select_synapse_idx(cell, pop_type, idx, layer),
                    X=X,
                    SpCell=self.SpCells[cellindex][X][layer],
                    synDelays=syn_delays,
                )

    # -- cell placement -----------------------------------------------------
    def rejection_sample_ellipse(self, extents, sample_order, ellipse_axes,
                                 min_cell_interdist, sort_axis):
        """Uniformly fill an elliptic cylinder, then enforce a minimum spacing.

        extents      {axis: (lo, hi)} bounding box, one entry per axis
        sample_order the order the axes consume random numbers, see the module
                     docstring; changing it changes every drawn position
        ellipse_axes the two axes forming the elliptic cross-section; the third
                     is the cylinder's long axis, bounded by its extent alone
        sort_axis    somas are returned ordered along this axis, so cell index
                     order is tonotopic order (what post_index assumes)

        Returns hybridLFPy's [{'x':..., 'y':..., 'z':...}, ...].
        """
        n_cells = self.POPULATION_SIZE
        centres = {a: 0.5 * (lo + hi) for a, (lo, hi) in extents.items()}
        radii = {a: 0.5 * (hi - lo) for a, (lo, hi) in extents.items()}

        def draw(n):
            drawn = {}
            for axis in sample_order:          # order matters, see docstring
                lo, hi = extents[axis]
                # Written as (u - 0.5) * width + centre, not u * width + lo: the
                # two are equal in exact arithmetic but round differently, and
                # the difference propagates through the rejection loop into
                # which cells get resampled. This form reproduces the per-nucleus
                # samplers it replaced bit for bit.
                drawn[axis] = (np.random.rand(n) - 0.5) * (hi - lo) + centres[axis]
            return drawn

        def outside(pos):
            total = 0.0
            for axis in ellipse_axes:
                r = radii[axis]
                if r == 0:
                    continue
                total = total + ((pos[axis] - centres[axis]) / r) ** 2
            return np.where(total > 1)[0]

        pos = draw(n_cells)

        def refill_outside():
            bad = outside(pos)
            while len(bad):
                fresh = draw(len(bad))
                for axis in pos:
                    pos[axis][bad] = fresh[axis]
                bad = outside(pos)

        refill_outside()

        def crowded():
            return np.where(self.calc_min_cell_interdist(pos['x'], pos['y'], pos['z'])
                            < min_cell_interdist)[0]

        close = crowded()
        while len(close):
            fresh = draw(len(close))
            for axis in pos:
                pos[axis][close] = fresh[axis]
            refill_outside()
            close = crowded()

        soma_pos = [{'x': pos['x'][i], 'y': pos['y'][i], 'z': pos['z'][i]}
                    for i in range(n_cells)]
        soma_pos.sort(key=lambda p: p[sort_axis])
        return soma_pos
