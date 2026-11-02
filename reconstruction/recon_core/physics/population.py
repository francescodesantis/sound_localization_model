"""Base class shared by every nucleus's hybridLFPy population.

Subclasses answer three questions with their own rules:

get_all_SpCells      which presynaptic cells drive this postsynaptic cell
insert_all_synapses  where their synapses land on the morphology
draw_rand_pos        where in the nucleus each cell sits

They provide N_POST_TOTAL, a synapse placement rule and a nucleus geometry.
Cell positions and synapse placement draw from the global NumPy generator, so
draw order determines the result; rejection_sample_ellipse takes an explicit
sample_order (the LSO samples x, z, y, the others x, y, z).
"""

import numpy as np
from hybridLFPy.population import Population


class ReconstructionPopulation(Population):
    """hybridLFPy Population with the pipeline's shared wiring and placement."""

    #: Synapse parameters per presynaptic population name (from recon_core.config.params).
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

    # -- the chain hooks -----------------------------------------------------
    def cellsim(self, cellindex, return_just_cell=False):
        """hybridLFPy.Population.cellsim (0.2) with two hooks for the NEURON chain.

        `_chain_prepare(cell)` runs after the cell is built and before simulate;
        `_chain_collect(cellindex)` after simulate, while the cell exists. A
        population without these attributes runs hybridLFPy's body unchanged
        (including the decimate step and the float32 casts).
        """
        import LFPy
        import scipy.signal as ss
        from time import time

        tic = time()

        cell = LFPy.Cell(**self.cellParams)
        cell.set_pos(**self.pop_soma_pos[cellindex])
        cell.set_rotation(**self.rotations[cellindex])

        if return_just_cell:
            return cell
        else:
            self.insert_all_synapses(cellindex, cell)

            # set LFPykit.models instance cell attribute
            for probe in self.probes:
                probe.cell = cell

            # Chain hook 1/2: a spike-emitting subclass attaches its recorder
            # here, strictly before simulate.
            if hasattr(self, '_chain_prepare'):
                self._chain_prepare(cell)

            if 'rec_imem' in self.simulationParams.keys():
                try:
                    assert self.simulationParams['rec_imem']
                    cell.simulate(**self.simulationParams)
                    for probe in self.probes:
                        M = probe.get_transformation_matrix()
                        probe.data = M @ cell.imem
                    del cell.imem
                except AssertionError:
                    cell.simulate(probes=self.probes, **self.simulationParams)
            else:
                cell.simulate(probes=self.probes, **self.simulationParams)

            # Chain hook 2/2: read the recorded times, strictly after simulate
            # and while the cell still exists.
            if hasattr(self, '_chain_collect'):
                self._chain_collect(cellindex)

            # downsample probe.data attribute and unset cell
            for probe in self.probes:
                probe.data = ss.decimate(probe.data,
                                         q=self.decimatefrac
                                         ).astype(np.float32)
                probe.cell = None

            # put all necessary cell output in output dict
            for attrbt in self.savelist:
                attr = getattr(cell, attrbt)
                if isinstance(attr, np.ndarray):
                    self.output[cellindex][attrbt] = attr.astype('float32')
                else:
                    try:
                        self.output[cellindex][attrbt] = attr
                    except BaseException:
                        self.output[cellindex][attrbt] = str(attr)
                self.output[cellindex]['srate'] = 1E3 / self.dt_output

            # collect probe output
            for probe in self.probes:
                self.output[cellindex][probe.__class__.__name__] = \
                    probe.data.copy()

            # clean up hoc namespace
            cell.__del__()

            print('cell %s population %s in %.2f s' % (cellindex, self.y,
                                                       time() - tic))

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
                # (u - 0.5) * width + centre, not u * width + lo: the rounding
                # differs and propagates through the rejection loop.
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
