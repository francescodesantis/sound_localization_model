"""Per-stage performance metrics of the chain: wall time, per-rank busy and idle
time, cells per rank, GDF size and I/O time, collective time, peak memory.
Each stage writes a metrics JSON beside its dipole records. Cell cost varies
widely (an extended GBC against a five-section MNTB), so load balance is
recorded per rank.
"""

import json
import os
import resource
import time

from recon_core.utils.mpi_utils import COMM, RANK, SIZE

#: The metric set the performance chapter requires. Keep in step with the spec.
REQUIRED_METRICS = ('wall_s', 'rank_busy_s', 'rank_idle_s',
                    'n_cells_per_rank', 'gdf_bytes', 'gdf_io_s',
                    'collective_s', 'peak_rss_mb', 'n_ranks', 'nucleus',
                    'side', 'stage')


def metrics_path_for(dipoles_dir, nucleus, side, condition='binaural'):
    """Metrics file beside the dipole records, named to match them.

    A non-binaural stage-3 condition gets its own file: two conditions of the
    same nucleus/side otherwise overwrite one metrics record."""
    tag = '' if condition == 'binaural' else f'__{condition}'
    return os.path.join(dipoles_dir, f'metrics__{nucleus}__{side}{tag}.json')


def load_metrics(path):
    with open(path) as f:
        return json.load(f)


def missing_metrics(data):
    """Which required metrics a record lacks. Empty list means complete."""
    return [k for k in REQUIRED_METRICS if k not in data]


class _Phase:
    def __init__(self, timer, name):
        self.timer, self.name = timer, name

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.timer.phases[self.name] = (
            self.timer.phases.get(self.name, 0.0)
            + time.perf_counter() - self.t0)
        return False


class StageTimer:
    """Times one nucleus/side stage and every phase inside it."""

    def __init__(self, nucleus, side, stage):
        self.nucleus, self.side, self.stage = nucleus, side, int(stage)
        self.phases = {}
        self.n_cells = 0
        self.gdf_bytes = 0
        self.gdf_io_s = 0.0
        self.collective_s = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.wall_s = time.perf_counter() - self._t0
        return False

    def phase(self, name):
        return _Phase(self, name)

    def record_cells(self, n):
        """How many cells THIS rank simulated -- the load-balance numerator."""
        self.n_cells = int(n)

    def record_gdf_io(self, bytes_written, seconds):
        self.gdf_bytes += int(bytes_written)
        self.gdf_io_s += float(seconds)

    def record_collective(self, seconds):
        self.collective_s += float(seconds)

    def _peak_rss_mb(self):
        # ru_maxrss is KiB on Linux.
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    def write(self, path):
        """Gather every rank's timings on rank 0 and write one JSON.

        Returns the path on rank 0 and None elsewhere, following reduce_sum's
        contract so a forgotten RANK guard fails loudly instead of deadlocking.
        """
        mine = {'rank': RANK, 'busy_s': sum(self.phases.values()),
                'n_cells': self.n_cells, 'rss_mb': self._peak_rss_mb(),
                'wall_s': self.wall_s}
        allranks = COMM.gather(mine, root=0)
        if RANK != 0:
            return None

        busy = [r['busy_s'] for r in allranks]
        slowest = max(busy) if busy else 0.0
        data = {
            'nucleus': self.nucleus, 'side': self.side, 'stage': self.stage,
            'n_ranks': SIZE,
            # Stage makespan is the SLOWEST rank's wall time, not rank 0's --
            # under index-striped load imbalance rank 0 need not be the
            # slowest, and a rank-0-local value would understate makespan and
            # inflate speedup/efficiency for exactly the imbalanced stages
            # this instrumentation exists to expose.
            'wall_s': max(r['wall_s'] for r in allranks),
            'phases': self.phases,
            'rank_busy_s': busy,
            # Idle = how long each rank waited for the slowest one. This is the
            # load imbalance, and it is what index-striped distribution over
            # cells of unequal cost is expected to produce.
            'rank_idle_s': [slowest - b for b in busy],
            'n_cells_per_rank': [r['n_cells'] for r in allranks],
            'gdf_bytes': self.gdf_bytes,
            'gdf_io_s': self.gdf_io_s,
            'collective_s': self.collective_s,
            'peak_rss_mb': max(r['rss_mb'] for r in allranks),
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        return path
