"""
Extract MSO presynaptic spike trains from a .pic file into GDF files.

Output (in spikes_dir/):
  spikes-SBC_{side}-0.gdf    tab-separated: neuron_id  spike_time_ms
  spikes-MNTBC_{side}-0.gdf
  spikes-LNTBC_{side}-0.gdf
  metadata.json               GID info for CachedNetwork setup
"""

import os
import sys
import json

import dill
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recon_core import brainstem, paths                         # noqa: E402

# Unpickling a .pic reconstructs objects defined under simulate/, so that package
# has to be importable before dill.load() is called.
sys.path.insert(0, brainstem.SIMULATE_DIR)

# Ipsilateral populations extracted for every reconstruction.
# ANF is the auditory-nerve endbulb drive for the AVCN (GBC) pipeline; MSO and
# LSO reference only their own X_pops, so the extra GDF is harmless.
POPS = ['ANF', 'SBC', 'MNTBC', 'LNTBC']


def _write_pop(data_side, pop, label, spikes_dir, metadata):
    """Extract one population from data_side dict, write GDF, update metadata."""
    pop_data   = data_side[pop]
    times      = np.array(pop_data['times'],      dtype=float)
    senders    = np.array(pop_data['senders'],    dtype=int)
    global_ids = np.array(pop_data['global_ids'], dtype=int)

    sorted_ids = np.sort(global_ids)
    id_map     = {gid: i + 1 for i, gid in enumerate(sorted_ids)}
    remapped   = np.array([id_map[s] for s in senders], dtype=int)

    gdf_path = os.path.join(spikes_dir, f'spikes-{label}-0.gdf')
    arr = np.column_stack([remapped, times])
    np.savetxt(gdf_path, arr, fmt=['%d', '%.4f'], delimiter='\t')
    print(f'  {label}: {len(arr):,} spikes -> {gdf_path}')

    metadata[label] = {
        'first_gid': 1,
        'n_neurons':  int(len(global_ids)),
        'n_spikes':   int(len(times)),
        't_min_ms':   float(times.min()) if len(times) > 0 else None,
        't_max_ms':   float(times.max()) if len(times) > 0 else None,
    }


def _select_condition(angle_map, key):
    """Pick the condition key from a .pic rate map.

    Exact match first, which preserves existing behaviour (integer angles, 0/0.0
    and exact float ITD keys) and returns the identical sub-dict, so downstream
    GDF output stays byte-identical. Only when the exact key is absent does it
    fall back to the nearest numeric key within 1e-6 (1 us for ITD in seconds),
    to absorb float representation drift. A missing condition raises KeyError.
    """
    if key in angle_map:
        return key
    numeric = [k for k in angle_map if isinstance(k, (int, float))]
    if numeric:
        nearest = min(numeric, key=lambda k: abs(k - key))
        if abs(nearest - key) <= 1e-6:
            print(f'[extract] condition {key} not exact; using nearest {nearest}')
            return nearest
    raise KeyError(f'condition {key} not found; available: {sorted(angle_map)}')


def extract_and_save(pic_file, angle, side, spikes_dir):
    """
    Load pic_file, extract presynaptic spikes for the given angle/side,
    write GDF files and metadata.json to spikes_dir.

    Extracts:
      SBC_{contra_side}   contralateral SBC (MSO medial dendrite input)
      GBC_{contra_side}   contralateral GBC (MNTB calyx drive, decussating)
      SBC_{side}          ipsilateral SBC   (lateral dendrite input)
      MNTBC_{side}        ipsilateral MNTBC (soma inhibition)
      LNTBC_{side}        ipsilateral LNTBC (soma inhibition)

    Returns
    -------
    metadata : dict  {pop_label: {first_gid, n_neurons, n_spikes, t_min_ms, t_max_ms}}
    """
    os.makedirs(spikes_dir, exist_ok=True)
    contra_side = 'R' if side == 'L' else 'L'

    print(f'Loading {pic_file} ...', flush=True)
    with open(pic_file, 'rb') as f:
        result = dill.load(f, ignore=True)

    angle_map   = result.get('angle_to_rate') or result['cue_to_rate']
    sel         = _select_condition(angle_map, angle)
    data_ipsi   = angle_map[sel][side]
    data_contra = angle_map[sel][contra_side]

    try:
        stim_freq_hz = float(result['sounds']['base_sound'].frequency)
    except Exception:
        stim_freq_hz = None

    metadata = {'stim_freq_hz': stim_freq_hz}

    # Ipsilateral SBC + inhibitory pops
    for pop in POPS:
        _write_pop(data_ipsi, pop, f'{pop}_{side}', spikes_dir, metadata)

    # Contralateral SBC (lateral dendrite excitation)
    _write_pop(data_contra, 'SBC', f'SBC_{contra_side}', spikes_dir, metadata)

    # Contralateral GBC, the calyx-of-Held drive for the MNTB pipeline (the GBC
    # to MNTB projection decussates, so the ipsilateral MNTB is driven by the
    # contralateral GBC). MSO, LSO and AVCN ignore this extra GDF.
    _write_pop(data_contra, 'GBC', f'GBC_{contra_side}', spikes_dir, metadata)

    # Ipsilateral LSO output train, the LSO projection-neuron spikes that drive
    # the spiking-LSO ABR (travelling-wave dipole up the lateral lemniscus).
    # This is the nucleus's own output, not an input; the others ignore it.
    if 'LSO' in data_ipsi:
        _write_pop(data_ipsi, 'LSO', f'LSO_{side}', spikes_dir, metadata)

    meta_path = os.path.join(spikes_dir, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f'Metadata -> {meta_path}')

    return metadata


if __name__ == '__main__':
    pic_file   = paths.DEFAULT_PIC
    spikes_dir = os.path.join(paths.LFP_TMP_DIR, 'spikes')
    extract_and_save(pic_file, angle=0, side='L', spikes_dir=spikes_dir)