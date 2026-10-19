#!/usr/bin/env python3
"""
Verify the ventromedial axon-alignment rotation.

For a given morphology and target direction, compute LFPy angles with
gbc_biophysics.lfpy_align_angles, apply them, and re-measure the axon direction;
it must match the target. Exercises VCN_c09 (has Myelinated_Axon) and VCN_c02
(no myelinated axon -> AIS fallback), for both L and R (mirrored) targets.
"""
import os
import numpy as np
import neuron
import LFPy

HERE = os.path.dirname(os.path.abspath(__file__))
try:
    neuron.load_mechanisms(HERE)
except RuntimeError as e:
    if 'already exists' not in str(e):
        raise
import gbc_biophysics as gb

def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)

TARGETS = {'L': _unit([-1, 0, -1]), 'R': _unit([-1, 0, +1])}
CELLS = ['VCN_c09_Full_MeshInflate.hoc', 'VCN_c02_Full_MeshInflate.hoc']

def build(path):
    return LFPy.Cell(morphology=path, passive=False, v_init=-65.0,
                     dt=0.025, tstart=0., tstop=1.,
                     nsegs_method='lambda_f', lambda_f=100)

ok = True
for cellfile in CELLS:
    path = os.path.join(HERE, 'morphology', 'dryad', cellfile)
    for side, tgt in TARGETS.items():
        cell = build(path)
        native = gb.native_axon_direction(cell)
        angles = gb.lfpy_align_angles(native, tgt)
        cell.set_rotation(**angles)
        achieved = gb.native_axon_direction(cell)
        err = np.linalg.norm(achieved - tgt)
        status = 'OK' if err < 1e-2 else 'FAIL'
        if err >= 1e-2:
            ok = False
        print(f'{cellfile:32s} side {side}: native=[{native[0]:5.2f} {native[1]:5.2f} '
              f'{native[2]:5.2f}] -> achieved=[{achieved[0]:5.2f} {achieved[1]:5.2f} '
              f'{achieved[2]:5.2f}] target=[{tgt[0]:5.2f} {tgt[1]:5.2f} {tgt[2]:5.2f}] '
              f'err={err:.4f} {status}')

print('\nALL PASS' if ok else '\nSOME FAILED')
