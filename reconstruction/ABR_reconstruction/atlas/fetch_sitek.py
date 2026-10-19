#!/usr/bin/env python3
"""Download the Sitek et al. (2019) subcortical auditory atlas ROIs.

Sitek KR, Gulban OF, Calabrese E, Johnson GA, Lage-Castellanos A, Moerel M,
Ghosh SS, De Martino F (2019). "Mapping the human subcortical auditory system
using histology, post mortem MRI and in vivo MRI at 7T." eLife 8:e48932.
Repository: https://github.com/sitek/subcortical-auditory-atlas (BSD-3-Clause)

Three ROI volumes, all in MNI ICBM152 2009b Nonlinear Symmetric space:
  bigbrain     post mortem histology (BigBrain 2015), conjunction ROIs
  postmortem   7T post mortem MRI (Duke Center for In Vivo Microscopy)
  invivo       in vivo 7T fMRI atlas (thresholded)

Cached (gitignored) in data/atlases/sitek/.

    python ABR_reconstruction/atlas/fetch_sitek.py
"""

import hashlib
import json
import os
import sys
import time
import urllib.request

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, PACKAGE_ROOT)

# reconstruction/ is a couple of levels up; put it on sys.path so the atlas
# scripts share the pipeline's own repository root.
from recon_core.paths import REPO_ROOT                            # noqa: E402
CACHE_DIR = os.path.join(REPO_ROOT, 'data', 'atlases', 'sitek')

_BASE = 'https://github.com/sitek/subcortical-auditory-atlas/raw/master/atlases'

FILES = {
    'bigbrain':   'sub-bigbrain_MNI_conjunction_rois.nii.gz',
    'postmortem': 'sub-postmortem_MNI_rois.nii.gz',
    'invivo':     'sub-invivo_MNI_rois.nii.gz',
}

MNI_SPACE = 'MNI ICBM152 2009b Nonlinear Symmetric'


def cached_path(key):
    return os.path.join(CACHE_DIR, FILES[key])


def fetch(force=False, verbose=True):
    """Download any missing ROI volume, returning {key: local path}."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    out, manifest = {}, {}
    for key, name in FILES.items():
        dest = os.path.join(CACHE_DIR, name)
        if force or not os.path.exists(dest):
            url = '%s/%s' % (_BASE, name)
            if verbose:
                print('  downloading %-42s <- %s' % (name, url))
            with urllib.request.urlopen(url, timeout=120) as r, open(dest, 'wb') as f:
                f.write(r.read())
        elif verbose:
            print('  cached      %-42s (%d bytes)' % (name, os.path.getsize(dest)))
        out[key] = dest
        manifest[key] = {
            'file': name,
            'url': '%s/%s' % (_BASE, name),
            'bytes': os.path.getsize(dest),
            'sha256': hashlib.sha256(open(dest, 'rb').read()).hexdigest(),
        }

    meta = {
        'source': 'Sitek et al. 2019 eLife 8:e48932',
        'repository': 'https://github.com/sitek/subcortical-auditory-atlas',
        'license': 'BSD-3-Clause',
        'space': MNI_SPACE,
        'downloaded': time.strftime('%Y-%m-%d %H:%M:%S'),
        'files': manifest,
    }
    with open(os.path.join(CACHE_DIR, 'MANIFEST.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    return out


if __name__ == '__main__':
    print('Sitek 2019 subcortical auditory atlas -> %s' % CACHE_DIR)
    paths = fetch(force='--force' in sys.argv)
    print()
    for k, p in paths.items():
        print('  %-11s %s' % (k, p))
