#!/usr/bin/env python3
"""Download the ANCHOR human brainstem atlas annotations (route B).

ANCHOR is the Atlas of Neurochemical Characterization of the Human brainstem
with 3D Reconstruction. Sudha Gopalakrishnan Brain Centre, IIT Madras (2026).
Viewer: https://anchor.humanbrain.in/   doi:10.64898/2026.06.03.727794

Specimens (viewer 'data=' index to internal brain id):
    data=0  bid 342   Specimen 1, 25 gestational weeks (fetal, whole brain)
    data=1  bid 421   Specimen 2, 9 years  (brainstem)
    data=2  bid 296   Specimen 3, 54 years (brainstem), the adult one used here
    data=3  bid 539   Specimen 2 midbrain, 9 years

Per specimen the site serves
    section_data/data_<bid>_v<n>.json     per-section metadata, including a real
                                          rostrocaudal coordinate 'mm'
    annotations/<bid>/<STAIN>/<bid>_<secID>.geojson
                                          polygon annotations, each feature
                                          carrying the structure name, acronym
                                          and id from
                                          annotation_tree/brainstemnomenclature-v1.json

Cached (gitignored) in data/atlases/anchor/.

    python ABR_reconstruction/atlas/fetch_anchor.py
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, PACKAGE_ROOT)

# reconstruction/ is a couple of levels up; put it on sys.path so the atlas
# scripts share the pipeline's own repository root.
from recon_core.paths import REPO_ROOT                            # noqa: E402
CACHE_DIR = os.path.join(REPO_ROOT, 'data', 'atlases', 'anchor')

BASE = 'https://anchor.humanbrain.in/'

# viewer index to (brain id, section-data file, description)
SPECIMENS = {
    0: ('342', 'section_data/data_342_v1.json', 'Specimen 1, 25 GW fetal, whole brain'),
    1: ('421', 'section_data/data_421_v1.json', 'Specimen 2, 9 years, brainstem'),
    2: ('296', 'section_data/data_296_v2.json', 'Specimen 3, 54 years, brainstem (ADULT)'),
    3: ('539', 'section_data/data_539_v1.json', 'Specimen 2 midbrain, 9 years'),
}

ADULT = 2                       # the specimen route B uses
NOMENCLATURE = 'annotation_tree/brainstemnomenclature-v1.json'


def _get(rel, timeout=90):
    """Fetch <BASE><rel>, caching the raw bytes under CACHE_DIR/<rel>."""
    dest = os.path.join(CACHE_DIR, rel.replace('/', os.sep))
    if os.path.exists(dest):
        with open(dest, 'rb') as f:
            return f.read()
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with urllib.request.urlopen(BASE + rel, timeout=timeout) as r:
        blob = r.read()
    with open(dest, 'wb') as f:
        f.write(blob)
    return blob


def nomenclature():
    """The brainstem structure tree (name / acronym / id / parent)."""
    return json.loads(_get(NOMENCLATURE))


def sections(specimen=ADULT):
    """[{bid, secID, mm, stain, width, height, path, annotation?}, ...]"""
    _bid, rel, _desc = SPECIMENS[specimen]
    raw = json.loads(_get(rel))['sections']
    return [list(s.values())[0] for s in raw]


def annotation(rel_path):
    """One section's GeoJSON FeatureCollection, or None if the server 404s."""
    try:
        return json.loads(_get(rel_path))
    except urllib.error.HTTPError:
        return None


def fetch_all_annotations(specimen=ADULT, verbose=True):
    """Download every annotated section of a specimen, as [(rec, geojson)]."""
    recs = [r for r in sections(specimen) if r.get('annotation')]
    out, missing = [], 0
    for i, r in enumerate(recs, 1):
        g = annotation(r['annotation'])
        if g is None:
            missing += 1
            continue
        out.append((r, g))
        if verbose and (i % 10 == 0 or i == len(recs)):
            print('    %3d/%d sections' % (i, len(recs)))
    if verbose and missing:
        print('    (%d annotation files unavailable)' % missing)
    return out


def write_manifest(specimen=ADULT, n_sections=None):
    bid, rel, desc = SPECIMENS[specimen]
    meta = {
        'source': 'ANCHOR, Sudha Gopalakrishnan Brain Centre, IIT Madras (2026)',
        'doi': '10.64898/2026.06.03.727794',
        'viewer': BASE,
        'specimen_index': specimen,
        'brain_id': bid,
        'specimen': desc,
        'section_data': rel,
        'nomenclature': NOMENCLATURE,
        'annotated_sections_downloaded': n_sections,
        'downloaded': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(os.path.join(CACHE_DIR, 'MANIFEST.json'), 'w') as f:
        json.dump(meta, f, indent=2)


def main():
    print('ANCHOR human brainstem atlas -> %s' % CACHE_DIR)
    print()
    print('Specimen survey')
    print('-' * 78)
    for idx, (bid, rel, desc) in sorted(SPECIMENS.items()):
        recs = sections(idx)
        mm = [r['mm'] for r in recs]
        n_ann = sum(1 for r in recs if r.get('annotation'))
        print('  data=%d bid=%-4s %-42s entries=%3d annotated=%3d mm=[%.2f, %.2f]'
              % (idx, bid, desc, len(recs), n_ann, min(mm), max(mm)))

    print()
    print('Downloading annotations for the ADULT specimen (data=%d, bid=%s)'
          % (ADULT, SPECIMENS[ADULT][0]))
    pairs = fetch_all_annotations(ADULT)
    write_manifest(ADULT, len(pairs))
    print()
    print('  %d annotated sections cached' % len(pairs))

    nom = nomenclature()
    print('  nomenclature: %d bytes' % len(json.dumps(nom)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
