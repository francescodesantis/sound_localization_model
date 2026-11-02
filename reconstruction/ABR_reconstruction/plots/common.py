"""
Shared by the ABR figure scripts: electrode list, the nucleus colour key and
figure saving.
"""

import os

from recon_core.config import params as P

ELECTRODES = list(P.ELECTRODES)


# ---------------------------------------------------------------------------
# Nucleus colour key, used by every figure:
#   ANF purple, AVCN blue, MNTB green, MSO red, LSO orange.
# Generators of one nucleus use shades of its colour (GENERATOR_COLOUR).
NUCLEUS_COLOUR = {
    'ANF':  '#7b2cbf',    # purple
    'AVCN': '#1f6fb4',    # blue
    'MNTB': '#2a9d8f',    # green
    'MSO':  '#c1121f',    # red
    'LSO':  '#e09f3e',    # orange
}
COMPOSITE_COLOUR = 'k'
UNKNOWN_COLOUR = '#8d99ae'      # any generator not in the key, so it is obvious

# Per-generator shades. Each stays recognisably inside its nucleus' hue.
GENERATOR_COLOUR = {
    'ANF:fibre':        '#7b2cbf',
    'AVCN:SBC':         '#7fb3d5',
    'AVCN:GBCsoma':     '#5b9bd5',
    'AVCN:GBCaxon':     '#0d3c61',
    'MNTB:principal':   '#2a9d8f',
    'MNTB:calyx':       '#76c7bd',
    'MSO:postsynaptic': '#c1121f',
    'LSO:synaptic':     '#e09f3e',
}


def nucleus_colour(name):
    """Colour for a nucleus or a 'NUCLEUS:generator' key, from the one key."""
    if name in GENERATOR_COLOUR:
        return GENERATOR_COLOUR[name]
    return NUCLEUS_COLOUR.get(name.split(':', 1)[0], UNKNOWN_COLOUR)


def save(fig, path, dpi=150):
    """Write a figure, creating its directory."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=dpi)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f'figure saved to {path}')
    return path
