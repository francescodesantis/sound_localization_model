# Brainstem ABR and LFP reconstruction

Reconstructs the auditory brainstem response (ABR) at the scalp, and the
near-field LFP of single nuclei, from a NEST spiking simulation of the human
auditory brainstem (`simulate/`). Each nucleus is simulated as a population of
morphologically detailed NEURON cells; its current dipole is projected through
a 4-sphere head model and the nuclei are summed into the composite ABR.

Nuclei: auditory nerve (ANF), anteroventral cochlear nucleus (globular and
spherical bushy cells), MNTB (principal cell and calyx of Held), MSO, LSO.

## Installation

```bash
git submodule update --init --recursive
cd external/cochlea-1 && export CFLAGS="-std=c99" && python setup.py build_ext --inplace && cd ../..

mamba create -n sl_env -c conda-forge python=3.9.23 nest-simulator=3.8.0 -f environment.yml
mamba activate sl_env

for d in anf avcn mntb mso; do (cd reconstruction/models/$d && nrnivmodl); done
```

## Quickstart

1. Produce a `.pic` with the NEST model (`cd simulate && python main.py`).
2. From the repository root, run the complete ABR with either pipeline:

```bash
# every nucleus driven by the NEST spike trains
python reconstruction/main.py nest  --pic-file RESULTS/<f>.pic --itd-us 0 --condition binaural

# GBC/SBC -> MNTB -> MSO/LSO chained in NEURON
python reconstruction/main.py chain --pic-file RESULTS/<f>.pic --itd-us 0 --condition binaural
```

Both take the same options (`--help`): stimulus key (`--angle`, `--itd-us`,
`--ild-db`), stimulated ear (`--condition`), hemispheres (`--side`), MPI ranks
(`--ranks`), output directory (`--out-dir`) and the auditory-nerve
representation (`--anf-source`, `--wave1-source`).

## Output

```
reconstruction/abr_results/<stem>_<stimulus>_<condition>[_<side>][_anf<mode>]_<pipeline>/
  ABR_full.h5                              composite, per nucleus, per generator (µV)
  figures/full_abr_contributions.png       composite + nuclei, ipsilateral derivation
  figures/full_abr_electrodes.png          composite + nuclei at Cz, M1, M2
  cz_m1_m2/cz_m1_m2_{anf,gbc,sbc,mntb,mso,lso}.png
```

## Layout

```
reconstruction/
  main.py              entry point: nest | chain | lfp
  pipeline.py          complete-ABR orchestration
  commands.txt         command reference
  recon_core/
    config/            parameters (mirrored live from simulate/), head and nerve geometry
    physics/           dipoles, 4-sphere head model, cell populations
    utils/             paths, HDF5 I/O, signal processing, MPI
  ABR_reconstruction/  per-nucleus ABR, composite, figures
  NEURON_chain/        chained NEURON pipeline
  LFP_reconstruction/  near-field LFP per nucleus
  models/              NEURON morphologies (.hoc) and mechanisms (.mod)
```

Synaptic weights, delays, population sizes and convergences are read from
`simulate/models/BrainstemModel/params.py`; reconstruction-only constants live
in `recon_core/config/params.py`.

See `commands.txt` for every option and the LFP commands.
