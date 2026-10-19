# Sound Localization Model

A biologically grounded computational framework for simulating the human auditory periphery and brainstem during sound localization, and for reconstructing the electrophysiological signals (LFP and scalp ABR) that such a brainstem would produce.

The pipeline has three stages:

1. **Periphery** — a sound stimulus is spatialised (HRTF, or artificial ITD/ILD) and passed through a cochlear model to obtain auditory-nerve-fibre (ANF) spike trains.
2. **Brainstem** — the ANF spikes drive a bilateral spiking network in NEST (cochlear nucleus → MNTB/LNTB → MSO/LSO).
3. **Reconstruction** — the brainstem spikes drive morphologically detailed NEURON cells, from which near-field LFPs and the far-field scalp ABR are computed.

```
SOFA HRTF / artificial ITD-ILD ─┐
custom stimulus ────────────────┴─► cochlea (Zilany | Gammatone | Tan–Carney | CI)
                                     │  ANF spike trains (cached on disk)
                                     ▼
                         NEST brainstem network, left + right
                         SBC, GBC, MNTB, LNTB, MSO, LSO
                                     │  results/<MODE>/<experiment>/*.pic
              ┌──────────────────────┼──────────────────────────┐
              ▼                      ▼                          ▼
        plot/ notebooks      LFP reconstruction         ABR reconstruction
                             (hybridLFPy + NEURON)      per-nucleus dipoles →
                                                        4-sphere head → scalp ABR
```

---

## Repository layout

```
simulate/            Stages 1–2: stimuli, cochlear models, NEST brainstem network
  main.py              main simulation script (configured by editing it)
  main_linux.py        same, with the Tolnai ITD/ILD condition grid
  main_ataxia.py       same, with the ataxia parameter set
  params_GUI.py        Tk GUI for interactively tuning network parameters
  cochleas/            Zilany, Gammatone, Tan–Carney and CI front-ends
  models/
    SpikingModel.py      base class
    BrainstemModel/      network (BrainstemModel.py, BrainstemModel_SPN.py) and
                         parameter sets (params.py, paper_params.py, params_ataxia.py)
  utils/               stimuli, HRTF handling, ANF caching, paths, logging,
                       patched brian2hears / NEST helpers

reconstruction/      Stage 3: LFP and ABR reconstruction
  main.py              dispatcher for every LFP / ABR / plot / validation target
  run_abr.py           one-shot ABR pipeline (per-nucleus → composite → figures)
  recon_core/          shared framework: parameters, geometry, head model, I/O, MPI
  LFP_reconstruction/  near-field LFP, one script per nucleus
  ABR_reconstruction/  far-field ABR, one script per nucleus + composite; atlas/, plots/
  models/              NEURON cell models (.hoc + .mod) for AVCN, MNTB/calyx, MSO/LSO
  tests/               parameter checks and regression harness

plot/                Jupyter notebooks for analysis and paper figures
                     (plot/simulate and plot/utils are symlinks into simulate/)
data/SOFA/           HRTF datasets (RIEC, IRCAM, ARI, CIPIC, HUTUBS, SADIE, AACHEN, ...)
external/
  cochlea-1/           fork of the `cochlea` package (git submodule)
  CI_Model_1/          MATLAB cochlear-implant + auditory-nerve model
scripts/             helper scripts (e.g. run_CI_cochlea.m)
```

---

## Installation

### 1. Clone with submodules

```bash
git clone --recurse-submodules https://github.com/francescodesantis/sound_localization_model.git
cd sound_localization_model
# if already cloned without submodules:
git submodule update --init --recursive
```

### 2. Create the environment

The environment (`sl_env`, Python 3.9, NEST 3.8, Brian2, NEURON, LFPy/hybridLFPy, mpi4py) is defined in `environment.yml`:

```bash
mamba env create -f environment.yml
mamba activate sl_env
```

`environment.yml` pins Linux builds. On macOS, create a Python 3.9 environment with `nest-simulator=3.8` from conda-forge and install the remaining Python packages from `requirements.txt` and the `pip:` section of `environment.yml`.

### 3. Build the cochlea submodule

```bash
cd external/cochlea-1
export CFLAGS="-std=c99"
python setup.py build_ext --inplace
cd ../..
```

After building, the models can be imported from anywhere in the repository, e.g. `from cochlea.zilany2014 import run_zilany2014`.

### 4. Compile the NEURON mechanisms (reconstruction only)

```bash
for d in reconstruction/models/avcn reconstruction/models/mntb reconstruction/models/mso; do
  (cd "$d" && nrnivmodl)
done
```

### 5. Create the log directory

```bash
mkdir -p logs
```

### Optional: cochlear-implant front-end

`simulate/cochleas/CICochlea.py` calls the MATLAB model in `external/CI_Model_1`, so MATLAB must be available on the `PATH`.

---

## Stages 1–2: running the brainstem simulation

Simulations are configured by editing the `__main__` block of `simulate/main.py`, then running it **from inside `simulate/`** (paths such as `../data/`, `../results/` and `../logs/` are relative to that directory):

```bash
cd simulate
python main.py
```

What to configure in `main.py`:

| Setting | Where | Options |
|---|---|---|
| Stimulus | `inputs` | `Tone`, `ToneBurst`, `Click`, `Click_Train`, `WhiteNoise`, `HarmonicComplex` (from `utils/custom_sounds.py`) |
| Spatial cue | `MODE` | `"angle"` (HRTF azimuths), `"artificial_itd"` (seconds), `"artificial_ild"` (dB) |
| Condition grid | `loop_range` | the azimuths / ITDs / ILDs to sweep |
| Cochlea | `cochlea_key` | `ZI_COC_KEY` (Zilany, default), `CI_COC_KEY`, ... |
| Network parameters | `params(...)` objects in `ps` | override fields such as `p.SYN_WEIGHTS.*`, `p.SYN_DELAYS.*`, RNG seeds |

Defaults for every network parameter (population sizes, convergence, weights, delays, membrane constants) are in `simulate/models/BrainstemModel/params.py`. Use `python params_GUI.py` (from `simulate/`) to explore them interactively.

For each condition the script resets NEST, generates (or loads) the ANF response, builds `BrainstemModel`, simulates it and records spikes from every population on both sides.

### Outputs

- **ANF spike trains** are cached with joblib in `data/ANF_SPIKETRAINS/<cochlea>/`. Generating them is the most expensive step; set `ignore_cache=False` in `load_anf_response` to reuse them across runs.
- **Results** are written with `dill` to `results/<MODE>/<experiment>/<sound>&<param key>.pic`. Each file contains the stimuli, the per-condition spike output of every population (`cue_to_rate`) and a snapshot of the full configuration.
- **Logs** go to `logs/`.

---

## Stage 3: LFP and ABR reconstruction

All reconstruction targets go through one dispatcher. Run it **from the repository root**:

```bash
python reconstruction/main.py list                       # show every target
python reconstruction/main.py lfp mso  --pic-file RESULTS/<f>.pic --side L
python reconstruction/main.py abr full --pic-file RESULTS/<f>.pic --side both
python reconstruction/main.py plot positions --view coronal
python reconstruction/main.py validate params
```

| Stage | Targets |
|---|---|
| `lfp` | `mso`, `gbc`, `sbc`, `lso`, `mntb`, `replot`, `tuning` |
| `abr` | `mso`, `avcn`, `lso`, `mntb`, `full` (composite of all nuclei), `bi` |
| `plot` | `positions`, `morphology`, `electrodes`, `full-abr`, `itd-sweep`, `binaural`, `tolnai` |
| `validate` | `params`, `head-geometry`, `regression`, `mntb`, `calyx`, `gbc`, `gbc-axon`, `lso-axon` |

Any extra flags are passed through to the underlying script, and every target runs under MPI:

```bash
mpiexec -n 4 python reconstruction/main.py abr mso --pic-file RESULTS/<f>.pic --itd-us 0
```

### Full ABR in one command

`reconstruction/run_abr.py` chains the whole ABR pipeline: simulate each nucleus and store its head-frame dipole, superpose them into the composite ABR, and draw the figures. Edit its `CONFIG` block (`PIC_FILE`, `CONDITION`, `SELECTOR`, `N_CELLS`, `MPI_RANKS`, `FORCE_RERUN`, `DRY_RUN`) and run:

```bash
python reconstruction/run_abr.py
```

Output is written to `reconstruction/abr_results/<pic stem>_<condition>/`.

### How it works

- `recon_core/brainstem.py` reads delays, reversal potentials and population sizes **directly** from `simulate/models/BrainstemModel/params.py`, so the reconstruction always matches the network that produced the spikes.
- Each nucleus (MSO, LSO, AVCN GBC/SBC, MNTB and calyx) is a hybridLFPy population built on `recon_core/population.py`, using the NEURON morphologies in `reconstruction/models/`.
- Nucleus positions (MNI atlas coordinates) and electrodes live in `recon_core/head_geometry.py`. The auditory-nerve trajectory lives in `recon_core/anf_geometry.py`.
- Each nucleus's current dipole is projected to the scalp through a 4-sphere volume conductor (`recon_core/head_model.py`). Because projection is linear, nuclei are simulated independently and summed afterwards by `main_abr_full.py`.
- File locations and HDF5 formats shared between stages are defined in `recon_core/paths.py` and `recon_core/io_utils.py`.

---

## Plots and analysis

The notebooks in `plot/` analyse `.pic` results and reproduce the paper figures:

| Notebook | Content |
|---|---|
| `paper_plots.ipynb` | figures for the paper |
| `ANFs.ipynb` | auditory-nerve responses |
| `BCs.ipynb` | bushy-cell (SBC/GBC) activity |
| `ILDs.ipynb` | ILD tuning (LSO) |
| `hrtf.ipynb` | HRTF / HRIR inspection |
| `CI.ipynb` | cochlear-implant front-end |

`plot/analyze/` contains reusable analysis code (`report.py`, `sound_analysis.py`).

---

## Data

HRTFs are read from SOFA files in `data/SOFA/`. The dataset used is set by `Paths.SOFA_DIR` in `simulate/utils/path_utils.py` (RIEC by default), and the subject by `hrtf_params['subj_number']` in the cochlea parameters. Datasets included: RIEC, IRCAM (LISTEN), ARI, CIPIC, HUTUBS (measured and simulated), SADIE II and AACHEN.
