"""
CICochlea.py  —  simulate/cochleas/CICochlea.py
------------------------------------------------
Cochlear Implant model cochlea backend.

Pipeline:
    1. Apply HRTF (same as Zilany)
    2. Add omnidirectional noise (same as Zilany)
    3. Resample to the sample rate expected by the CI MATLAB model
    4. Write left/right WAVs to a temp directory
    5. Call external/CI_Model_1 via scripts/run_CI_cochlea.m using subprocess
    6. Load the resulting .mat spike files
    7. Convert to {fiber_idx -> brian2 Quantity (seconds)} — same as Zilany
    8. Return AnfResponse — compatible with load_anf_response / spikes_to_nestgen

Usage:
    from cochleas.CICochlea import COCHLEA_KEY as CI_COC_KEY
    from cochleas.CICochlea import sound_to_spikes as ci_cochlea
"""

from __future__ import division, print_function, absolute_import

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Union

import numpy as np
from joblib import Memory
from scipy import signal
from scipy.io import loadmat

from brian2 import Hz, second, seed
from brian2hears import Sound, dB

from utils.cochlea_utils import AnfResponse
from utils.hrtf_utils import run_hrtf
from utils.log_utils import logger
import pandas as pd

# =============================================================================
# Constants
# =============================================================================

COCHLEA_KEY = "CI"

# Repo root: simulate/cochleas/CICochlea.py
#   parents[0] = simulate/cochleas/
#   parents[1] = simulate/
#   parents[2] = sound_localization_model/   <-- repo root
_repo_root = Path(__file__).resolve().parents[2]

# CI model paths — hardcoded relative to repo root
CI_MODEL_ROOT = _repo_root / "external" / "CI_Model_1"
CI_PARAM_FILE = CI_MODEL_ROOT / "parameterstore" / "my_params.m"
MATLAB_SCRIPT = _repo_root / "scripts" / "run_CI_cochlea.m"

# MATLAB binary — replace "matlab" with the full path from `which matlab`
MATLAB_BIN = "/Applications/MATLAB_R2025b.app/Contents/MacOS/MATLAB"

# Sample rate expected by the CI MATLAB model
CI_TARGET_FS = 16000   # Hz

# Auditory nerve sampling rate inside the CI model (spike index -> seconds)
CI_AN_FS = 10000       # Hz — must match fs_AN in run_CI_cochlea.m

# joblib cache
CACHE_DIR = str(_repo_root / "data" / "ANF_SPIKETRAINS" / COCHLEA_KEY) + "/"
os.makedirs(CACHE_DIR, exist_ok=True)
memory = Memory(location=CACHE_DIR, verbose=0)

# Sanity checks at import time
assert CI_MODEL_ROOT.exists(), f"CI_MODEL_ROOT not found: {CI_MODEL_ROOT}"
assert CI_PARAM_FILE.exists(), f"CI_PARAM_FILE not found: {CI_PARAM_FILE}"
assert MATLAB_SCRIPT.exists(), f"MATLAB_SCRIPT not found: {MATLAB_SCRIPT}"


# =============================================================================
# Audio resampling helpers (mirrors ZilanyCochlea)
# =============================================================================

def _resample_sound(sound: Sound, original_fs: float, target_fs: float) -> Sound:
    """Resample a single-channel Sound object to target_fs."""
    ratio = target_fs / original_fs
    sound_data = np.array(sound)
    new_length = int(len(sound_data) * ratio)
    resampled_data = signal.resample(sound_data, new_length)
    return Sound(resampled_data, samplerate=target_fs * Hz)


def _resample_binaural_sound(binaural_sound: Sound, target_fs: float = CI_TARGET_FS) -> Sound:
    """Resample both channels of a binaural Sound to target_fs for the CI model."""
    original_fs = float(binaural_sound.samplerate / Hz)
    left_resampled  = _resample_sound(binaural_sound.left,  original_fs, target_fs)
    right_resampled = _resample_sound(binaural_sound.right, original_fs, target_fs)
    return Sound((left_resampled, right_resampled), samplerate=target_fs * Hz)


# =============================================================================
# WAV writer
# =============================================================================

def _write_wav(sound: Sound, path: Union[str, Path]) -> None:
    """Save a single-channel Sound object to a WAV file."""
    sound.save(str(path))


# =============================================================================
# MATLAB subprocess runner
# =============================================================================

def _run_matlab_ci_model(wav_left: Path, wav_right: Path, out_dir: Path) -> None:
    """
    Call MATLAB non-interactively to run the CI model on left/right WAV files.

    run_CI_cochlea.m must be a function accepting five string arguments:
        signalfilename_left, signalfilename_right, paramfilename, savepath, ci_model_root
    """
    matlab_cmd = (
        f"addpath('{MATLAB_SCRIPT.parent}'); "
        f"run_CI_cochlea("
        f"'{wav_left}', "
        f"'{wav_right}', "
        f"'{CI_PARAM_FILE}', "
        f"'{out_dir}', "
        f"'{CI_MODEL_ROOT}'"
        f")"
    )

    cmd = [MATLAB_BIN, "-nodisplay", "-nosplash", "-batch", matlab_cmd]

    logger.info(f"[CICochlea] Launching MATLAB...")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
            logger.error(f"[CICochlea] MATLAB stdout:\n{result.stdout}")   # full output on error
            logger.error(f"[CICochlea] MATLAB stderr:\n{result.stderr}")
            raise RuntimeError(
                f"MATLAB CI model failed (exit code {result.returncode}).\n"
                f"stderr: {result.stderr[-2000:]}"
            )
    else:
        logger.info("[CICochlea] MATLAB exited successfully.")

# =============================================================================
# .mat loader -> Zilany-compatible spike dict
# =============================================================================


def _load_mat_spikes(mat_path: Path) -> pd.Series:
    """
    Load spike_times_*.mat and return a pandas Series matching Zilany's format:
        index  = fiber index (int)
        values = list of spike times in seconds (plain floats, no brian2 units)
    """
    mat = loadmat(str(mat_path), squeeze_me=True)
    key = next(k for k in mat if not k.startswith("__"))
    raw = mat[key]   # numpy object array, one entry per fiber

    spike_list = []
    for fiber_spikes in raw:
        fiber_spikes = np.atleast_1d(fiber_spikes).astype(float)
        fiber_spikes = fiber_spikes[fiber_spikes > 0]
        fiber_spikes = fiber_spikes * 1000.0    # convert to milliseconds
        spike_list.append(fiber_spikes.tolist())         # plain list of floats, seconds

    return pd.Series(spike_list, name="spikes")

# =============================================================================
# Main cached function — mirrors ZilanyCochlea.sound_to_spikes exactly
# =============================================================================

@memory.cache
def sound_to_spikes(sound, angle, params, plot_spikes=False) -> AnfResponse:
    """
    Generate binaural ANF spike trains using the MATLAB CI model.

    Parameters
    ----------
    sound       : custom Sound object (Tone, WhiteNoise, etc.)
    angle       : spatial angle in degrees
    params      : dict with keys 'hrtf_params', 'rng_seed', 'omni_noise_level'
                  optional: 'ci_target_fs' (default: CI_TARGET_FS = 16000 Hz)
    plot_spikes : unused — kept for interface compatibility
    """
    logger.info(f"[CICochlea] Generating ANF spikes")

    hrtf_params = params["hrtf_params"]
    rng_seed    = params["rng_seed"]
    noise_level = params["omni_noise_level"] * dB
    target_fs   = params.get("ci_target_fs", CI_TARGET_FS)

    seed(rng_seed)

    # 1. HRTF filtering — identical to ZilanyCochlea
    logger.debug("[CICochlea] Running HRTF...")
    binaural_raw, gated_sound = run_hrtf(sound, angle, hrtf_params)

    # 2. Add omnidirectional noise — identical to ZilanyCochlea
    noise = Sound.whitenoise(binaural_raw.duration).atlevel(noise_level)
    binaural_noisy = binaural_raw + noise

    # 3. Resample to CI model target fs
    logger.debug(f"[CICochlea] Resampling to {target_fs} Hz...")
    binaural_sound = _resample_binaural_sound(binaural_noisy, target_fs=target_fs)
    L_sound = binaural_sound.left
    R_sound = binaural_sound.right

    # 4. Write WAVs -> run MATLAB -> load .mat (all inside a temp dir)
    with tempfile.TemporaryDirectory(prefix="ci_cochlea_") as tmp_dir:
        tmp       = Path(tmp_dir)
        wav_left  = tmp / "binaural_left.wav"
        wav_right = tmp / "binaural_right.wav"

        logger.debug(f"[CICochlea] Writing WAVs to {tmp_dir}")
        _write_wav(L_sound, wav_left)
        _write_wav(R_sound, wav_right)

        _run_matlab_ci_model(wav_left=wav_left, wav_right=wav_right, out_dir=tmp)

        logger.debug("[CICochlea] Loading MATLAB spike output...")
        spikes_L = _load_mat_spikes(tmp / "spike_times_left.mat")
        spikes_R = _load_mat_spikes(tmp / "spike_times_right.mat")

    # temp dir and WAVs deleted here; spike data already in memory
    logger.info(f"[CICochlea] Done — {len(spikes_L)} left fibers, {len(spikes_R)} right fibers.")

    return AnfResponse(
        binaural_anf_spiketrain={"L": spikes_L, "R": spikes_R},
        gated_sound=gated_sound,
        l_hrtf_sound=L_sound,
        r_hrtf_sound=R_sound,
    )