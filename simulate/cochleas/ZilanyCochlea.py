from __future__ import division, print_function, absolute_import
import sys
from pathlib import Path
from typing import Union 
import dill
import faulthandler
faulthandler.enable()

model_root = Path(__file__).resolve().parents[2]            # top-level directory (where cochlea-1 lives)
cochlea_local_path = model_root / "external/cochlea-1/"

""" repo_root = Path(__file__).resolve().parents[3]            # top-level directory (where cochlea-1 lives)
cochlea_local_path = repo_root / "cochlea-1" """

# ---------------------------------------------------------------------
# 2. Add local cochlea path (so it overrides installed version)
# ---------------------------------------------------------------------
sys.path.insert(0, str(cochlea_local_path))
print(f"✅ Using local cochlea package from: {cochlea_local_path}")

import cochlea  # import it now while path is active

""" simulate_dir = Path(__file__).resolve().parents[1]
print(simulate_dir)
sys.path.insert(0, str(simulate_dir))
print(f"✅ Using utils from: {simulate_dir}")  """

from os import makedirs
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from joblib import Memory

# --- External model libraries ---
import thorns as th

# --- Brian2Hears sound objects still used for convenience ---
from brian2 import Hz, kHz, seed, ms, second
from brian2hears import Sound, dB, erbspace

# --- Project utilities ---
from utils.path_utils import Paths
from utils.custom_sounds import Tone, ToneBurst
from utils.log_utils import logger, tqdm
from utils.cochlea_utils import CFMAX, CFMIN, NUM_CF, AnfResponse
from utils.hrtf_utils import apply_artificial_ild_to_both, run_hrtf, apply_itd_to_sound, apply_artificial_ild

# --- Setup ---
COCHLEA_KEY = f"Zilany"
CACHE_DIR = Paths.ANF_SPIKES_DIR + COCHLEA_KEY + "/"
makedirs(CACHE_DIR, exist_ok=True)
memory = Memory(location=CACHE_DIR, verbose=0)

# =============================================================================
# Utility Functions
# =============================================================================

def resample_sound(sound: Sound, original_fs, target_fs=50000):
    """Resample a single-channel Sound object to target_fs."""
    ratio = target_fs / original_fs
    sound_data = np.array(sound)
    new_length = int(len(sound_data) * ratio)
    resampled_data = signal.resample(sound_data, new_length)
    return Sound(resampled_data, samplerate=target_fs * Hz)


def resample_binaural_sound(binaural_sound: Sound):
    """Resample both channels of a binaural Sound to 100 kHz for Zilany model."""
    original_fs = float(binaural_sound.samplerate / Hz)
    left_resampled = resample_sound(binaural_sound.left, original_fs, target_fs=100000)
    right_resampled = resample_sound(binaural_sound.right, original_fs, target_fs=100000)
    return Sound((left_resampled, right_resampled), samplerate=100 * kHz)


# =============================================================================
# Main Model Function
# =============================================================================

@memory.cache
def sound_to_spikes(sound, condition_val, params, plot_spikes=False) -> AnfResponse:

    hrtf_params = params["hrtf_params"]
    mode = hrtf_params.get("simulation_mode", "angle")
    rng_seed = params["rng_seed"]
    # noise_level = params["omni_noise_level"] * dB
    coch_par = params.get("cochlea_params", {})
    seed(rng_seed)

    # --- 1. HRTF ---
    # --- 1. Generate Binaural Sound ---
    if mode == "angle":
        # Existing logic: Full HRTF or HRTF-extracted cues
        logger.info(f"[sound_to_spikes] Generating ANF spikes for angle = {condition_val}°")
        binaural_raw, gated_sound = run_hrtf(sound, condition_val, hrtf_params)
    
    elif mode == "artificial_itd":
        # condition_val is ITD in seconds
        logger.info(f"[sound_to_spikes] Generating ANF spikes for ITD = {condition_val*1000000} us")
        binaural_raw = apply_itd_to_sound(sound.sound, condition_val)
        gated_sound = sound.sound
        
    elif mode == "artificial_ild":
        # condition_val is ILD in dB
        logger.info(f"[sound_to_spikes] Generating ANF spikes for ILD = {condition_val} dB")
        binaural_raw = apply_artificial_ild(sound.sound, condition_val)
        gated_sound = sound.sound

    elif mode == "artificial_ild_to_both":
        # condition_val is ILD in dB
        logger.info(f"[sound_to_spikes] Generating ANF spikes for ILD (to both sounds)= {condition_val} dB")
        binaural_raw = apply_artificial_ild_to_both(sound.sound, condition_val)
        gated_sound = sound.sound



    # --- 2. Add noise ---
    logger.debug(f"Gated sound level={gated_sound.level}")
    logger.debug(f"Binaural sound post-HRTF level={binaural_raw.level}")
    # noise = Sound.whitenoise(binaural_raw.duration).atlevel(noise_level)
    # binaural_sound = resample_binaural_sound(binaural_raw + noise)
    binaural_sound = resample_binaural_sound(binaural_raw)

    L_sound = binaural_sound.left
    R_sound = binaural_sound.right

    left_data = np.asarray(L_sound)[:,0]
    right_data = np.asarray(R_sound)[:,0]
    fs = float(binaural_sound.samplerate / Hz)

    cf = (CFMIN/Hz, CFMAX/Hz, NUM_CF)
    binaural_IHC_response = {}
    logger.info("[sound_to_spikes]Generating simulated ANF spikes using Zilany model...")

    for channel, data in zip(["L", "R"], [left_data, right_data]):
        logger.debug(f"[sound_to_spikes] Running Zilany model for {channel} ear...")
        anf = cochlea.run_zilany2014(
            data,
            fs,
            cf=cf,
            seed=rng_seed,
            **coch_par
        )

        # Collect spikes
        binaural_IHC_response[channel] = anf.spikes

        # Optional neurogram plot
        if plot_spikes:
            anf_acc = th.accumulate(anf, keep=['cf', 'duration'])
            anf_acc.sort_values('cf', ascending=False, inplace=True)
            plt.figure(figsize=(7, 3))
            th.plot_neurogram(anf_acc, fs)
            plt.title(f"Zilany2014 ANF Neurogram - {channel} ear")
            plt.xlabel("Time (s)")
            plt.ylabel("CF (Hz)")
            plt.tight_layout()
            plt.show()

    logger.info("[sound_to_spikes] Zilany spike generation complete.")

    return AnfResponse(
        binaural_anf_spiketrain=binaural_IHC_response,
        gated_sound=gated_sound,
        left_sound=L_sound,
        right_sound=R_sound
    )

def run_repeated_sound_psth(
    sound,
    n_reps: int = 10,
    save_name: str = "anf_297Hz_repeated.pic",
    params = None
):
    """
    Run repeated tone bursts through the Zilany ANF model,
    save spike times with dill, reload them, and plot a PSTH
    (Fig. 2B-style).

    Single CF, single ear, onset-aligned.
    """


    all_spikes = []
    coch_par = params.get("cochlea_params", {})


    for rep in tqdm(range(n_reps), desc="Running repetitions"):

        sound_n = resample_sound(sound.sound, original_fs=float(sound.sound.samplerate / Hz), target_fs=100000)
        data = np.asarray(sound_n)[:, 0]
        fs = float(sound_n.samplerate / Hz)
        cf = (CFMIN/Hz, CFMAX/Hz, NUM_CF)

        anf = cochlea.run_zilany2014(
            data,
            fs,
            cf=cf,
            seed=rep,  # vary seed across repetitions
            **coch_par
        )
        all_spikes.append(anf.spikes)

    # ------------------------------------------------------------------
    # Save spikes
    # ------------------------------------------------------------------
    save_path = Path(Paths.RESULTS_DIR) / save_name
    with open(save_path, "wb") as f:
        dill.dump(all_spikes, f)

    logger.info(f"[run_repeated_tone_psth] Saved spikes to {save_path}")
