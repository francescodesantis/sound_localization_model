import math
import brian2 as b2
import numpy as np
from brian2 import ms, Hz
from brian2hears import IRCAM_LISTEN, Sound, dB
from typing import Union  # <-- ADD THIS

from utils.path_utils import Paths
from utils.custom_sounds import Tone, ToneBurst
from utils.log_utils import logger
from utils.manual_fixes_to_b2h.HeadlessDatabase import HeadlessDatabase
from sorcery import dict_of
from .cochlea_utils import ANGLE_TO_SOFA, SOFA_TO_ANGLE
import os
from scipy.signal import fftconvolve, resample
import pysofaconventions as sofa
from scipy.fft import rfft, irfft
import matplotlib.pyplot as plt

def load_hrir(subject_id: int, azimuth_deg: float):
    """
    Load left/right HRIR for a given subject and azimuth (elevation = 0°).

    Returns
    -------
    left_ir, right_ir : np.ndarray
    sofa_fs           : float
    """

    files = sorted(
        f for f in os.listdir(Paths.SOFA_DIR)
        if f.lower().endswith(".sofa")
    )

    sofa_path = os.path.join(Paths.SOFA_DIR, files[subject_id])
    sofa_file = sofa.SOFAFile(sofa_path, "r")

    ir = sofa_file.getDataIR()                     # (M, R, N)
    positions = sofa_file.getSourcePositionValues()  # (M, C)

    az = positions[:, 0]
    el = positions[:, 1]

    az_target = azimuth_deg

    # Find closest measurement (elevation fixed to 0)
    dist = np.sqrt((az - az_target)**2 + (el - 0)**2)
    m_idx = int(np.argmin(dist))

    left_ir  = ir[m_idx, 0, :]
    right_ir = ir[m_idx, 1, :]

    sofa_fs = float(np.array(sofa_file.getSamplingRate()).ravel()[0])

    return left_ir, right_ir, sofa_fs

def load_hrtf_subject(subj: int):
    # --- List and sort all .sofa files directly from the directory ---
    files = sorted(
        f for f in os.listdir(Paths.SOFA_DIR)
        if f.lower().endswith(".sofa")
    )

    if not files:
        raise FileNotFoundError(f"No .sofa files found in directory: {Paths.SOFA_DIR}")

    # --- Index validation ---
    if not (0 <= subj < len(files)):
        raise IndexError(
            f"subj={subj} is out of range: directory contains {len(files)} SOFA files."
        )

    sofa_path = files[subj]

    # --- Load the selected SOFA file ---
    hrtf = sofa.SOFAFile(str(Paths.SOFA_DIR + sofa_path), "r")  # SOFAFile expects a string path

    return hrtf

def apply_sofa_hrtf_to_sound(
    sofa_file,
    sound: Sound,
    azimuth_deg: float,
    elevation_deg: float = 0.0,
) -> Sound:
    """
    Apply the SOFA HRTF for the closest (azimuth, elevation) to a mono
    Brian2Hears Sound. Automatically resamples SOFA IRs to match the
    Sound's samplerate if needed.
    """

    # ---------- Extract IRs and positions ----------
    ir = sofa_file.getDataIR()                 # shape (M, R, N)
    positions = sofa_file.getSourcePositionValues()  # shape (M, C)

    # Spherical coords expected: az = positions[:,0], el = positions[:,1]
    az = positions[:, 0]
    el = positions[:, 1]

    # ---------- Find nearest measurement ----------
    diff_az = az - azimuth_deg
    diff_el = el - elevation_deg
    dist = np.sqrt(diff_az**2 + diff_el**2)
    m_idx = int(np.argmin(dist))

    left_ir = ir[m_idx, 0, :]
    right_ir = ir[m_idx, 1, :]

    # ---------- Sampling rate handling ----------
    # SOFA may store fs either as [I] or [M]; flatten and get the first value
    sofa_fs = float(np.array(sofa_file.getSamplingRate()).ravel()[0])
    sound_fs = float(sound.samplerate)

    # If different → resample IRs to match the sound samplerate
    if sound_fs != sofa_fs:
        # Compute new length after resampling
        N_old = left_ir.shape[0]
        N_new = int(round(N_old * (sound_fs / sofa_fs)))

        left_ir  = resample(left_ir,  N_new)
        right_ir = resample(right_ir, N_new)

    # ---------- Extract mono signal ----------
    x = np.asarray(sound).flatten()

    N = len(x)

    ir_len = max(len(left_ir), len(right_ir))

    # nmax = next power of 2 >= (signal + ir_length)
    nmax = max(N + ir_len, ir_len)
    nmax = 2**int(np.ceil(np.log2(nmax)))

    # Zero-pad HRIRs to nmax
    Lpad = np.hstack([left_ir,  np.zeros(nmax - len(left_ir))])
    Rpad = np.hstack([right_ir, np.zeros(nmax - len(right_ir))])
    xpad = np.hstack([np.zeros(ir_len), x, np.zeros(nmax - ir_len - N)])

    # ---------- FFT convolution ----------
    L_fft = rfft(Lpad, n=nmax)
    R_fft = rfft(Rpad, n=nmax)
    X_fft = rfft(xpad, n=nmax)

    L_out = irfft(L_fft * X_fft).real
    R_out = irfft(R_fft * X_fft).real

    L_final = L_out[ir_len:ir_len + N]
    R_final = R_out[ir_len:ir_len + N]

    # ---------- Return binaural Brian2Hears sound ----------
    stereo = np.column_stack([L_final, R_final])
    return Sound(stereo, sound.samplerate)

def estimate_itd_from_hrir(left_ir, right_ir, fs):
    """
    Estimate ITD (in seconds) from HRIR using cross-correlation.
    Positive ITD = sound arrives first at left ear.
    """

    corr = np.correlate(left_ir, right_ir, mode='full')
    lag = np.argmax(corr) - (len(right_ir) - 1)

    itd_sec = lag / fs
    return itd_sec

def apply_itd_to_sound(sound: Sound, itd_sec: float) -> Sound:
    """
    Apply pure ITD to mono sound by delaying one channel.
    Output is (n + delay_samples) long — both channels have identical energy.
    """
    fs = float(sound.samplerate)
    x = np.asarray(sound).flatten()
    delay_samples = int(round(abs(itd_sec) * fs))

    if delay_samples == 0:
        stereo = np.column_stack([x, x])
        return Sound(stereo, samplerate=fs * Hz)

    pad = np.zeros(delay_samples)

    if itd_sec < 0:
        # left leads → right is delayed
        left  = np.concatenate([x,   pad])  # signal then silence
        right = np.concatenate([pad, x  ])  # silence then signal
    else:
        # right leads → left is delayed
        left  = np.concatenate([pad, x  ])
        right = np.concatenate([x,   pad])

    stereo = np.column_stack([left, right])
    return Sound(stereo, samplerate=fs * Hz)

def apply_ild_to_sound(sofa_file, sound: Sound, azimuth_deg: float) -> Sound:
    """
    Apply only the ILD extracted from the HRIRs to a mono sound.
    Uses apply_sofa_hrtf_to_sound to measure per-ear level,
    then returns two copies of the original mono sound with those absolute levels.
    """
    # --- Apply full HRTF to measure per-ear levels ---
    binaural_sound = apply_sofa_hrtf_to_sound(
        sofa_file=sofa_file,
        sound=sound,
        azimuth_deg=azimuth_deg,
        elevation_deg=0.0,
    )

    l_level = Sound(np.asarray(binaural_sound)[:, 0], samplerate=sound.samplerate).level
    r_level = Sound(np.asarray(binaural_sound)[:, 1], samplerate=sound.samplerate).level

    # --- Set original mono signal to each ear's absolute level ---
    left_scaled  = sound.copy()
    right_scaled = sound.copy()

    left_scaled.level  = l_level
    right_scaled.level = r_level

    stereo = np.column_stack([np.asarray(left_scaled).flatten(),
                               np.asarray(right_scaled).flatten()])
    return Sound(stereo, samplerate=sound.samplerate)

def run_hrtf(
    sound,
    condition_val: float,
    hrtf_params
):
    """
    Unified HRTF interface with selectable binaural cue:

    cue_to_apply:
        - "HRTF"     → full HRIR convolution (default, original behavior)
        - "itd_only" → apply only ITD (pure delay)
        - "ild_only" → apply only ILD (level difference)

    Returns
    -------
    binaural_sound : Sound (stereo)
    sound          : Sound (mono, possibly preprocessed)
    """
    angle = condition_val
    logger.debug(f"[run_hrtf] Starting for angle={angle}")

    subj = hrtf_params["subj_number"]
    cue  = hrtf_params["cue_to_apply"]

    # --- unwrap sound ---
    if type(sound) is not Sound:
        sound = sound.sound

    sound_fs = float(sound.samplerate)

    if cue == "HRTF":
        logger.debug("[run_hrtf] Applying FULL HRTF")

        sofa_file = load_hrtf_subject(subj)

        binaural_sound = apply_sofa_hrtf_to_sound(
            sofa_file=sofa_file,
            sound=sound,
            azimuth_deg=ANGLE_TO_SOFA[angle],
            elevation_deg=0.0,
        )
        return binaural_sound, sound

    # ==========================================================
    left_ir, right_ir, sofa_fs = load_hrir(subj, ANGLE_TO_SOFA[angle])

    # --- resample HRIR if needed ---
    if sound_fs != sofa_fs:
        N_old = left_ir.shape[0]
        N_new = int(round(N_old * (sound_fs / sofa_fs)))

        left_ir  = resample(left_ir,  N_new)
        right_ir = resample(right_ir, N_new)

    # ==========================================================
    if cue == "itd_only":
        logger.debug("[run_hrtf] Applying ITD only")

        itd = estimate_itd_from_hrir(left_ir, right_ir, sound_fs)
        binaural_sound = apply_itd_to_sound(sound, itd)

        return binaural_sound, sound
    # ==========================================================
    elif cue == "ild_only":
        logger.debug("[run_hrtf] Applying ILD only")
        sofa_file = load_hrtf_subject(subj)
        binaural_sound = apply_ild_to_sound(
            sofa_file=sofa_file,
            sound=sound,
            azimuth_deg=ANGLE_TO_SOFA[angle],
        )
        return binaural_sound, sound

    else:
        raise ValueError(f"Unknown cue_to_apply: {cue}")
    
def apply_artificial_ild(sound: Sound, ild_db: float) -> Sound:
    """Applies a pure gain difference to a mono sound."""
    fs = float(sound.samplerate)
    left = sound.copy()
    right = sound.copy()
    
    # Apply half the ILD to each side (one up, one down) or relative to 0
    left.level -= ((ild_db / 2.0) * dB)
    right.level += ((ild_db / 2.0) * dB)
    
    stereo = np.column_stack([np.asarray(left).flatten(), np.asarray(right).flatten()])
    return Sound(stereo, samplerate=fs*Hz)

def apply_artificial_ild_exp(sound: Sound, lev_db: float) -> Sound:
    """Applies a pure gain difference to a mono sound."""
    fs = float(sound.samplerate)
    left = sound.copy()
    right = sound.copy()
    right.level = lev_db * dB
    logger.debug(
    f"Right Sound Level: {right.level} dB\n"
    f"Left Sound Level: {left.level} dB\n"
    )
    stereo = np.column_stack([np.asarray(left).flatten(), np.asarray(right).flatten()])
    return Sound(stereo, samplerate=fs*Hz)

#------------------- PLOT ---------------------------------------

def plot_hrir(subject_id: int, azimuth_deg: float, target_fs=None):
    """
    Plot HRIR (left & right) in time domain.
    """

    left_ir, right_ir, sofa_fs = load_hrir(subject_id, azimuth_deg)

    # Optional resampling
    fs = sofa_fs
    if target_fs is not None and target_fs != sofa_fs:
        N_new = int(round(len(left_ir) * target_fs / sofa_fs))
        left_ir  = resample(left_ir,  N_new)
        right_ir = resample(right_ir, N_new)
        fs = target_fs

    times = np.arange(len(left_ir)) / fs * 1000  # ms

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, left_ir,  color='m', label='Left')
    ax.plot(times, right_ir, color='g', label='Right')

    ax.set_title(f"HRIR – Subject {subject_id}, Azimuth {SOFA_TO_ANGLE[azimuth_deg]}°, Elevation 0°")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()

def plot_hrtf_magnitude(subject_id: int, azimuth_deg: float, fs: float, figsize=(10, 4)):
    """
    Plot HRTF magnitude response (dB) for left and right ears.
    """

    left_ir, right_ir, sofa_fs = load_hrir(subject_id, azimuth_deg)

    # Resample HRIR if needed
    if sofa_fs != fs:
        N_new = int(round(len(left_ir) * fs / sofa_fs))
        left_ir  = resample(left_ir,  N_new)
        right_ir = resample(right_ir, N_new)

    # FFT padding (same logic as your notebook)
    ir_len = max(len(left_ir), len(right_ir))
    nmax = 2**int(np.ceil(np.log2(ir_len)))

    Lpad = np.hstack([left_ir,  np.zeros(nmax - len(left_ir))])
    Rpad = np.hstack([right_ir, np.zeros(nmax - len(right_ir))])

    L_fft = rfft(Lpad)
    R_fft = rfft(Rpad)

    freqs = np.fft.rfftfreq(nmax, d=1/fs)

    eps = 1e-12
    L_mag = 20 * np.log10(np.abs(L_fft) + eps)
    R_mag = 20 * np.log10(np.abs(R_fft) + eps)

    fig, ax = plt.subplots(figsize=figsize)

    ax.semilogx(freqs, L_mag, color='m', label="Left")
    ax.semilogx(freqs, R_mag, color='g', label="Right")

    ax.set_xlim(200, 20000)

    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title(f"HRTF Magnitude – Subject {subject_id}, Azimuth {SOFA_TO_ANGLE[azimuth_deg]}°, Elevation 0°")

    ax.set_xscale('log')
    ax.set_xticks(np.array([0.2, 0.5, 1, 2, 5, 10, 20]) * 1000)
    ax.set_xticklabels(['0.2', '0.5', '1', '2', '5', '10', '20'])

    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()

