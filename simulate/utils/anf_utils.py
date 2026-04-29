from brian2 import Hz, kHz, ms
from brian2hears import Sound
from joblib.memory import MemorizedFunc
from sorcery import dict_of
from utils.custom_sounds import Click, Tone, ToneBurst, WhiteNoise, Click_Train, HarmonicComplex
from utils.log_utils import logger
from typing import Union   # ✅ added
import numpy as np

from cochleas.GammatoneCochlea import COCHLEA_KEY as GAMMATONE_COC_KEY
from cochleas.GammatoneCochlea import sound_to_spikes as gammatone_cochlea
from cochleas.TanCarneyCochlea import COCHLEA_KEY as TC_COC_KEY
from cochleas.TanCarneyCochlea import sound_to_spikes as tc_cochlea
from cochleas.ZilanyCochlea import COCHLEA_KEY as ZI_COC_KEY
from cochleas.ZilanyCochlea import sound_to_spikes as zi_cochlea
from cochleas.CICochlea import COCHLEA_KEY as CI_COC_KEY
from cochleas.CICochlea import sound_to_spikes as ci_cochlea

from utils.cochlea_utils import ANGLES, NUM_ANF_PER_HC, NUM_CF, AnfResponse
import nest


SOUND_FREQUENCIES = [100 * Hz, 1 * kHz, 10 * kHz]
INFO_FILE_NAME = "info.txt"
INFO_HEADER = "this directory holds all computed angles, for a specific sound, with a specific cochlear backend. the pickled sound is also available. for cochleas that do not use HRTF, left and right sounds are the same. \n"
COCHLEAS = {
    GAMMATONE_COC_KEY: gammatone_cochlea,
    TC_COC_KEY: tc_cochlea,
    ZI_COC_KEY: zi_cochlea,
    CI_COC_KEY: ci_cochlea,
}


def create_sound_key(sound):
    
    add_info = None
    if type(sound) is Tone:
        add_info = str(sound.frequency).replace(" ", "")
        level = round(sound.sound.level)
        sound_type = "tone"
    elif type(sound) is ToneBurst:
        add_info = str(sound.frequency).replace(" ", "")
        level = round(sound.sound.level)
        sound_type = "toneburst"
    elif type(sound) is WhiteNoise:
        sound_type = "whitenoise"
        level = round(sound.sound.level)
    elif type(sound) is Click:
        sound_type = "click"
        level = round(sound.peak)
    elif type(sound) is Click_Train:
        sound_type = "click_train"
        level = round(sound.peak)
    elif type(sound) is HarmonicComplex:
        sound_type = "harmonic"
    else:
        raise NotImplementedError(f"unknown sound {sound}")
    if add_info:
        return f"{sound_type}_{add_info}_{level}dB"
    else:
        return f"{sound_type}_{level}dB"


def load_anf_response(sound, angle, cochlea_key, params, ignore_cache=False):

    cochlea_func: MemorizedFunc = COCHLEAS[cochlea_key]
    logger.info(f"Subject chosen: {params[cochlea_key]['hrtf_params']['subj_number']}")
    params = params[cochlea_key]

    if not cochlea_func.check_call_in_cache(sound, angle, params):
        logger.info("[load_anf_response] Saved ANF not found. Regenerating...")

    if ignore_cache:
        logger.info("[load_anf_response] Ignoring cache — forcing recompute.")
        cochlea_func = cochlea_func.call

    logger.info(f"[load_anf_response] Generating ANF for "
                f"sound={sound}, angle={angle}, key={cochlea_key}")

    if ignore_cache:
        cochlea_func = cochlea_func.call  # forces execution
    try:
        anf = cochlea_func(sound, angle, params, plot_spikes=False)
    except TypeError as e:
        if "unexpected" in e.args[0]:
            logger.error(f"{e}, please check the signature of cochlea")
        raise e
    return anf


def spikes_to_nestgen(anf_response: AnfResponse):
    nest.set_verbosity("M_ERROR")
    anfs_per_ear = {}
    for channel, response_ANF in anf_response.binaural_anf_spiketrain.items():
        s = []
        # make sure all ANFs have a spiking array and no zero-time spikes are present
        for i in range(NUM_CF * NUM_ANF_PER_HC):
            spiketimes = response_ANF.get(i, [] * ms)
            s.append(spiketimes[spiketimes != 0] / ms)

        anfs = nest.Create(
            "spike_generator",
            NUM_CF * NUM_ANF_PER_HC,
            params={
                "spike_times": s,
                "allow_offgrid_times": [True] * NUM_CF * NUM_ANF_PER_HC,
            },
        )
        anfs_per_ear[channel] = anfs

    return anfs_per_ear
