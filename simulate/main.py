import datetime
from datetime import timedelta
from pathlib import Path
from random import seed
from timeit import default_timer as timer
import brian2 as b2
import brian2hears as b2h
import dill
import numpy as np
import nest
from brian2 import Hz
from utils.anf_utils import ZI_COC_KEY, CI_COC_KEY, create_sound_key, load_anf_response
from utils.cochlea_utils import ANGLES
from utils.path_utils import Paths, save_current_conf
from models.BrainstemModel.BrainstemModel import BrainstemModel
from models.BrainstemModel.params import Parameters as params
from utils.custom_sounds import Click, Tone, ToneBurst, WhiteNoise, Click_Train, HarmonicComplex
from utils.log_utils import logger, tqdm


nest.set_verbosity("M_ERROR")

def create_execution_key(i, p):
    return f"{create_sound_key(i)}&{p}"

def ex_key_with_time(*args):
    return f"{datetime.datetime.now().isoformat()[:-7]}&{create_execution_key(*args)}"

def create_save_result_object(
    input,
    gated_sound,
    left_sounds,
    right_sounds,
    cue_to_rate,
    MODE,
    model,
    param,
    cochlea_key,
    result_file,
    **kwargs,
):
    result = {}
    result["sounds"] = {
        "base_sound": input,
        "gated_sound": gated_sound,
        "left_sounds": left_sounds,
        "right_sounds": right_sounds,
    }
    result["cue_to_rate"] = cue_to_rate
    for key, arg in kwargs.items():
        result[key] = arg
    result["conf"] = save_current_conf(
        MODE, model, param, cochlea_key, create_sound_key(input)
    )
    logger.info(f"\tSaving results to {result_file.absolute()}...\n")
    with open(result_file, "wb") as f:
        dill.dump(result, f)
    del result


if __name__ == "__main__":

    TIME_SIMULATION = 100
    TIME_ON = 100
    TIME_OFF = TIME_SIMULATION - TIME_ON 
    RAMP_MS = 5     
    LEVEL = 55

    inputs = [
        # Tone(16 * b2.kHz, duration=TIME_ON * b2.ms, level=LEVEL * b2h.dB, ramp_ms=RAMP_MS, offset_silence_duration= TIME_OFF * b2.ms),
        #Click(duration=TIME_SIMULATION * b2.ms, click_duration=0.05*b2.ms, level=70 * b2h.dB),  
    Tone(17.6 * b2.kHz, duration=TIME_ON * b2.ms, level=LEVEL * b2h.dB, ramp_ms=RAMP_MS, offset_silence_duration= TIME_OFF * b2.ms),
    #Click_Train(duration=TIME_ON * b2.ms, click_duration=0.05*b2.ms, level=70 * b2h.dB, interval=5*b2.ms, offset_silence_duration= TIME_OFF * b2.ms),
    #     Click_Train(duration=TIME_ON * b2.ms, click_duration=0.05*b2.ms, level=70 * b2h.dB, interval=4*b2.ms, offset_silence_duration= TIME_OFF * b2.ms),
    #     Click_Train(duration=TIME_ON * b2.ms, click_duration=0.05*b2.ms, level=70 * b2h.dB, interval=3*b2.ms, offset_silence_duration= TIME_OFF * b2.ms),
    #     Click_Train(duration=TIME_ON * b2.ms, click_duration=0.05*b2.ms, level=70 * b2h.dB, interval=2*b2.ms, offset_silence_duration= TIME_OFF * b2.ms),
    #     Click_Train(duration=TIME_ON * b2.ms, click_duration=0.05*b2.ms, level=70 * b2h.dB, interval=1*b2.ms, offset_silence_duration= TIME_OFF * b2.ms),
    #     Click_Train(duration=TIME_SIMULATION * b2.ms, click_duration=0.05*b2.ms, level=70 * b2h.dB, interval=3*b2.ms),
    ]

    # CONFIGURATION
    MODE = "artificial_ild_exp"  # options: "angle", "artificial_itd", "artificial_ild"
    we = 5
    wi = -100
    experiment_folder = f'we_{we}_wi_{wi}' 
    
    if MODE == "angle":
        loop_range = ANGLES
    elif MODE == "artificial_itd":
        # loop_range = [0]
        loop_range = np.concatenate([
                    np.linspace(-5000, -1000, 8, endpoint=False),
                    np.linspace(-1000, 1000, 11),
                    np.linspace(1000, 5000, 9)[1:]
                ]) * 1e-6   # us to seconds
        #loop_range = np.linspace(-1000, 1000, 11) * 1e-6   # us to seconds
    elif MODE == "artificial_ild":
        loop_range = np.linspace(-25, 25, 11) # dB
    elif MODE == "artificial_ild_exp":
        loop_range = np.linspace(0, 90, 19) # dB


    models = [BrainstemModel]
    cochlea_key = ZI_COC_KEY

    ps = []

    # for m in ['itd_only', 'HRTF']:
    #     seed = 0
    #     rng = 42 + seed
    #     p = params(f"{m}")
    #     p.cochlea[ZI_COC_KEY]["rng_seed"] = rng
    #     p.CONFIG.NEST_KERNEL_PARAMS["rng_seed"] = rng
    #     p.cochlea[ZI_COC_KEY]['hrtf_params']['simulation_mode'] = MODE
    #     p.cochlea[ZI_COC_KEY]['hrtf_params']['cue_to_apply'] = m
    #     ps.append(p)

        # p.POP_CONV.MNTBCs2MSOs = 2
        # p.SYN_WEIGHTS.MNTBCs2MSO = -15
        # p.E_L.MSO = -55
        # p.V_RESET.MSO = -57
        # p.INH_REV.MSO = -75
        # p.TAUS_EX_RISE.MSO = 0.5
        # p.TAUS_EX_DECAY.MSO = 1.0

    for seed in range(2):
        rng = 42 + seed
        p = params(f"seed{seed}")
        p.cochlea[ZI_COC_KEY]["rng_seed"] = rng
        p.CONFIG.NEST_KERNEL_PARAMS["rng_seed"] = rng
        p.SYN_WEIGHTS.SBCs2LSO = we
        p.SYN_WEIGHTS.MNTBCs2LSO = wi
        p.cochlea[ZI_COC_KEY]['hrtf_params']['simulation_mode'] = MODE
        ps.append(p)


    num_runs = len(inputs) * len(ps)
    current_run = 0
    logger.info(f"launching {num_runs} trials...\n")
    times = {}
    result_dir = Path(Paths.RESULTS_DIR) / MODE / experiment_folder
    result_dir.mkdir(parents=True, exist_ok=True)
    trials_pbar = tqdm(total=num_runs, desc="trials")

    for Model in models:
        for input in inputs:
            for param in ps:
                result_paths = []
                L_sounds = {}
                R_sounds = {}
                gated_sound_global = None
                start = timer()
                ex_key = create_execution_key(input, param.key)
                logger.info(f">>>>> Now testing arch n.{current_run+1} of {num_runs}: {ex_key}\n")
                cue_to_rate = {}
                for val in tqdm(loop_range, desc=f"Looping {MODE}"):
                    nest.ResetKernel()
                    nest.SetKernelStatus(param.CONFIG.NEST_KERNEL_PARAMS)

                    logger.info(f"starting trial for {val}")
                    # this section is cached on disk
                    anf = load_anf_response(input, val, cochlea_key, param.cochlea)

                    L_sounds[val] = anf.left_sound
                    R_sounds[val] = anf.right_sound
                    if gated_sound_global is None:
                        gated_sound_global = anf.gated_sound
                    logger.info("ANF loaded. Creating model...")

                    model = Model(param, anf)
                    model.simulate(TIME_SIMULATION)

                    model_result = model.analyze()
                    logger.debug(
                        f"Left MSO is spiking at {len(model_result['L']['MSO']['times'])/TIME_SIMULATION*1000}Hz\n"
                        f"Left LSO is spiking at {len(model_result['L']['LSO']['times'])/TIME_SIMULATION*1000}Hz"
                    )
                    cue_to_rate[val] = model_result
                    logger.info("Trial Complete.")

                logger.info(f"Saving all values for model {ex_key}...")
                # save model results to file
                filename = f"{ex_key}.pic"
                result_file = result_dir / filename
                result_paths.append(result_file)

                end = timer()
                timetaken = timedelta(seconds=end - start)
                current_run = current_run + 1
                times[ex_key] = timetaken
                create_save_result_object(
                    input,
                    gated_sound_global,
                    L_sounds,
                    R_sounds,
                    cue_to_rate,
                    MODE,
                    model,
                    param,
                    cochlea_key,
                    result_file,
                    filename=filename,
                    simulation_time=TIME_SIMULATION,
                    times={"start": start, "end": end, "timetaken": timetaken},
            )

    trials_pbar.close()
    logger.debug(times)
    logger.info({k: str(v) for k, v in times.items()})
