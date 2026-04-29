import datetime
from datetime import timedelta
from pathlib import Path
from timeit import default_timer as timer
import brian2 as b2
import brian2hears as b2h
import dill
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

def create_execution_key(i, c, p):
    return f"{create_sound_key(i)}&{c}&{p}"

def ex_key_with_time(*args):
    return f"{datetime.datetime.now().isoformat()[:-7]}&{create_execution_key(*args)}"

def create_save_result_object(
    input,
    gated_sound,
    l_hrtf_sounds,
    r_hrtf_sounds,
    angle_to_rate,
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
        "l_hrtf_sounds": l_hrtf_sounds,
        "r_hrtf_sounds": r_hrtf_sounds,
    }
    result["angle_to_rate"] = angle_to_rate
    for key, arg in kwargs.items():
        result[key] = arg
    result["conf"] = save_current_conf(
        model, param, cochlea_key, create_sound_key(input)
    )
    logger.info(f"\tSaving results to {result_file.absolute()}...")
    with open(result_file, "wb") as f:
        dill.dump(result, f)
    del result


if __name__ == "__main__":

    TIME_SIMULATION = 60 
    TIME_ON = 50
    TIME_OFF = TIME_SIMULATION - TIME_ON 
    RAMP_MS = 5     
    LEVEL = 60

    inputs = [
        Tone(0.5 * b2.kHz, duration=TIME_ON * b2.ms, level=LEVEL * b2h.dB, ramp_ms=RAMP_MS, offset_silence_duration= TIME_OFF * b2.ms),
        #Click(duration=TIME_SIMULATION * b2.ms, click_duration=0.05*b2.ms, level=70 * b2h.dB),
        #Click_Train(duration=TIME_SIMULATION * b2.ms, click_duration=0.05*b2.ms, level=70 * b2h.dB, interval=5*b2.ms),
    ]

    models = [BrainstemModel]
    cochlea_key = ZI_COC_KEY

    p = params("baseline")
    p1 = params("itd_only")
    p1.cochlea[cochlea_key]['hrtf_params']['cue_to_apply'] = "itd_only"
    p2 = params("ild_only")
    p2.cochlea[cochlea_key]['hrtf_params']['cue_to_apply'] = "ild_only" 
    ps = [p]

    # for d in [0.78,1.03,1.53,1.78,2.03]:
    #     p = params(f"delay_{d}ms_winh_{-2.3}")
    #     p.SYN_WEIGHTS.MNTBCs2LSO = -2.3
    #     p.SYN_WEIGHTS.LNTBCs2MSO = 0
    #     p.SYN_DELAYS.MNTBCs2MSO = d
    #     p.SYN_DELAYS.MNTBCs2LSO = d
    #     ps.append(p)

    num_runs = len(inputs) * len(ps)
    current_run = 0
    logger.info(f"launching {num_runs} trials...")
    times = {}
    result_dir = Path(Paths.RESULTS_DIR)
    trials_pbar = tqdm(total=num_runs, desc="trials")

    for Model in models:
        for input in inputs:
            for param in ps:
                curr_ex = f"{Model.key}&{cochlea_key}&{param.key}"
                result_paths = []
                L_sounds = {}
                R_sounds = {}
                gated_sound_global = None
                start = timer()
                ex_key = create_execution_key(input, cochlea_key, param.key)
                logger.info(f">>>>> Now testing arch n.{current_run+1} of {num_runs}")
                angle_to_rate = {}
                for angle in tqdm(ANGLES, "⮡ Angles"):
                    nest.ResetKernel()
                    nest.SetKernelStatus(param.CONFIG.NEST_KERNEL_PARAMS)

                    logger.info(f"starting trial for {angle}")
                    # this section is cached on disk
                    anf = load_anf_response(input, angle, cochlea_key, param.cochlea)
                    L_sounds[angle] = anf.l_hrtf_sound
                    R_sounds[angle] = anf.r_hrtf_sound
                    if gated_sound_global is None:
                        gated_sound_global = anf.gated_sound
                    logger.info("ANF loaded. Creating model...")

                    model = Model(param, anf)
                    model.simulate(TIME_SIMULATION)

                    model_result = model.analyze()
                    logger.debug(
                        f"Left MSO is spiking at {len(model_result['L']['MSO']['times'])/TIME_SIMULATION*1000}Hz"
                    )
                    angle_to_rate[angle] = model_result
                    logger.info("Trial Complete.")

                logger.info(f"Saving all angles for model {ex_key}...")
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
                    angle_to_rate,
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
