import datetime
from datetime import timedelta
from pathlib import Path
from timeit import default_timer as timer
import os


import brian2 as b2
import brian2hears as b2h
import dill
import numpy as np
from brian2 import Hz

# ── Project utilities (no NEST yet) ──────────────────────────────────────────
from utils.log_utils import logger, tqdm
from utils.path_utils import Paths, save_current_conf
from utils.cochlea_utils import ANGLES
from utils.custom_sounds import Click, Tone, ToneBurst, WhiteNoise, Click_Train, HarmonicComplex
from utils.anf_utils import ZI_COC_KEY, CI_COC_KEY, create_sound_key, load_anf_response
from models.BrainstemModel.params import Parameters as params

# =============================================================================
# CONFIGURATION  (edit here)
# =============================================================================

TIME_SIMULATION = 60
TIME_ON         = 50
TIME_OFF        = TIME_SIMULATION - TIME_ON
RAMP_MS         = 5
LEVEL           = 60

MODE = "artificial_ild"   # "angle" | "artificial_itd" | "artificial_ild" | "artificial_ild_exp"

SEED = int(os.environ.get("SLURM_SEED", 0))   # fallback to 0 for local runs

nucleus = 'LSO'
we = 5
wi = -80
d = 0.78
test_folder = f"we_{we}_wi_{wi}"

inputs = [
    # Click_Train(duration=TIME_ON * b2.ms, click_duration=0.05*b2.ms, level=70 * b2h.dB,
    #              interval=5*b2.ms, offset_silence_duration= TIME_OFF * b2.ms),
    # Tone(0.5 * b2.kHz, duration=TIME_ON * b2.ms, level=LEVEL * b2h.dB,
    #      ramp_ms=RAMP_MS, offset_silence_duration=TIME_OFF * b2.ms),
    Tone(19 * b2.kHz, duration=TIME_ON * b2.ms, level=LEVEL * b2h.dB,
         ramp_ms=RAMP_MS, offset_silence_duration=TIME_OFF * b2.ms),
]


# =============================================================================
# Derived loop ranges
# =============================================================================

if MODE == "angle":
    loop_range = ANGLES
elif MODE == "artificial_itd":
    loop_range = np.concatenate([
        np.linspace(-5000, -1000,  8, endpoint=False),
        np.linspace(-1000,  1000, 11),
        np.linspace( 1000,  5000,  9)[1:]
    ]) * 1e-6
elif MODE == "artificial_ild":
    loop_range = np.linspace(-25, 25, 11)
elif MODE == "artificial_ild_exp":
    loop_range = np.linspace(0, 90, 19) # dB

cochlea_key = ZI_COC_KEY

# =============================================================================
# PHASE 1 — Build param sets
#   prefetch_ps : one entry per unique ANF cache key (seed only, no delay)
#   ps          : full cartesian product (seed x delay) for NEST simulation
# =============================================================================

prefetch_ps = []
rng = 42 + SEED
p = params(f"prefetch_seed_{SEED}")
p.cochlea[ZI_COC_KEY]["rng_seed"] = rng
p.cochlea[ZI_COC_KEY]['hrtf_params']['simulation_mode'] = MODE
p.cochlea[ZI_COC_KEY]['hrtf_params']['cue_to_apply'] = 'ild_only'
prefetch_ps.append(p)

ps = []
rng = 42 + SEED
p = params(f"d_{d}_seed_{SEED}")
p.cochlea[ZI_COC_KEY]["rng_seed"] = rng
p.CONFIG.NEST_KERNEL_PARAMS["rng_seed"] = rng
p.SYN_WEIGHTS.SBCs2LSO = we
p.SYN_WEIGHTS.MNTBCs2LSO = wi
p.SYN_DELAYS.MNTBCs2LSO = d

p.cochlea[ZI_COC_KEY]['hrtf_params']['simulation_mode'] = MODE
ps.append(p)

# =============================================================================
# PHASE 1 — Pre-generate / warm ANF cache (cochlea runs here, NEST not loaded)
# =============================================================================

def prefetch_anf(inputs, loop_range, cochlea_key, prefetch_ps):
    logger.info("=== PHASE 1: pre-generating ANF responses (before NEST init) ===")
    for input in inputs:
        for p in prefetch_ps:
            for val in loop_range:
                load_anf_response(input, val, cochlea_key, p.cochlea)
    logger.info("=== PHASE 1 complete ===\n")


prefetch_anf(inputs, loop_range, cochlea_key, prefetch_ps)

# =============================================================================
# PHASE 2 — Now safe to import NEST (cochlea will never run again)
# =============================================================================

import nest
from models.BrainstemModel.BrainstemModel import BrainstemModel

nest.set_verbosity("M_ERROR")

# =============================================================================
# Helpers
# =============================================================================

def create_execution_key(i, p):
    return f"{create_sound_key(i)}&{p}"

def create_save_result_object(
    input, gated_sound, left_sounds, right_sounds,
    cue_to_rate, MODE, model, param, cochlea_key, result_file, **kwargs,
):
    result = {}
    result["sounds"] = {
        "base_sound":   input,
        "gated_sound":  gated_sound,
        "left_sounds":  left_sounds,
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

# =============================================================================
# PHASE 3 — Main simulation loop (NEST active, ANF always from cache)
# =============================================================================

if __name__ == "__main__":

    Model = BrainstemModel
    num_runs    = len(inputs) * len(ps)
    current_run = 0
    logger.info(f"launching {num_runs} trials...\n")
    times      = {}
    result_dir = Path(Paths.RESULTS_DIR) / MODE / nucleus / test_folder
    result_dir.mkdir(parents=True, exist_ok=True)
    trials_pbar = tqdm(total=num_runs, desc="trials")


    for input in inputs:
        for param in ps:
            L_sounds           = {}
            R_sounds           = {}
            gated_sound_global = None
            result_paths       = []
            start              = timer()
            ex_key             = create_execution_key(input, param.key)
            logger.info(f">>>>> Now testing arch n.{current_run+1} of {num_runs}: {ex_key}\n")
            cue_to_rate = {}

            for val in loop_range:
                nest.ResetKernel()
                nest.SetKernelStatus(param.CONFIG.NEST_KERNEL_PARAMS)

                logger.info(f"starting trial for {val}")
                # ANF already cached — reads from disk, cochlea not called
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
                    f"Left MSO is spiking at "
                    f"{len(model_result['L']['MSO']['times'])/TIME_SIMULATION*1000}Hz\n"
                    f"Left LSO is spiking at "
                    f"{len(model_result['L']['LSO']['times'])/TIME_SIMULATION*1000}Hz"
                )
                cue_to_rate[val] = model_result
                logger.info("Trial Complete.")

            logger.info(f"Saving all values for model {ex_key}...")
            filename    = f"{ex_key}.pic"
            result_file = result_dir / filename
            result_paths.append(result_file)

            end       = timer()
            timetaken = timedelta(seconds=end - start)
            current_run += 1
            times[ex_key] = timetaken

            create_save_result_object(
                input, gated_sound_global, L_sounds, R_sounds,
                cue_to_rate, MODE, model, param, cochlea_key, result_file,
                filename        = filename,
                simulation_time = TIME_SIMULATION,
                times           = {"start": start, "end": end, "timetaken": timetaken},
            )

    trials_pbar.close()
    logger.debug(times)
    logger.info({k: str(v) for k, v in times.items()})
