"""
ILD50 parameter-sweep driver (SLURM job arrays, derived from main_linux.py).

Each task = one (configuration, seed) pair: the whole contra-level curve
(CONTRA_LEVELS) is simulated and the per-neuron spike counts of SAVE_POPS are
saved to a small .npz file. Sweeps are defined in ild50_sweep_config.py.

Usage (run from simulate/):
    python main_ild50_sweep.py list     --sweep oat [--chunk-size 10]
    python main_ild50_sweep.py prefetch --sweep oat --seed 3
    python main_ild50_sweep.py run      --sweep oat --chunk 7 --chunk-size 10 [--threads 14]

Sweeps with "timing": True also save, for the left LSO neuron with CF closest to
the tone, its g_ex/g_in traces and the spikes of its input tree (SBC, MNTBC,
contra GBC, ipsi/contra ANF), used to measure exc/inh arrival times.

`prefetch` runs the cochlea for one seed (no NEST import; the cochlea must run
before `import nest` on Linux). `run` only reads the ANF cache; tasks whose
output already exists are skipped, so failed/timed-out arrays can simply be
resubmitted.
"""
import argparse
import json
import os
from datetime import timedelta
from pathlib import Path
from timeit import default_timer as timer

import brian2 as b2
import brian2hears as b2h
import dill
import numpy as np

# ── Project utilities (no NEST yet) ──────────────────────────────────────────
from utils.log_utils import logger
from utils.path_utils import Paths, save_current_conf
from utils.custom_sounds import Tone
from utils.anf_utils import ZI_COC_KEY, create_sound_key, load_anf_response
from utils.cochlea_utils import CFMAX, CFMIN
from models.BrainstemModel.params import Parameters as params
import ild50_sweep_config as C

MODE = "artificial_ild"  # Zilany mode: right level = cue (absolute dB), left fixed
SIDES = ["L", "R"]

# =============================================================================
# Sweep -> tasks
# =============================================================================


def _get(p, path):
    group, attr = path.split(".")
    obj = getattr(p, group)
    if not hasattr(obj, attr):
        raise AttributeError(f"Unknown parameter '{path}'")
    return getattr(obj, attr)


def _set(p, path, value):
    group, attr = path.split(".")
    default = _get(p, path)
    if isinstance(default, int) and not isinstance(default, bool):
        if not float(value).is_integer():
            raise ValueError(f"{path} is an int parameter, got {value}")
        value = int(value)
    setattr(getattr(p, group), attr, value)


def effective_overrides(overrides):
    """Drop overrides equal to the params.py default and round floats (stable names)."""
    p = params()
    eff = {}
    for path, v in sorted(overrides.items()):
        v = round(float(v), 6)
        if not np.isclose(v, float(_get(p, path))):
            eff[path] = int(v) if isinstance(_get(p, path), int) else v
    return eff


def config_name(eff):
    if not eff:
        return "baseline"
    return "__".join(f"{k}={v:g}" for k, v in eff.items())


def sweep_paths(sweep):
    return sorted({k for cfg in C.SWEEPS[sweep]["configs"] for k in cfg})


def build_configs(sweep):
    """Unique configurations (name -> effective overrides), in definition order."""
    configs = {}
    for cfg in C.SWEEPS[sweep]["configs"]:
        eff = effective_overrides(cfg)
        configs.setdefault(config_name(eff), eff)
    return configs


def build_tasks(sweep):
    seeds = range(C.SWEEPS[sweep]["n_seeds"])
    return [(name, eff, seed) for name, eff in build_configs(sweep).items() for seed in seeds]


def make_params(eff, seed, threads=None):
    p = params(config_name(eff))
    rng = 42 + seed
    # cochlea dict must be identical to the prefetch one (joblib cache key)
    p.cochlea[ZI_COC_KEY]["rng_seed"] = rng
    p.cochlea[ZI_COC_KEY]["hrtf_params"]["simulation_mode"] = MODE
    p.CONFIG.NEST_KERNEL_PARAMS["rng_seed"] = rng
    if threads is not None:
        p.CONFIG.NEST_KERNEL_PARAMS["total_num_virtual_procs"] = threads
        p.CONFIG.NEST_KERNEL_PARAMS["local_num_threads"] = threads
    for path, v in eff.items():
        _set(p, path, v)
    return p


def make_tone(freq_hz, level_db):
    return Tone(
        freq_hz * b2.Hz,
        duration=C.TIME_ON * b2.ms,
        level=level_db * b2h.dB,
        ramp_ms=C.RAMP_MS,
        offset_silence_duration=(C.TIME_SIMULATION - C.TIME_ON) * b2.ms,
    )


def out_root(sweep, freq_hz, level_db):
    return (
        Path(os.environ.get("ILD50_SWEEP_DIR", Path(Paths.RESULTS_DIR) / "ild50_sweep"))
        / sweep
        / f"tone_{freq_hz:g}Hz_{level_db:g}dB"
    )


def out_file(root, name, seed):
    return root / name / f"seed_{seed:03d}.npz"


def check_config_baseline():
    """Warn if the baseline values hard-coded in the sweep config drifted from params.py."""
    p = params()
    for path, v in C._B.items():
        if not np.isclose(float(v), float(_get(p, path))):
            logger.warning(f"ild50_sweep_config._B[{path}]={v} but params.py default is {_get(p, path)}")


# =============================================================================
# Commands
# =============================================================================


def cmd_list(args):
    check_config_baseline()
    configs = build_configs(args.sweep)
    tasks = build_tasks(args.sweep)
    root = out_root(args.sweep, args.freq, args.ipsi_level)
    missing = [t for t in tasks if not out_file(root, t[0], t[2]).exists()]
    n_chunks = -(-len(tasks) // args.chunk_size)
    for name in configs:
        print(name)
    print(f"\noutput: {root.resolve()}")
    print(f"configs={len(configs)} seeds={C.SWEEPS[args.sweep]['n_seeds']} "
          f"tasks={len(tasks)} missing={len(missing)} sims={len(tasks) * len(C.CONTRA_LEVELS)}")
    # last lines are parsed by submit_ild50_sweep.sh
    print(f"N_SEEDS={C.SWEEPS[args.sweep]['n_seeds']}")
    print(f"N_CHUNKS={n_chunks}")


def prefetch(tone, seeds):
    for seed in seeds:
        p = make_params({}, seed)
        for val in C.CONTRA_LEVELS:
            load_anf_response(tone, val, ZI_COC_KEY, p.cochlea)


def cmd_prefetch(args):
    tone = make_tone(args.freq, args.ipsi_level)
    logger.info(f"=== prefetching ANF for seed {args.seed} ===")
    prefetch(tone, [args.seed])
    logger.info("=== prefetch complete ===")


def target_lso_index(n_neurons, freq_hz):
    """Index of the neuron with CF closest to freq_hz, as select_neurons() in plot/ild50_sweep_analysis.py
    (Greenwood human map over CFMIN..CFMAX, same as plot/analyze/report.greenwood_cf_array)."""
    A, a = 165.4, 2.1
    x = np.linspace(*(np.log10(np.array([CFMIN / b2.Hz, CFMAX / b2.Hz]) / A + 1) / a), n_neurons)
    cf = A * (10 ** (a * x) - 1)
    pos = int(np.searchsorted(cf, freq_hz))
    if pos in (0, n_neurons):
        return min(pos, n_neurons - 1)
    return pos if cf[pos] - freq_hz < freq_hz - cf[pos - 1] else pos - 1


def input_tree(nest, model, lso_idx):
    """Input tree (TIMING_TREE) of one left LSO neuron: the SBCs and MNTBCs projecting onto it, the
    contra GBCs driving those MNTBCs and the ANFs driving SBCs/GBCs. Returns {tree name: pop-relative
    indices}. Call it after Simulate: GetConnections before Simulate slows NEST down several times.
    Connectivity depends only on the seed, so the tree is the same at every contra level."""
    pops = model.pops
    nodes = {"LSO": pops["L"]["LSO"][lso_idx]}

    def sources(target, side, pop):
        gids = np.atleast_1d(nest.GetConnections(source=pops[side][pop], target=target).get("source"))
        return nest.NodeCollection(sorted(set(int(g) for g in gids)))

    for name, (side, pop, target) in C.TIMING_TREE.items():
        nodes[name] = sources(nodes[target], side, pop)
    return {name: np.asarray(n.get("global_id"), dtype=np.int64).ravel()
            - int(pops[C.TIMING_POPS[name][0]][C.TIMING_POPS[name][1]][0].get("global_id"))
            for name, n in nodes.items()}


def record_conductances(nest, model, lso_idx):
    """Multimeter on g_ex/g_in of one left LSO neuron. Created after the network, so population ids
    and dynamics are unchanged."""
    mm = nest.Create("multimeter", params={"record_from": ["g_ex", "g_in"], "interval": C.TIMING_INTERVAL_MS})
    nest.Connect(mm, model.pops["L"]["LSO"][lso_idx])
    return mm


def run_task(Model, nest, tone, eff, seed, path, threads, save_full, value_paths, timing=False):
    p = make_params(eff, seed, threads)
    counts = {}
    tree_idx, tree_spikes, g = None, {}, {"g_ex": [], "g_in": []}
    lso_idx = target_lso_index(p.POP_NUM.n_LSOs, float(tone.frequency / b2.Hz)) if timing else None
    ipsi_level, contra_level = None, []
    cue_to_rate, L_sounds, R_sounds = {}, {}, {}

    for i, val in enumerate(C.CONTRA_LEVELS):
        nest.ResetKernel()
        nest.SetKernelStatus(p.CONFIG.NEST_KERNEL_PARAMS)
        anf = load_anf_response(tone, val, ZI_COC_KEY, p.cochlea)
        if ipsi_level is None:
            ipsi_level = float(anf.left_sound.level)
        contra_level.append(float(anf.right_sound.level))

        model = Model(p, anf)
        if timing:
            mm = record_conductances(nest, model, lso_idx)
        model.simulate(C.TIME_SIMULATION)
        result = model.analyze()

        if timing:
            if tree_idx is None:
                tree_idx = input_tree(nest, model, lso_idx)
            ev = mm.get("events")
            t_mm = np.asarray(ev["times"])
            for k in g:
                g[k].append(np.asarray(ev[k], dtype=np.float32))
            for name, rel in tree_idx.items():
                side, pop = C.TIMING_POPS[name]
                e = result[side][pop]
                pos = np.asarray(e["senders"], dtype=np.int64) - int(np.atleast_1d(e["global_ids"])[0])
                keep = np.isin(pos, rel)
                s = tree_spikes.setdefault(name, {"times": [], "neuron": [], "level": []})
                s["times"].append(np.asarray(e["times"])[keep].astype(np.float32))
                s["neuron"].append(np.searchsorted(rel, pos[keep]).astype(np.uint16))
                s["level"].append(np.full(keep.sum(), i, dtype=np.uint8))

        for side in SIDES:
            for pop in C.SAVE_POPS:
                ev = result[side][pop]
                gids = np.atleast_1d(np.asarray(ev["global_ids"]))
                key = f"{side}_{pop}"
                if key not in counts:
                    counts[key] = np.zeros((len(C.CONTRA_LEVELS), len(gids)), dtype=np.uint16)
                senders = np.asarray(ev["senders"], dtype=np.int64) - int(gids[0])
                counts[key][i] = np.bincount(senders, minlength=len(gids))

        if save_full:
            cue_to_rate[val] = result
            L_sounds[val], R_sounds[val] = anf.left_sound, anf.right_sound

    timing_arrays = {}
    if timing:
        # one LSO neuron: conductances (levels x samples); input tree: pop-relative ids and the
        # spikes of each tree population, flattened with their tree-neuron and contra-level index
        timing_arrays = {"timing_t": t_mm.astype(np.float32),
                         **{f"timing_{k}": np.stack(v) for k, v in g.items()},
                         **{f"timing_idx_{n}": v for n, v in tree_idx.items()},
                         **{f"timing_{n}_{k}": np.concatenate(v) for n, s in tree_spikes.items() for k, v in s.items()}}

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.tmp.npz")
    np.savez_compressed(
        tmp,
        cues=np.asarray(C.CONTRA_LEVELS, dtype=float),
        contra_level_measured=np.asarray(contra_level),
        ipsi_level=ipsi_level,
        sim_time_ms=C.TIME_SIMULATION,
        tone_freq_hz=float(tone.frequency / b2.Hz),
        seed=seed,
        threads=p.CONFIG.NEST_KERNEL_PARAMS["total_num_virtual_procs"],
        overrides=json.dumps(eff),
        values=json.dumps({k: _get(p, k) for k in value_paths}),
        **{f"counts_{k}": v for k, v in counts.items()},
        **timing_arrays,
    )
    os.replace(tmp, path)

    if save_full:
        with open(path.with_suffix(".pic"), "wb") as f:
            dill.dump({
                "sounds": {"base_sound": tone, "gated_sound": tone.sound,
                           "left_sounds": L_sounds, "right_sounds": R_sounds},
                "cue_to_rate": cue_to_rate,
                "simulation_time": C.TIME_SIMULATION,
                "conf": save_current_conf(MODE, model, p, ZI_COC_KEY, create_sound_key(tone)),
            }, f)


def cmd_run(args):
    tone = make_tone(args.freq, args.ipsi_level)
    root = out_root(args.sweep, args.freq, args.ipsi_level)
    tasks = build_tasks(args.sweep)
    chunk = tasks[args.chunk * args.chunk_size:(args.chunk + 1) * args.chunk_size]
    todo = [t for t in chunk if not out_file(root, t[0], t[2]).exists()]
    logger.info(f"chunk {args.chunk}: {len(chunk)} tasks, {len(todo)} to run -> {root}")
    if not todo:
        return

    # Warm the ANF cache before NEST is imported (normally a no-op after `prefetch`)
    prefetch(tone, sorted({seed for _, _, seed in todo}))

    import nest
    from models.BrainstemModel.BrainstemModel import BrainstemModel

    nest.set_verbosity("M_ERROR")
    for name, eff, seed in todo:
        start = timer()
        logger.info(f">>>>> {name} | seed {seed}")
        run_task(BrainstemModel, nest, tone, eff, seed, out_file(root, name, seed),
                 args.threads, args.save_full, sweep_paths(args.sweep), C.SWEEPS[args.sweep].get("timing", False))
        logger.info(f"done in {timedelta(seconds=timer() - start)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for cmd in ["list", "prefetch", "run"]:
        s = sub.add_parser(cmd)
        s.add_argument("--sweep", required=True, choices=list(C.SWEEPS))
        s.add_argument("--freq", type=float, default=C.TONE_FREQ_HZ, help="tone frequency (Hz)")
        s.add_argument("--ipsi-level", type=float, default=C.IPSI_LEVEL, help="ipsi tone level (dB)")
        if cmd in ("list", "run"):
            s.add_argument("--chunk-size", type=int, default=1, help="tasks per array element")
        if cmd == "prefetch":
            s.add_argument("--seed", type=int, required=True)
        if cmd == "run":
            s.add_argument("--chunk", type=int, required=True, help="array index")
            s.add_argument("--threads", type=int, default=None,
                           help="NEST virtual procs (keep identical across the whole sweep)")
            s.add_argument("--save-full", action="store_true", help="also dump the full .pic result")
    args = parser.parse_args()
    {"list": cmd_list, "prefetch": cmd_prefetch, "run": cmd_run}[args.cmd](args)
