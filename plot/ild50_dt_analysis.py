"""
Relative excitation/inhibition timing at the LSO vs ILD50 (sweeps "dt" and "dt_invariance" of
simulate/ild50_sweep_config.py, optionally merged with the delay configs of the oat results).

Sign convention: dt < 0 inhibition leads, dt > 0 excitation leads.
- dt_syn = (ANFs2GBCs + GBCs2MNTBCs + MNTBCs2LSO) - (ANFs2SBCs + SBCs2LSO): synaptic delays only.
- dt_eff(L) = t_inh - t_exc measured at the analysed LSO neuron, as the half-rise times of the
  seed-averaged g_in and g_ex onsets at each contra level L. It includes the ANF latencies and the
  integration times of SBC, GBC and MNTB. Stage latencies (half-rise of the input-tree PSTHs,
  at the spiking neuron, before the synaptic delay to the next stage) decompose it.
  Only sweeps run with "timing": True have these recordings. For the other delay configs dt_eff is
  predicted as baseline dt_eff(L) + (dt_syn - dt_syn_baseline) (the network is feedforward, so
  synaptic delays add to the arrival times); the configs with recordings check this additivity.
  dt_eff at ILD 0 (contra = ipsi level) is the top x axis of ild50_vs_dt.pdf and column dt_eff_ild0.

Invariance: configs with the same dt_syn must have identical LSO and MNTB spike counts on every
shared seed (a common shift of both branches only delays the LSO output).

Usage (run from plot/):
    python ild50_dt_analysis.py <dt_dir> [<dt_invariance_dir> <oat_dir>] [--out <dir>]
Options: --bw 0 --n-boot 500 --window 50
Outputs (<dt_dir>/analysis/ by default): dt_summary.csv, invariance.csv, latencies.csv,
ild50_vs_dt.pdf, latencies.pdf, dt_eff.pdf.
"""
import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ild50_sweep_analysis import load_sweeps, select_neurons, summarize
from simulate.models.BrainstemModel.params import Parameters

EXC_PATH = ["SYN_DELAYS.ANFs2SBCs", "SYN_DELAYS.SBCs2LSO"]
INH_PATH = ["SYN_DELAYS.ANFs2GBCs", "SYN_DELAYS.GBCs2MNTBCs", "SYN_DELAYS.MNTBCs2LSO"]
STAGES = ["ANF_ipsi", "SBC", "ANF_contra", "GBC", "MNTBC"]
PSTH_BIN = 0.1  # ms
PSTH_SMOOTH = 5  # bins (boxcar)
G_IN_MIN_PEAK = 0.1  # g_in onset below this fraction of its largest peak (over levels) -> no inhibition
REF_LEVEL = 60.0  # contra level [dB] where the nominal ILD is 0 (ipsi 60 dB): reference for dt_eff

_P = Parameters()


def _default(path):
    group, attr = path.split(".")
    return float(getattr(getattr(_P, group), attr))


def dt_syn(values):
    v = lambda k: float(values.get(k, _default(k)))
    return round(sum(v(k) for k in INH_PATH) - sum(v(k) for k in EXC_PATH), 6)


def is_delay_config(values):
    return all(k.startswith("SYN_DELAYS.") for k, v in values.items() if not np.isclose(v, _default(k)))


def config_dirs(sweep_dirs):
    """{config name: [dirs holding it]} (a config may appear in several sweeps, e.g. the baseline)."""
    out = {}
    for d in sweep_dirs:
        for p in sorted(Path(d).iterdir()):
            if p.is_dir() and p.name != "analysis" and any(p.glob("seed_*.npz")):
                out.setdefault(p.name, []).append(p)
    return out


# =============================================================================
# Invariance
# =============================================================================


def check_invariance(groups, dirs, bw=0):
    """Within each dt_syn group, compare every config with the first one on shared seeds.
    Reports the absolute spike-count difference of the analysed LSO neuron(s) and of all left
    LSO and MNTB neurons. Only relative timing is preserved: the absolute arrival time differs,
    so neurons whose response overlaps the start (initial V_m relaxing to E_L) or the end of the
    simulation can differ; the analysed neuron is the one the ILD50 claim rests on."""
    rows = []
    for dt, names in groups.items():
        ref = names[0]
        for name in names[1:]:
            files = {f.name: f for f in dirs[name][0].glob("seed_*.npz")}
            ref_files = {f.name: f for f in dirs[ref][0].glob("seed_*.npz")}
            shared = sorted(set(files) & set(ref_files))
            n_sel = n_lso = n_mntb = 0
            for s in shared:
                a, b = np.load(files[s]), np.load(ref_files[s])
                d_lso = np.abs(a["counts_L_LSO"].astype(int) - b["counts_L_LSO"].astype(int))
                sel, _ = select_neurons(d_lso.shape[1], float(a["tone_freq_hz"]), bw)
                n_sel += int(d_lso[:, sel].sum())
                n_lso += int(d_lso.sum())
                n_mntb += int(np.abs(a["counts_L_MNTBC"].astype(int) - b["counts_L_MNTBC"].astype(int)).sum())
            rows.append(dict(dt_syn=dt, config=name, reference=ref, n_seeds=len(shared),
                             diff_analysed=n_sel, diff_all_LSO=n_lso, diff_all_MNTB=n_mntb,
                             identical_analysed=n_sel == 0 and len(shared) > 0,
                             identical_all=n_sel + n_lso + n_mntb == 0 and len(shared) > 0))
    return pd.DataFrame(rows)


# =============================================================================
# Timing
# =============================================================================


def half_rise(t, y, window, min_peak=0.0):
    """First time y reaches half of its maximum within t < window (linear interpolation)."""
    m = t < window
    t, y = t[m], y[m]
    peak = y.max() if len(y) else 0.0
    if peak <= min_peak:
        return np.nan
    i = int(np.argmax(y >= peak / 2))
    if i == 0:
        return float(t[0])
    return float(np.interp(peak / 2, [y[i - 1], y[i]], [t[i - 1], t[i]]))


def load_timing(cfg_dir, window):
    """Seed-averaged g traces and tree PSTHs of one config -> per-level arrival times (ms)."""
    files = sorted(cfg_dir.glob("seed_*.npz"))
    if not files or "timing_g_ex" not in np.load(files[0]).files:
        return None
    d0 = np.load(files[0])
    cues, t, sim = d0["cues"], d0["timing_t"].astype(float), float(d0["sim_time_ms"])
    sel, _ = select_neurons(d0["counts_L_LSO"].shape[1], float(d0["tone_freq_hz"]))
    if int(d0["timing_idx_LSO"][0]) != int(sel[0]):
        raise ValueError(f"{cfg_dir}: recorded LSO neuron {d0['timing_idx_LSO']} is not the analysed one {sel}")
    g_ex = np.mean([np.load(f)["timing_g_ex"] for f in files], axis=0)
    g_in = np.mean([np.load(f)["timing_g_in"] for f in files], axis=0)
    edges = np.arange(0, sim + PSTH_BIN, PSTH_BIN)
    centers = edges[:-1] + PSTH_BIN / 2
    kernel = np.ones(PSTH_SMOOTH) / PSTH_SMOOTH
    out = {"cues": cues, "g_ex": [], "g_in": []}
    g_in_floor = G_IN_MIN_PEAK * g_in[:, t < window].max()
    for i in range(len(cues)):
        out["g_ex"].append(half_rise(t, g_ex[i], window))
        out["g_in"].append(half_rise(t, g_in[i], window, g_in_floor))
    for stage in STAGES:
        times = np.concatenate([np.load(f)[f"timing_{stage}_times"] for f in files])
        level = np.concatenate([np.load(f)[f"timing_{stage}_level"] for f in files])
        out[stage] = [half_rise(centers, np.convolve(np.histogram(times[level == i], edges)[0], kernel, "same"),
                                window) for i in range(len(cues))]
    out = {k: np.asarray(v, dtype=float) for k, v in out.items()}
    out["dt_eff"] = out["g_in"] - out["g_ex"]
    return out


def at_level(cues, y, level):
    ok = np.isfinite(y)
    if np.isnan(level) or ok.sum() < 2 or not (cues[ok].min() <= level <= cues[ok].max()):
        return np.nan
    return float(np.interp(level, cues[ok], y[ok]))


# =============================================================================
# Plots
# =============================================================================


def plot_ild50_vs_dt(df, eff_offset=None):
    """eff_offset = dt_eff(ILD 0) - dt_syn at baseline: adds a top axis in dt_eff at ILD 0."""
    df = df.sort_values("dt_syn")
    ok = df["ok"].astype(bool)
    base = df[df["config"] == "baseline"]["dt_syn"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    ax = axes[0]
    ax.errorbar(df["dt_syn"][ok], df["delta_ild50"][ok],
                yerr=[df["delta_ild50"][ok] - df["delta_lo"][ok], df["delta_hi"][ok] - df["delta_ild50"][ok]],
                fmt="o-", color="k", ms=4, capsize=2)
    ax.plot(df["dt_syn"][~ok], df["delta_ild50"][~ok], "x", color="C3", label="failed fit")
    ax.set_ylabel("ΔILD50 vs baseline [dB]")
    axes[1].plot(df["dt_syn"], df["max_rate"], "o-", ms=4, label="max")
    axes[1].plot(df["dt_syn"], df["min_rate"], "o-", ms=4, label="min")
    axes[1].set_ylabel("LSO rate [Hz]")
    axes[1].legend(fontsize=8)
    axes[2].plot(df["dt_syn"][ok], df["slope"][ok], "o-", color="k", ms=4)
    axes[2].set_ylabel("sigmoid slope [dB]")
    for ax in axes:
        ax.axvline(0, color="0.6", lw=0.8)
        for b in base:
            ax.axvline(b, color="C3", ls="--", lw=0.8)
        ax.set_xlabel("Δt_syn [ms]  (<0 inhibition leads)")
        if eff_offset is not None and np.isfinite(eff_offset):
            top = ax.secondary_xaxis("top", functions=(lambda x: x + eff_offset, lambda x: x - eff_offset))
            top.set_xlabel("Δt_eff at ILD 0 [ms]")
    fig.tight_layout()
    return fig


def plot_latencies(tm, title="baseline"):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for k, style in [("ANF_ipsi", "C2:"), ("SBC", "C2-"), ("g_ex", "C2o-"),
                     ("ANF_contra", "C3:"), ("GBC", "C3--"), ("MNTBC", "C3-"), ("g_in", "C3o-")]:
        ax.plot(tm["cues"], tm[k], style, ms=3, label=k + (" (at LSO)" if k.startswith("g_") else ""))
    ax.set_xlabel("contra level [dB]")
    ax.set_ylabel("half-rise time of onset [ms]")
    ax.set_title(f"{title}: stage latencies (green exc, red inh)", fontsize=9)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return fig


def plot_dt_eff(df, timing):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    cmap = plt.get_cmap("viridis")
    lo, hi = df["dt_syn"].min(), df["dt_syn"].max()
    for name, tm in timing.items():
        row = df[df["config"] == name].iloc[0]
        c = cmap((row["dt_syn"] - lo) / (hi - lo or 1))
        axes[0].plot(tm["cues"], tm["dt_eff"], "-", color=c, lw=1)
        axes[0].plot(row["half"], row["dt_eff_at_half"], "o", color=c, ms=4)
    axes[0].axhline(0, color="0.6", lw=0.8)
    axes[0].set_xlabel("contra level [dB]")
    axes[0].set_ylabel("Δt_eff = t_inh − t_exc [ms]")
    axes[0].set_title("measured; dot = ILD50 half-point level", fontsize=9)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(lo, hi))
    fig.colorbar(sm, ax=axes[0], label="Δt_syn [ms]")
    ok = df["ok"].astype(bool)
    for src, mk in (("measured", "o"), ("predicted", "s")):
        m = ok & (df["dt_eff_source"] == src)
        axes[1].plot(df["dt_syn"][m], df["dt_eff_at_half"][m], mk, mfc="none" if src == "predicted" else None,
                     color="k", ms=5, label=src)
    axes[1].set_xlabel("Δt_syn [ms]")
    axes[1].set_ylabel("Δt_eff at the ILD50 level [ms]")
    axes[1].set_title("constant → ILD50 set by a fixed inhibitory lead", fontsize=9)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    return fig


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep_dirs", nargs="+", help="dt sweep dir, optionally dt_invariance / oat dirs")
    ap.add_argument("--out", help="output dir (default <first sweep_dir>/analysis)")
    ap.add_argument("--bw", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--window", type=float, default=50.0, help="ms; onset peaks are searched before this time")
    args = ap.parse_args()

    out = Path(args.out) if args.out else Path(args.sweep_dirs[0]) / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    configs = load_sweeps(args.sweep_dirs, "LSO", "L", args.bw)
    configs = {k: c for k, c in configs.items() if is_delay_config(c["values"])}
    dirs = config_dirs(args.sweep_dirs)
    print(f"{len(configs)} delay configs (incl. baseline)")

    # one row per config: ILD50 metrics with paired CIs vs baseline, and dt_syn
    df = summarize(configs, n_boot=args.n_boot)
    df["dt_syn"] = [dt_syn(configs[n]["values"]) for n in df["config"]]

    # invariance: identical spike counts within each dt_syn group
    groups = {}
    for n in sorted(configs, key=lambda n: (n != "baseline", n)):
        groups.setdefault(dt_syn(configs[n]["values"]), []).append(n)
    inv = check_invariance({k: v for k, v in groups.items() if len(v) > 1}, dirs, args.bw)
    inv.to_csv(out / "invariance.csv", index=False)
    if len(inv):
        print(f"\ninvariance vs the reference with the same dt_syn: analysed neuron identical in "
              f"{int(inv['identical_analysed'].sum())}/{len(inv)} configs, whole LSO+MNTB in "
              f"{int(inv['identical_all'].sum())}/{len(inv)}")
        print(inv.to_string(index=False))

    # timing: measured where recorded, otherwise predicted from the baseline by additivity
    timing = {}
    for n in configs:
        for d in dirs[n]:
            tm = load_timing(d, args.window)
            if tm is not None:
                timing[n] = tm
                break
    print(f"timing recordings for {len(timing)} configs")
    base_tm = timing.get("baseline")
    dt0 = dt_syn(configs["baseline"]["values"])
    meas, src, err, ref = [], [], [], []
    for _, row in df.iterrows():
        tm = timing.get(row["config"])
        pred = None if base_tm is None else base_tm["dt_eff"] + (row["dt_syn"] - dt0)
        if tm is not None:
            meas.append(at_level(tm["cues"], tm["dt_eff"], row["half"]))
            ref.append(at_level(tm["cues"], tm["dt_eff"], REF_LEVEL))
            src.append("measured")
            err.append(np.nanmax(np.abs(tm["dt_eff"] - pred)) if pred is not None and np.isfinite(tm["dt_eff"] - pred).any()
                       else np.nan)
        else:
            meas.append(np.nan if pred is None else at_level(base_tm["cues"], pred, row["half"]))
            ref.append(np.nan if pred is None else at_level(base_tm["cues"], pred, REF_LEVEL))
            src.append("none" if pred is None else "predicted")
            err.append(np.nan)
    df["dt_eff_at_half"], df["dt_eff_source"], df["additivity_err_ms"] = meas, src, err
    df["dt_eff_ild0"] = ref
    df = df.sort_values("dt_syn")
    df.to_csv(out / "dt_summary.csv", index=False)
    cols = ["config", "dt_syn", "ild50", "delta_ild50", "delta_lo", "delta_hi", "max_rate", "min_rate", "slope",
            "half", "dt_eff_ild0", "dt_eff_at_half", "dt_eff_source", "additivity_err_ms", "ok"]
    print(df[cols].round(3).to_string(index=False))

    eff_offset = (at_level(base_tm["cues"], base_tm["dt_eff"], REF_LEVEL) - dt0) if base_tm is not None else None
    plot_ild50_vs_dt(df, eff_offset).savefig(out / "ild50_vs_dt.pdf")
    if base_tm is not None:
        lat = pd.DataFrame({k: v for k, v in base_tm.items()})
        lat.to_csv(out / "latencies.csv", index=False)
        print("\nbaseline half-rise times [ms] per contra level:")
        print(lat.round(3).to_string(index=False))
        plot_latencies(base_tm).savefig(out / "latencies.pdf")
    if timing:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plot_dt_eff(df, timing).savefig(out / "dt_eff.pdf")
    print(f"results in {out.resolve()}")
