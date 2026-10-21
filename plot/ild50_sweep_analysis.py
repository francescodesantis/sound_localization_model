"""
Analysis of an ILD50 parameter sweep produced by simulate/main_ild50_sweep.py.

For every configuration, the rate-vs-contra-level curve of the LSO neuron whose
CF is closest to the tone frequency is averaged across seeds and fitted with
the same 4-parameter sigmoid used in paper_plots.ipynb
(extract_ild50_metrics); ILD50 = ipsi_level - half-point. Seeds are
bootstrapped to get confidence intervals, and differences from the baseline
are paired (same bootstrap seed indices for both), since all configurations
share the same seeds.

Usage (run from plot/):
    python ild50_sweep_analysis.py <sweep_dir>                 # e.g. .../ild50_sweep/oat/tone_16000Hz_60dB
    python ild50_sweep_analysis.py <sweep_dir> --pilot         # SE(ILD50) vs number of seeds
    python ild50_sweep_analysis.py <oat_dir> <oat2_dir> --out <dir>  # merge sweeps sharing seeds
Options: --pop LSO --side L --bw 0 --n-boot 500 --split

Outputs go to <sweep_dir>/analysis/: summary.csv, and figures for one-at-a-time
effects, tornado, seed-averaged curves, baseline curve with fit, 2D grids and (with --pilot) seed count.
If MNTBC counts were saved, also the MNTB diagnostic (mntb_*.pdf, mntb in
summary.csv): rate-level curves of the MNTB inhibiting the analysed LSO
(same side, driven by contra GBCs) and the shift of its half-max level
against the shift of the LSO half-point.
Functions can also be imported in a notebook.
"""
import argparse
import json
import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from brian2 import Hz

from analyze.report import extract_ild50_metrics, greenwood_cf_array, take_closest
from cochleas.consts import CFMAX, CFMIN

R2_MIN = 0.9  # fits below this are flagged
MIN_DYN_RANGE = 0.1  # (max-min)/max below this -> curve considered flat
MIN_MAX_RATE = 20.0  # Hz (~1 spike per trial in 60 ms); below this the LSO is considered silent

# =============================================================================
# Loading
# =============================================================================


def select_neurons(n_neurons, target_cf_hz, bw=0):
    """Indices of the neuron closest to target_cf_hz (+- bw neighbours), as in get_avg_rate_vs_cue."""
    cf = greenwood_cf_array(CFMIN / Hz, CFMAX / Hz, n_neurons) / Hz
    _, idx = take_closest(cf, target_cf_hz)
    idx = min(idx, n_neurons - 1)
    return np.arange(max(0, idx - bw), min(n_neurons - 1, idx + bw) + 1), cf[idx]


def load_sweep(sweep_dir, pop="LSO", side="L", bw=0):
    """
    Returns {config_name: {"values", "overrides", "seeds", "rates" (n_seeds x n_cues, Hz),
    "cues", "ipsi_level", "cf_hz", "threads"}}.
    """
    sweep_dir = Path(sweep_dir)
    configs = {}
    for cfg_dir in sorted(p for p in sweep_dir.iterdir() if p.is_dir() and p.name != "analysis"):
        files = sorted(cfg_dir.glob("seed_*.npz"))
        if not files:
            continue
        seeds, rates, ipsi, threads = [], [], [], set()
        for f in files:
            d = np.load(f)
            counts = d[f"counts_{side}_{pop}"]
            sel, cf_hz = select_neurons(counts.shape[1], float(d["tone_freq_hz"]), bw)
            rates.append(counts[:, sel].sum(axis=1) / (len(sel) * float(d["sim_time_ms"]) / 1000))
            seeds.append(int(d["seed"]))
            ipsi.append(float(d["ipsi_level"]))
            threads.add(int(d["threads"]))
        order = np.argsort(seeds)
        configs[cfg_dir.name] = {
            "values": json.loads(str(d["values"])),
            "overrides": json.loads(str(d["overrides"])),
            "seeds": np.asarray(seeds)[order],
            "rates": np.asarray(rates)[order],
            "cues": np.asarray(d["cues"], dtype=float),
            "ipsi_level": float(np.mean(ipsi)),
            "cf_hz": float(cf_hz),
            "threads": threads,
        }
    all_threads = set().union(*(c["threads"] for c in configs.values())) if configs else set()
    if len(all_threads) > 1:
        warnings.warn(f"Runs used different NEST thread counts {all_threads}: seeds are not paired across them.")
    return configs


def load_sweeps(sweep_dirs, pop="LSO", side="L", bw=0):
    """load_sweep over several sweep dirs (e.g. oat + oat2) merged into one dict.
    A config present in more than one dir (the baseline) must have identical data on the shared
    seeds, otherwise the sweeps are not paired; its seeds are pooled. Every config gets the values
    of all swept params (defaults filled in), so OAT tables span all sweeps."""
    merged = {}
    for d in sweep_dirs:
        for name, c in load_sweep(d, pop, side, bw).items():
            if name not in merged:
                merged[name] = c
                continue
            m = merged[name]
            common = np.intersect1d(m["seeds"], c["seeds"])
            if not np.array_equal(m["rates"][np.searchsorted(m["seeds"], common)],
                                  c["rates"][np.searchsorted(c["seeds"], common)]):
                raise ValueError(f"'{name}' differs between sweeps on the shared seeds {common}: not paired")
            new = ~np.isin(c["seeds"], m["seeds"])
            seeds = np.concatenate([m["seeds"], c["seeds"][new]])
            order = np.argsort(seeds)
            m["seeds"] = seeds[order]
            m["rates"] = np.concatenate([m["rates"], c["rates"][new]])[order]
            m["values"] = {**c["values"], **m["values"]}
            m["threads"] |= c["threads"]
    # default of each swept param: its value in any config that does not override it (configs of
    # different sweeps record different params, so the baseline alone may not list them all)
    defaults = {}
    for c in merged.values():
        for k, v in c["values"].items():
            if k not in c["overrides"]:
                defaults.setdefault(k, v)
    for c in merged.values():
        c["values"] = {**defaults, **c["values"]}
    return merged


# =============================================================================
# Fitting / bootstrap
# =============================================================================


def fit_curve(cues, rate, ipsi_level):
    """ILD50 metrics of one (seed-averaged) curve; NaNs if the fit fails."""
    out = dict(ild50=np.nan, half=np.nan, slope=np.nan, max_rate=float(np.max(rate)),
               min_rate=float(np.min(rate)), r2=np.nan, ok=False)
    if out["max_rate"] <= 0:
        return out
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = extract_ild50_metrics(cues, rate, normalize=False)
    except (RuntimeError, ValueError):
        return out
    out.update(ild50=ipsi_level - m["ild50"], half=m["ild50"], slope=m["slope"], r2=m["r_squared"])
    flat = (out["max_rate"] - out["min_rate"]) / out["max_rate"] < MIN_DYN_RANGE
    in_range = cues.min() <= m["ild50"] <= cues.max()
    silent = out["max_rate"] < MIN_MAX_RATE
    out["ok"] = bool(in_range and not flat and not silent and m["r_squared"] >= R2_MIN)
    return out


def bootstrap_ild50(rates, cues, ipsi_level, boot_idx):
    """ILD50 of the seed-averaged curve for each row of boot_idx (seed indices)."""
    return np.array([fit_curve(cues, rates[idx].mean(axis=0), ipsi_level)["ild50"] for idx in boot_idx])


def _boot_idx(n_seeds, n_boot):
    # same generator seed -> same resamples for configs with the same seed set (paired deltas)
    return np.random.default_rng(0).integers(0, n_seeds, size=(n_boot, n_seeds))


def summarize(configs, n_boot=500, baseline="baseline"):
    base = configs.get(baseline)
    base_vals = base["values"] if base else {}
    base_boot_cache = {}
    rows = []
    for name, c in configs.items():
        fit = fit_curve(c["cues"], c["rates"].mean(axis=0), c["ipsi_level"])
        boot = bootstrap_ild50(c["rates"], c["cues"], c["ipsi_level"], _boot_idx(len(c["seeds"]), n_boot))
        row = {"config": name, "n_seeds": len(c["seeds"]), "cf_hz": c["cf_hz"], "ipsi_level": c["ipsi_level"],
               **c["values"], **fit,
               "ild50_lo": np.nanpercentile(boot, 2.5), "ild50_hi": np.nanpercentile(boot, 97.5),
               "boot_fail_frac": float(np.mean(np.isnan(boot)))}

        varied = [k for k, v in c["values"].items() if k in base_vals and not np.isclose(v, base_vals[k])]
        row["varied"] = ",".join(varied)
        if base is not None:
            # paired delta on the seeds shared with the baseline
            common = np.intersect1d(c["seeds"], base["seeds"])
            ci = np.searchsorted(c["seeds"], common)
            bi = np.searchsorted(base["seeds"], common)
            idx = _boot_idx(len(common), n_boot)
            key = tuple(common)
            if key not in base_boot_cache:
                base_boot_cache[key] = bootstrap_ild50(base["rates"][bi], base["cues"], base["ipsi_level"], idx)
            d = bootstrap_ild50(c["rates"][ci], c["cues"], c["ipsi_level"], idx) - base_boot_cache[key]
            base_fit = fit_curve(base["cues"], base["rates"][bi].mean(axis=0), base["ipsi_level"])
            sub_fit = fit_curve(c["cues"], c["rates"][ci].mean(axis=0), c["ipsi_level"])
            row.update(delta_ild50=sub_fit["ild50"] - base_fit["ild50"],
                       delta_lo=np.nanpercentile(d, 2.5), delta_hi=np.nanpercentile(d, 97.5),
                       n_paired=len(common))
        rows.append(row)
    return pd.DataFrame(rows)


def half_max_level(cues, rate):
    """Level where an increasing rate-level curve first crosses (min+max)/2 (linear interpolation)."""
    lo, hi = float(np.min(rate)), float(np.max(rate))
    if hi <= 0 or (hi - lo) / hi < MIN_DYN_RANGE:
        return np.nan
    mid = (lo + hi) / 2
    i = int(np.argmax(rate >= mid))
    if i == 0:
        return float(cues[0])
    return float(np.interp(mid, [rate[i - 1], rate[i]], [cues[i - 1], cues[i]]))


def summarize_mntb(mntb_configs, baseline="baseline"):
    """Per config: MNTB max/min rate and half-max level of the seed-averaged curve, and its shift
    vs the baseline computed on the seeds shared with it (paired, like delta_ild50)."""
    base = mntb_configs.get(baseline)
    rows = []
    for name, c in mntb_configs.items():
        r = c["rates"].mean(axis=0)
        row = {"config": name, "mntb_max_rate": float(r.max()), "mntb_min_rate": float(r.min()),
               "mntb_half": half_max_level(c["cues"], r), "delta_mntb_half": np.nan}
        if base is not None:
            common = np.intersect1d(c["seeds"], base["seeds"])
            sub = c["rates"][np.searchsorted(c["seeds"], common)].mean(axis=0)
            ref = base["rates"][np.searchsorted(base["seeds"], common)].mean(axis=0)
            row["delta_mntb_half"] = half_max_level(c["cues"], sub) - half_max_level(base["cues"], ref)
        rows.append(row)
    return pd.DataFrame(rows)


def seed_count_curve(c, ns=None, n_rep=200):
    """SD of ILD50 across resamples of n seeds (with replacement): ≈ SE of ILD50 with n seeds."""
    n_tot = len(c["seeds"])
    ns = ns or [n for n in (3, 5, 8, 10, 15, 20, 25, 30, 40, 50) if n < n_tot] + [n_tot]
    rng = np.random.default_rng(0)
    out = []
    for n in ns:
        idx = rng.integers(0, n_tot, size=(n_rep, n))
        v = bootstrap_ild50(c["rates"], c["cues"], c["ipsi_level"], idx)
        out.append({"n_seeds": n, "se_ild50": np.nanstd(v), "fail_frac": float(np.mean(np.isnan(v)))})
    return pd.DataFrame(out)


# =============================================================================
# Plots
# =============================================================================


def _grid(n, ncols=4, size=(3.2, 2.6)):
    nrows = max(1, math.ceil(n / ncols))
    ncols = min(n, ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(size[0] * ncols, size[1] * nrows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, axes.flat


def oat_table(df):
    """{param: rows varying only that param, plus the baseline row, sorted by value}."""
    base = df[df["config"] == "baseline"]
    out = {}
    for p in sorted({v for v in df["varied"] if v and "," not in v}):
        out[p] = pd.concat([df[df["varied"] == p], base]).sort_values(p)
    return out


def _draw_oat(ax, p, t, metric):
    ok = t["ok"].astype(bool)
    y = t[metric].astype(float)
    ax.plot(t[p][ok], y[ok], "-", color="k", lw=1, zorder=1)
    if metric == "ild50":
        g = t[ok]
        yerr = np.vstack([g[metric] - g["ild50_lo"], g["ild50_hi"] - g[metric]])
        ax.errorbar(g[p], g[metric], yerr=yerr, fmt="none", ecolor="k", lw=1, capsize=2, zorder=1)
        span = pd.concat([g["ild50_lo"], g["ild50_hi"]])
    else:
        span = y[ok]
    # y range from the valid fits only; flagged points outside it are pinned to the edge (triangles)
    if span.notna().any():
        lo, hi = np.nanmin(span), np.nanmax(span)
        pad = 0.1 * (hi - lo) if hi > lo else 1.0
        lo, hi = lo - pad, hi + pad
        ax.set_ylim(lo, hi)
        bad = ~ok & y.notna()
        below, above = bad & (y < lo), bad & (y > hi)
        ax.plot(t[p][below], np.full(below.sum(), lo), "v", mfc="w", mec="k", ms=6, clip_on=False, zorder=2)
        ax.plot(t[p][above], np.full(above.sum(), hi), "^", mfc="w", mec="k", ms=6, clip_on=False, zorder=2)
    ax.plot(t[p][ok], y[ok], "o", color="k", ms=4, zorder=2)
    ax.plot(t[p][~ok], y[~ok], "o", mfc="w", mec="k", ms=4, zorder=2, label="flagged fit")
    b = t[t["config"] == "baseline"]
    ax.plot(b[p], b[metric], "D", color="C3", ms=6, zorder=3, label="baseline")
    ax.set_xlabel(p, fontsize=8)
    ax.set_ylabel({"ild50": "ILD50 [dB]", "max_rate": "max rate [Hz]", "slope": "slope [dB]",
                   "mntb_half": "MNTB half-max level [dB]",
                   "mntb_max_rate": "MNTB max rate [Hz]"}.get(metric, metric))


def plot_oat(df, metric="ild50"):
    tables = oat_table(df)
    fig, axes = _grid(len(tables))
    for ax, (p, t) in zip(axes, tables.items()):
        _draw_oat(ax, p, t, metric)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    return fig


def plot_tornado(df):
    """ΔILD50 at the lowest/highest tested value with a valid fit (flagged fits are excluded,
    and listed in the label), with paired 95% CI whiskers."""
    rows = []
    for p, t in oat_table(df).items():
        t = t[t["config"] != "baseline"]
        good = t[t["ok"].astype(bool)]
        bad = t[~t["ok"].astype(bool)][p]
        if good.empty:
            continue
        lo, hi = good.iloc[0], good.iloc[-1]  # smallest / largest value with a valid fit
        label = f"{p}  [{lo[p]:g} … {hi[p]:g}]"
        if len(bad):
            label += "  ✕ fails at " + ", ".join(f"{v:g}" for v in bad)
        rows.append((label, lo, hi))
    rows.sort(key=lambda r: np.nanmax(np.abs([r[1]["delta_ild50"], r[2]["delta_ild50"]])))
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(rows) + 1.2))
    y = np.arange(len(rows))
    for off, k, color, name in ((-0.2, 1, "C0", "lowest valid value"), (0.2, 2, "C3", "highest valid value")):
        d = np.array([r[k]["delta_ild50"] for r in rows])
        err = np.vstack([d - [r[k]["delta_lo"] for r in rows], [r[k]["delta_hi"] for r in rows] - d])
        ax.barh(y + off, d, height=0.4, color=color, label=name)
        ax.errorbar(d, y + off, xerr=err, fmt="none", ecolor="k", lw=0.8, capsize=2)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("ΔILD50 vs baseline [dB]", fontsize=11)
    ax.set_title("whiskers: paired 95% CI;  ✕: values with failed fit (excluded)", fontsize=8)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


def _draw_curves(ax, p, t, configs, ylabel):
    cmap = plt.get_cmap("viridis", len(t))
    for i, (_, row) in enumerate(t.iterrows()):
        c = configs[row["config"]]
        is_base = row["config"] == "baseline"
        ax.plot(c["cues"], c["rates"].mean(axis=0), "-", color="C3" if is_base else cmap(i),
                lw=2 if is_base else 1.2, label=f"{row[p]:g}")
    ax.set_title(p, fontsize=8)
    ax.set_xlabel("contra level [dB]", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.legend(fontsize=6, ncol=2)


def plot_curves(configs, df, ylabel="rate [Hz]"):
    tables = oat_table(df)
    fig, axes = _grid(len(tables))
    for ax, (p, t) in zip(axes, tables.items()):
        _draw_curves(ax, p, t, configs, ylabel)
    fig.tight_layout()
    return fig


def save_split(df, folder, draw, legend=False, dpi=200):
    """One PNG per OAT parameter in `folder` (same panel as the grid figure): draw(ax, param, table)."""
    folder.mkdir(parents=True, exist_ok=True)
    for p, t in oat_table(df).items():
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        draw(ax, p, t)
        # the grid panels use small hard-coded label sizes: make single panels uniform
        for txt in (ax.xaxis.label, ax.yaxis.label, ax.title):
            txt.set_fontsize(11)
            txt.set_fontweight("normal")
        ax.tick_params(labelsize=10)
        if legend:
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(folder / f"{p}.png", dpi=dpi)
        plt.close(fig)


def plot_mntb_vs_lso(df):
    """Shift of the LSO half-point vs shift of the MNTB half-max level, one point per OAT config.
    On the identity line the parameter moves ILD50 only by moving the MNTB threshold."""
    fig, ax = plt.subplots(figsize=(6.5, 5))
    base_half = df.loc[df["config"] == "baseline", "half"].iloc[0]
    tables = oat_table(df)
    cmap = plt.get_cmap("tab20")
    for i, (p, t) in enumerate(tables.items()):
        t = t[t["config"] != "baseline"]
        ok = t["ok"].astype(bool)
        x, y = t["delta_mntb_half"], t["half"] - base_half
        color, marker = cmap(i % 20), "osD^"[i // 20 % 4]  # tab20 has 20 colours: change marker after
        ax.plot(x[ok], y[ok], marker, color=color, ms=5, label=p)
        ax.plot(x[~ok], y[~ok], marker, mfc="w", mec=color, ms=5)
    lim = np.nanmax(np.abs(ax.get_xlim() + ax.get_ylim()))
    ax.plot([-lim, lim], [-lim, lim], "--", color="0.5", lw=0.8, zorder=0)
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("Δ MNTB half-max level [dB]")
    ax.set_ylabel("Δ LSO half-point [dB]  (= −ΔILD50)")
    ax.legend(fontsize=6, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig


def plot_grids(df):
    """Heatmap of ILD50 for every pair of params varied together (plus their 1D/baseline rows)."""
    figs = {}
    pairs = sorted({v for v in df["varied"] if v.count(",") == 1})
    for pair in pairs:
        p1, p2 = pair.split(",")
        t = df[df["varied"].isin([pair, p1, p2, ""])]
        piv = t.pivot_table(index=p2, columns=p1, values="ild50")
        if piv.shape[0] < 2 or piv.shape[1] < 2:
            continue
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        im = ax.imshow(piv.values, origin="lower", aspect="auto", cmap="viridis")
        ax.set_xticks(range(piv.shape[1]), [f"{v:g}" for v in piv.columns])
        ax.set_yticks(range(piv.shape[0]), [f"{v:g}" for v in piv.index])
        for (i, j), v in np.ndenumerate(piv.values):
            if np.isfinite(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7, color="w")
        ax.set_xlabel(p1)
        ax.set_ylabel(p2)
        fig.colorbar(im, ax=ax, label="ILD50 [dB]")
        fig.tight_layout()
        figs[pair] = fig
    return figs


def plot_baseline_curve(c, title="baseline"):
    """Rate vs contra level: single seeds (thin), seed mean ± SEM, sigmoid fit and ILD50."""
    rates, cues = c["rates"], c["cues"]
    mean = rates.mean(axis=0)
    sem = rates.std(axis=0, ddof=1) / np.sqrt(len(rates)) if len(rates) > 1 else np.zeros_like(mean)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for r in rates:
        ax.plot(cues, r, "-", color="0.8", lw=0.6, zorder=1)
    ax.errorbar(cues, mean, yerr=sem, fmt="o", color="k", ms=4, capsize=2, zorder=3,
                label=f"mean ± SEM ({len(rates)} seeds)")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = extract_ild50_metrics(cues, mean, normalize=False)
    x = np.linspace(cues.min(), cues.max(), 300)
    ax.plot(x, m["fit_curve_fn"](x), "-", color="C3", lw=1.5, zorder=2, label="sigmoid fit")
    ax.axvline(m["ild50"], color="C3", ls="--", lw=1)
    ax.set_title(f"{title}: ILD50 = {c['ipsi_level'] - m['ild50']:.1f} dB "
                 f"(half-point {m['ild50']:.1f} dB, ipsi {c['ipsi_level']:g} dB)", fontsize=9)
    ax.set_xlabel("contra level [dB]", fontsize=11)
    ax.set_ylabel(f"LSO rate [Hz]  (CF {c['cf_hz']:.0f} Hz)", fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_seed_count(sc, target_se=None):
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.plot(sc["n_seeds"], sc["se_ild50"], "o-", color="k")
    if target_se:
        ax.axhline(target_se, color="C3", ls="--", lw=1, label=f"target {target_se} dB")
        ax.legend(fontsize=8)
    ax.set_xlabel("number of seeds")
    ax.set_ylabel("SE of ILD50 [dB]")
    fig.tight_layout()
    return fig


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep_dirs", nargs="+", help="one sweep dir, or several to merge (e.g. oat and oat2)")
    ap.add_argument("--out", help="output dir (default <sweep_dir>/analysis; required when merging)")
    ap.add_argument("--pop", default="LSO")
    ap.add_argument("--side", default="L")
    ap.add_argument("--bw", type=int, default=0, help="neighbouring neurons on each side of the target CF")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--pilot", action="store_true", help="SE(ILD50) vs number of seeds for the baseline")
    ap.add_argument("--target-se", type=float, default=1.0, help="dB, used by --pilot")
    ap.add_argument("--split", action="store_true",
                    help="also save every OAT panel as its own PNG, in one folder per figure")
    args = ap.parse_args()

    if len(args.sweep_dirs) > 1 and not args.out:
        ap.error("--out is required when merging several sweep dirs")
    out = Path(args.out) if args.out else Path(args.sweep_dirs[0]) / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    configs = load_sweeps(args.sweep_dirs, args.pop, args.side, args.bw)
    print(f"loaded {len(configs)} configs, seeds per config: "
          f"{sorted({len(c['seeds']) for c in configs.values()})}")

    if args.pilot:
        sc = seed_count_curve(configs["baseline"])
        print(sc.to_string(index=False))
        enough = sc[sc["se_ild50"] <= args.target_se]
        print(f"-> smallest n with SE <= {args.target_se} dB: "
              f"{int(enough['n_seeds'].min()) if len(enough) else 'not reached'}")
        sc.to_csv(out / "seed_count.csv", index=False)
        plot_seed_count(sc, args.target_se).savefig(out / "seed_count.pdf")

    df = summarize(configs, n_boot=args.n_boot)
    if "baseline" in configs:
        plot_baseline_curve(configs["baseline"]).savefig(out / "baseline_curve.png", dpi=200)
    mntb_configs = None
    if args.pop == "LSO":
        try:
            mntb_configs = load_sweeps(args.sweep_dirs, "MNTBC", args.side, args.bw)
            df = df.merge(summarize_mntb(mntb_configs), on="config", how="left")
        except KeyError:
            print("no MNTBC counts saved: skipping MNTB diagnostic")
    df.to_csv(out / "summary.csv", index=False)
    cols = ["config", "n_seeds", "ild50", "ild50_lo", "ild50_hi", "delta_ild50", "delta_lo", "delta_hi",
            "max_rate", "slope", "r2", "ok"]
    print(df[[c for c in cols if c in df]].round(2).to_string(index=False))

    if any(v and "," not in v for v in df["varied"]):
        metrics = ["ild50", "max_rate", "slope"] + (["mntb_half", "mntb_max_rate"] if mntb_configs is not None else [])
        for metric in metrics:
            plot_oat(df, metric).savefig(out / f"oat_{metric}.pdf")
        plot_tornado(df).savefig(out / "tornado.pdf")
        plot_curves(configs, df).savefig(out / "curves.pdf")
        if mntb_configs is not None:
            plot_curves(mntb_configs, df, ylabel="MNTB rate [Hz]").savefig(out / "mntb_curves.pdf")
            plot_mntb_vs_lso(df).savefig(out / "mntb_vs_lso.pdf")
        if args.split:
            for metric in metrics:
                save_split(df, out / f"oat_{metric}", lambda ax, p, t, m=metric: _draw_oat(ax, p, t, m), legend=True)
            save_split(df, out / "curves", lambda ax, p, t: _draw_curves(ax, p, t, configs, "rate [Hz]"))
            if mntb_configs is not None:
                save_split(df, out / "mntb_curves",
                           lambda ax, p, t: _draw_curves(ax, p, t, mntb_configs, "MNTB rate [Hz]"))
            for name, fig in (("tornado", plot_tornado(df)),) + (
                    (("mntb_vs_lso", plot_mntb_vs_lso(df)),) if mntb_configs is not None else ()):
                (out / name).mkdir(exist_ok=True)
                fig.savefig(out / name / f"{name}.png", dpi=200)
    for pair, fig in plot_grids(df).items():
        fig.savefig(out / f"grid_{pair.replace(',', '__')}.pdf")
    print(f"results in {out.resolve()}")
