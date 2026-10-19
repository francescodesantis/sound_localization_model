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
Options: --pop LSO --side L --bw 0 --n-boot 500

Outputs go to <sweep_dir>/analysis/: summary.csv, and figures for one-at-a-time
effects, tornado, seed-averaged curves, 2D grids and (with --pilot) seed count.
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
    out["ok"] = bool(in_range and not flat and m["r_squared"] >= R2_MIN)
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


def plot_oat(df, metric="ild50"):
    tables = oat_table(df)
    fig, axes = _grid(len(tables))
    for ax, (p, t) in zip(axes, tables.items()):
        ok = t["ok"].astype(bool)
        if metric == "ild50":
            yerr = np.vstack([t[metric] - t["ild50_lo"], t["ild50_hi"] - t[metric]])
            ax.errorbar(t[p], t[metric], yerr=yerr, fmt="-", color="k", lw=1, capsize=2, zorder=1)
        else:
            ax.plot(t[p], t[metric], "-", color="k", lw=1, zorder=1)
        ax.plot(t[p][ok], t[metric][ok], "o", color="k", ms=4, zorder=2)
        ax.plot(t[p][~ok], t[metric][~ok], "o", mfc="w", mec="k", ms=4, zorder=2, label="flagged fit")
        b = t[t["config"] == "baseline"]
        ax.plot(b[p], b[metric], "D", color="C3", ms=6, zorder=3, label="baseline")
        ax.set_xlabel(p, fontsize=8)
        ax.set_ylabel({"ild50": "ILD50 [dB]", "max_rate": "max rate [Hz]", "slope": "slope [dB]"}.get(metric, metric))
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    return fig


def plot_tornado(df):
    rows = []
    for p, t in oat_table(df).items():
        t = t[t["config"] != "baseline"]
        lo, hi = t.iloc[0], t.iloc[-1]  # smallest / largest tested value
        rows.append((p, lo["delta_ild50"], hi["delta_ild50"], lo[p], hi[p]))
    rows.sort(key=lambda r: np.nanmax(np.abs(r[1:3])))
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(rows) + 1))
    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], color="C0", label="lowest value")
    ax.barh(y, [r[2] for r in rows], color="C3", alpha=0.7, label="highest value")
    ax.set_yticks(y, [f"{r[0]}  [{r[3]:g} … {r[4]:g}]" for r in rows], fontsize=8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("ΔILD50 vs baseline [dB]")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_curves(configs, df):
    tables = oat_table(df)
    fig, axes = _grid(len(tables))
    for ax, (p, t) in zip(axes, tables.items()):
        cmap = plt.get_cmap("viridis", len(t))
        for i, (_, row) in enumerate(t.iterrows()):
            c = configs[row["config"]]
            is_base = row["config"] == "baseline"
            ax.plot(c["cues"], c["rates"].mean(axis=0), "-", color="C3" if is_base else cmap(i),
                    lw=2 if is_base else 1.2, label=f"{row[p]:g}")
        ax.set_title(p, fontsize=8)
        ax.set_xlabel("contra level [dB]", fontsize=8)
        ax.set_ylabel("rate [Hz]", fontsize=8)
        ax.legend(fontsize=6, ncol=2)
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
    ap.add_argument("sweep_dir")
    ap.add_argument("--pop", default="LSO")
    ap.add_argument("--side", default="L")
    ap.add_argument("--bw", type=int, default=0, help="neighbouring neurons on each side of the target CF")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--pilot", action="store_true", help="SE(ILD50) vs number of seeds for the baseline")
    ap.add_argument("--target-se", type=float, default=1.0, help="dB, used by --pilot")
    args = ap.parse_args()

    out = Path(args.sweep_dir) / "analysis"
    out.mkdir(exist_ok=True)
    configs = load_sweep(args.sweep_dir, args.pop, args.side, args.bw)
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
    df.to_csv(out / "summary.csv", index=False)
    cols = ["config", "n_seeds", "ild50", "ild50_lo", "ild50_hi", "delta_ild50", "delta_lo", "delta_hi",
            "max_rate", "slope", "r2", "ok"]
    print(df[[c for c in cols if c in df]].round(2).to_string(index=False))

    if any(v and "," not in v for v in df["varied"]):
        for metric in ("ild50", "max_rate", "slope"):
            plot_oat(df, metric).savefig(out / f"oat_{metric}.pdf")
        plot_tornado(df).savefig(out / "tornado.pdf")
        plot_curves(configs, df).savefig(out / "curves.pdf")
    for pair, fig in plot_grids(df).items():
        fig.savefig(out / f"grid_{pair.replace(',', '__')}.pdf")
    print(f"results in {out.resolve()}")
