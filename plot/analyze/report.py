from bisect import bisect_left
from collections import defaultdict
import math
import brian2 as b2
from brian2 import Hz
import brian2hears as b2h
from brian2hears import erbspace
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sorcery import dict_of
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from cochleas.consts import CFMAX, CFMIN
from utils.custom_sounds import Tone, ToneBurst
import sys, os
from scipy.optimize import curve_fit
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
simulate_repo = PROJECT_ROOT + '/simulate'
sys.path.insert(0, simulate_repo)
from utils.anf_utils import create_sound_key

plt.rcParams["axes.grid"] = False
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.titleweight']= 'bold'
plt.rcParams['axes.spines.top']= False
plt.rcParams['axes.spines.right']= False
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 10   # Size of x-axis tick labels
plt.rcParams['ytick.labelsize'] = 10   # Size of y-axis tick labels
plt.rcParams['legend.fontsize'] = 10   # Size of the legend text
# Make axis labels bold
plt.rcParams['axes.labelweight'] = 'bold'  # Makes x and y axis labels bold

POPULATION_COLORS = {
    "ANF":  "#0077BB",  # blue
    "SBC":  "#21A530",  # orange
    "GBC":  "#009988",  # teal
    "MNTBC": "#CC3311",  # red
    "LNTBC": "#82240C",  # red
    "LSO":  "#EB9020",  # magenta
    "MSO":  "#66C2F0",  # cyan
}

def greenwood_human(x, A=165.4, a=2.1, k=1.0):
    return A * (10**(a * x) - k)

def greenwood_inverse(f, A=165.4, a=2.1, k=1.0):
    # compute x from frequency (inverse Greenwood)
    return (1.0 / a) * np.log10(f / A + k)

def greenwood_cf_array(CFMIN, CFMAX, n_neurons):
    # convert CF bounds (in Hz) -> positions
    x_min = greenwood_inverse(CFMIN)
    x_max = greenwood_inverse(CFMAX)

    # linearly spaced positions along the cochlea
    x = np.linspace(x_min, x_max, n_neurons)

    # forward Greenwood: positions -> frequencies
    cf = greenwood_human(x)
    return cf * Hz  # keep Brian2 unit

def create_xax_time_sound(res):
    x_times = np.linspace(0, res['simulation_time'], int((res['basesound'].sound.samplerate / b2.kHz)*res['simulation_time']))
    return x_times

def flatten(items):
    """Yield items from any nested iterable.
    from https://stackoverflow.com/a/40857703
    """
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x

def take_closest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return (myList[0], 0)
    if pos == len(myList):
        return (myList[-1], len(myList))
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return (after, pos)
    else:
        return (before, pos - 1)

def avg_fire_rate_actv_neurons(x):
    active_neurons = set(x["senders"])
    return (len(x["times"]) / len(active_neurons)) if len(active_neurons) > 0 else 0

def get_spike_phases(spike_times: np.ndarray, frequency: float) -> np.ndarray:
    times_sec = spike_times
    return 2 * np.pi * frequency * (times_sec % (1 / frequency))

def calculate_vector_strength(spike_times: np.ndarray, frequency: float) -> float:
    if len(spike_times) == 0:
        return 0
    phases = get_spike_phases(spike_times, frequency)
    x = np.mean(np.cos(phases))
    y = np.mean(np.sin(phases))
    return np.sqrt(x**2 + y**2)
    
def range_around_center(center, radius, min_val=0, max_val=np.iinfo(np.int64).max):
    start = max(min_val, center - radius)
    end = min(max_val + 1, center + radius + 1)
    return np.arange(start, end)

def calculate_vector_strength_from_result(
        res,
        cue,
        pop,
        side='L',
        freq=None,            # if None: freq = res['basesound'].frequency
        color=None,
        cf_target=None,
        bandwidth=0,
        n_bins=7,
        figsize=(7,5),
        display=True,
        x_ax="phase",         # "phase" or "time"
        ylim=None,
        center_at_peak=False,
        y_ax="percent"        # "percent" (original) or "ashida"
        ):

    from collections import defaultdict
    import numpy as np
    import matplotlib.pyplot as plt

    spikes = res["cue_to_rate"][cue][side][pop]

    sender2times = defaultdict(list)
    for sender, time in zip(spikes["senders"], spikes["times"]):
        if time <= 1000:
            sender2times[sender].append(time)

    sender2times = {k: np.array(v) / 1000 for k, v in sender2times.items()}

    num_neurons = len(spikes["global_ids"])
    cf = greenwood_cf_array(CFMIN / b2.Hz, CFMAX / b2.Hz, num_neurons) / Hz

    if freq is None:
        if type(res["sounds"]["base_sound"]) in (Tone, ToneBurst):
            freq = res["sounds"]["base_sound"].frequency / Hz
        else:
            raise ValueError("Frequency must be specified for non-Tone sounds")

    if cf_target is None:
        _, center_neuron_for_freq = take_closest(cf, freq)
    else:
        _, center_neuron_for_freq = take_closest(cf, cf_target)

    old2newid = {oldid: i for i, oldid in enumerate(spikes["global_ids"])}
    new2oldid = {v: k for k, v in old2newid.items()}

    relevant_neurons = range_around_center(
        center_neuron_for_freq,
        radius=bandwidth,
        max_val=num_neurons - 1
    )

    relevant_neurons_ids = [new2oldid[i] for i in relevant_neurons]
    spike_times_list = [sender2times[i] for i in relevant_neurons_ids]

    if len(spike_times_list) == 0:
        return 0 if not display else (0, None)

    spike_times_array = np.concatenate(spike_times_list)
    total_spikes = len(spike_times_array)

    phases = get_spike_phases(spike_times_array, freq)
    vs = calculate_vector_strength(spike_times_array, freq)

    if not display:
        return vs

    if color is None:
        color = {'L': 'm', 'R': 'g'}.get(side, 'b')

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # =========================
    # PHASE AXIS
    # =========================
    if x_ax == "phase":

        orig_bins = np.linspace(0, 2 * np.pi, n_bins + 1)
        hist_raw, _ = np.histogram(phases, bins=orig_bins)
        peak_bin_idx = np.argmax(hist_raw)

        if center_at_peak:
            bin_centers_orig = (orig_bins[:-1] + orig_bins[1:]) / 2
            peak_center = bin_centers_orig[peak_bin_idx]
            shifted_phases = np.cue(np.exp(1j * (phases - peak_center)))

            bins = np.linspace(-np.pi, np.pi, n_bins + 1)
            values = shifted_phases
            bin_centers = (bins[:-1] + bins[1:]) / 2
        else:
            bins = orig_bins
            values = phases
            bin_centers = (bins[:-1] + bins[1:]) / 2

        hist1, _ = np.histogram(values, bins=bins)
        bin_width = bins[1] - bins[0]

        if y_ax == "percent":
            hist_vals = (hist1 / total_spikes) * 100
            ylabel = "Spikes / bin (% of total)"
        elif y_ax == "ashida":
            hist_vals = hist1 / (total_spikes * bin_width)
            ylabel = "Probability density (rad$^{-1}$)"
        else:
            raise ValueError("y_ax must be 'percent' or 'ashida'")

        ax.bar(bin_centers, hist_vals, width=bin_width, alpha=0.7, color=color)

        ax.set_xlabel("Phase (cycles)")
        ax.set_xticks(np.array([0, 0.5, 1, 1.5, 2]) * np.pi if not center_at_peak
                      else np.array([-1, -0.5, 0, 0.5, 1]) * np.pi)
        ax.set_xticklabels(['0', '', '0.5', '', '1'] if not center_at_peak
                           else ['-0.5', '', '0', '', '0.5'])

    # =========================
    # TIME AXIS
    # =========================
    elif x_ax == "time":

        period_ms = 1000 / freq
        time_values = (phases / (2 * np.pi)) * period_ms

        orig_bins = np.linspace(0, period_ms, n_bins + 1)
        hist_raw, _ = np.histogram(time_values, bins=orig_bins)
        peak_bin_idx = np.argmax(hist_raw)

        if center_at_peak:
            bin_centers_orig = (orig_bins[:-1] + orig_bins[1:]) / 2
            peak_center = bin_centers_orig[peak_bin_idx]
            shifted_times = np.mod(time_values - peak_center + period_ms/2,
                                   period_ms) - period_ms/2

            bins = np.linspace(-period_ms/2, period_ms/2, n_bins + 1)
            values = shifted_times
            bin_centers = (bins[:-1] + bins[1:]) / 2
        else:
            bins = orig_bins
            values = time_values
            bin_centers = (bins[:-1] + bins[1:]) / 2

        hist1, _ = np.histogram(values, bins=bins)
        bin_width = bins[1] - bins[0]

        if y_ax == "percent":
            hist_vals = (hist1 / total_spikes) * 100
            ylabel = "Spikes / bin (% of total)"
        elif y_ax == "ashida":
            hist_vals = hist1 / (total_spikes * bin_width)
            ylabel = "Probability density (ms$^{-1}$)"
        else:
            raise ValueError("y_ax must be 'percent' or 'ashida'")

        ax.bar(bin_centers, hist_vals, width=bin_width, alpha=0.7, color=color)
        ax.set_xlabel("Time [ms]")

    ax.set_ylabel(ylabel)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_title(f"R = {vs:.3f}")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.show()
    return vs, fig

def calculate_vector_strength_from_result_polar(
        res,
        cue,
        side,
        pop,
        freq=None,  # if None: freq = res['basesound'].frequency
        cf_target=None,
        bandwidth=0,
        n_bins=7,
        display=False,
        color = None,
        figsize = [7,5]  # if True also return fig, show() in caller function
        ):
    
    # Get spikes and organize times per sender
    spikes = res["cue_to_rate"][cue][side][pop] 
    print(spikes)
    sender2times = defaultdict(list)
    for sender, time in zip(spikes["senders"], spikes["times"]):
        sender2times[sender].append(time)
    sender2times = {k: np.array(v) / 1000 for k, v in sender2times.items()}
    num_neurons = len(spikes["global_ids"])
    cf = greenwood_cf_array(CFMIN/ b2.Hz, CFMAX/ b2.Hz, num_neurons)*b2.Hz
    
    # Determine the frequency to use
    if freq is None:
        if type(res['basesound']) in (Tone, ToneBurst):
            freq = res['basesound'].frequency
        else:
            print("Frequency needs to be specified for non-Tone sounds")
    else:
        freq = freq * Hz
    
    # Determine the closest characteristic frequency (CF) neuron
    if cf_target is None:    
        cf_neuron, center_neuron_for_freq = take_closest(cf, freq)
    else:
        cf_neuron, center_neuron_for_freq = take_closest(cf, cf_target * Hz)
    
    # Map between old and new neuron IDs
    old2newid = {oldid: i for i, oldid in enumerate(spikes["global_ids"])}
    new2oldid = {v: k for k, v in old2newid.items()}
    
    # Choose relevant neurons based on the center neuron and bandwidth
    relevant_neurons = range_around_center(
        center_neuron_for_freq, radius=bandwidth, max_val=num_neurons - 1
    )
    relevant_neurons_ids = [new2oldid[i] for i in relevant_neurons]
    
    # Concatenate the spike times from the relevant neurons
    spike_times_list = [sender2times[i] for i in relevant_neurons_ids]
    spike_times_array = np.concatenate(spike_times_list)  # Flatten into a single array
    
    # Compute phases and vector strength
    phases = get_spike_phases(spike_times=spike_times_array, frequency=freq / Hz)
    vs = calculate_vector_strength(spike_times=spike_times_array, frequency=freq / Hz)
    
    if not display:
        return (vs, None)

    if color == None:
        if side == 'L': color = 'm'
        elif side == 'R': color = 'g'
        else: color = 'k'

    
    # Plot phases in polar coordinates
    bins = np.linspace(0, 2 * np.pi, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Create a polar subplot
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': 'polar'}, figsize=figsize)
    hist1, _ = np.histogram(phases, bins=bins, density = True)
    ax.bar(bin_centers, hist1, width=2 * np.pi / n_bins, alpha=0.7, color = color)
    
    if bandwidth == 0:
        ax.set_title(f"Neuron {relevant_neurons_ids[0]} (CF: {cf_neuron:.1f} Hz)\nVS={vs:.3f}")
    else:
        ax.set_title(f"Neurons {relevant_neurons_ids[0]} : {relevant_neurons_ids[-1]} (center CF: {cf_neuron:.1f} Hz)\nVS={vs:.3f}")
    
    # Remove all but the last yticks
    ax.set_yticks([])  # Keep only the last tick
    #ax.yaxis.set_tick_params(labelsize=10)  # Adjust size if needed

    plt.show()
    return

def calculate_firing_rates(cue_to_rate, pop, sides, cues, duration,
                           cf_interval=None):

    num_neurons = len(cue_to_rate[0][sides[0]][pop]["global_ids"])
    cf = greenwood_cf_array(
        CFMIN / b2.Hz,
        CFMAX / b2.Hz,
        num_neurons
    ) / b2.Hz

    # -------------------------------------------------------------
    # No CF filter
    # -------------------------------------------------------------
    if cf_interval is None:

        pop_rate = {
            side: [
                len(cue_to_rate[cue][side][pop]["times"]) / duration
                for cue in cues
            ]
            for side in sides
        }

        avg_neuron_rate = {
            side: [
                len(cue_to_rate[cue][side][pop]["times"])
                / (duration * num_neurons)
                for cue in cues
            ]
            for side in sides
        }

        raw_counts = {
            side: [
                len(cue_to_rate[cue][side][pop]["times"])
                for cue in cues
            ]
            for side in sides
        }

        return pop_rate, avg_neuron_rate, raw_counts

    # -------------------------------------------------------------
    # CF-filtered case
    # -------------------------------------------------------------
    pop_rate = {}
    avg_neuron_rate = {}
    raw_counts = {}

    for side in sides:

        pop_rate[side] = []
        avg_neuron_rate[side] = []
        raw_counts[side] = []

        for cue in cues:

            _, ymin_idx = take_closest(cf, cf_interval[0])
            _, ymax_idx = take_closest(cf, cf_interval[1])

            base_id = cue_to_rate[cue][side][pop]["global_ids"][0]

            ymin = base_id + ymin_idx
            ymax = base_id + ymax_idx

            cluster_mask = (
                (cue_to_rate[cue][side][pop]["senders"] >= ymin)
                &
                (cue_to_rate[cue][side][pop]["senders"] <= ymax)
            )

            cluster_times = (
                cue_to_rate[cue][side][pop]["times"][cluster_mask]
            )

            n_spikes = len(cluster_times)
            n_neurons = ymax - ymin + 1

            pop_rate[side].append(
                n_spikes / duration
            )

            avg_neuron_rate[side].append(
                n_spikes / (n_neurons * duration)
            )

            raw_counts[side].append(
                n_spikes
            )

    return pop_rate, avg_neuron_rate, raw_counts

def normalize_rates(plotted_rate, sides):
    """
    Normalize firing rates using min-max normalization.
    
    Parameters:
    - plotted_rate: Dictionary of firing rates by side
    - sides: List of sides to process
    
    Returns:
    - normalized_rate: Dictionary of normalized firing rates by side
    - original_values: Dictionary containing original min/max values and their indices
    """
    normalized_rate = {side: [] for side in sides}
    original_values = {}
    
    for side in sides:
        # Find the minimum and maximum values across all cues for this side
        min_value = min(plotted_rate[side])
        max_value = max(plotted_rate[side])
        
        # Store original min/max values before normalization
        original_values[side] = {
            'min_value': min_value,
            'max_value': max_value,
            'min_cue_idx': plotted_rate[side].index(min_value),
            'max_cue_idx': plotted_rate[side].index(max_value)
        }
        
        # Avoid division by zero - check if max and min are different
        if max_value > min_value:  
            # Apply min-max normalization: (x - min) / (max - min)
            normalized_rate[side] = [(val - min_value) / (max_value - min_value) for val in plotted_rate[side]]
        else:
            # If all values are the same, set normalized values to 0.5
            normalized_rate[side] = [0.5 for _ in plotted_rate[side]]
    
    return normalized_rate, original_values

# ─────────────────────────────────────────────────────────────────────────────
# LEFT vs RIGHT COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def plot_tonotopic_heatmaps(
    data,
    pop='LSO',
    num_cells_per_interval=1000,
    time_interval=None,
    row_norm=True,
    title=None,
    figsize=(8, 6),
    cmap='viridis',
    diff_cmap='coolwarm',
    norm_max_given=None,
    y_axis='cf',
    f_ticks=None,
    show_sides=True,
    cue_type="angle",
):
    """
    Generates heatmaps for auditory neural responses across cues and frequency bands.

    Parameters
    ----------
    data : dict
        Simulation data dict containing 'cue_to_rate' and either 'simulation_time'
        or data['sounds']['base_sound'].sound.duration.
    pop : str, default='LSO'
        Population name to analyse (e.g. 'LSO', 'MSO').
    num_cells_per_interval : int, default=50
        Number of cells per tonotopic frequency bin.
    row_norm : bool, default=True
        If True, normalise each frequency-row by its own maximum firing rate.
    title : str, optional
        Overall suptitle for the figure.
    figsize : tuple, default=(8, 6)
        Figure size (width, height) in inches.
    cmap : str, default='viridis'
        Colormap for the left- and right-ear heatmaps.
    diff_cmap : str, default='coolwarm'
        Diverging colormap for the L−R difference heatmap.
    norm_max_given : float, optional
        If provided, forces the symmetric colour scale of the difference map to
        ±norm_max_given (only when it is larger than the data range).
    y_axis : str, default='cf'
        'cf'    → y-ticks show characteristic frequency in Hz / kHz
        'cells' → y-ticks show cell-index ranges
    f_ticks : list of float, optional
        Explicit list of CFs (Hz) to mark on the y-axis. When None all intervals
        are labelled (can be dense — use f_ticks to thin them out).
    show_sides : bool, default=True
        When True a 3-panel figure is produced (L, R, L−R).
        When False only the difference panel is shown.
    cue_type : str, default='angle'
        Controls the x-axis label format:
        'angle' → degrees  (e.g. −90°)
        'itd'   → µs       (cue values treated as seconds, converted to µs)
        'ild'   → dB

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # ------------------------------------------------------------------
    # 1.  Extract top-level objects — new data layout
    # ------------------------------------------------------------------
# ------------------------------------------------------------------
    # 1.  Extract top-level objects
    # ------------------------------------------------------------------
    cue_to_rate = data["cue_to_rate"]

    default_duration = (
        data["basesound"].sound.duration / b2.ms
        if "basesound" in data
        else data["sounds"]["base_sound"].sound.duration / b2.ms
    )
    duration_ms = data.get("simulation_time", default_duration)

    # Time-interval filtering — mirrors draw_rate_vs_cue exactly
    def _filter_spike_dict(spike_dict, time_interval):
        times      = spike_dict["times"]
        senders    = spike_dict["senders"]
        gids       = spike_dict["global_ids"]
        if time_interval is None:
            return spike_dict
        mask = (times >= time_interval[0]) & (times <= time_interval[1])
        return {"times": times[mask], "senders": senders[mask], "global_ids": gids}

    if time_interval is not None:
        cue_to_rate_filtered = {}
        for cue in cue_to_rate:
            cue_to_rate_filtered[cue] = {}
            for side in ["L", "R"]:
                cue_to_rate_filtered[cue][side] = {}
                for p in cue_to_rate[cue][side]:
                    cue_to_rate_filtered[cue][side][p] = \
                        _filter_spike_dict(cue_to_rate[cue][side][p], time_interval)
        cue_to_rate = cue_to_rate_filtered
        effective_duration = (time_interval[1] - time_interval[0]) * b2.ms
    else:
        effective_duration = duration_ms * b2.ms

    duration = effective_duration   # this is what gets passed to calculate_firing_rates

    cues  = sorted(cue_to_rate.keys())
    sides = ["L", "R"]

    # ------------------------------------------------------------------
    # 2.  Tonotopic grid
    # ------------------------------------------------------------------
    # Number of neurons — read from the first cue/side entry
    num_neurons = len(cue_to_rate[cues[0]]['L'][pop]["global_ids"])

    # Full CF array (Hz, ascending)
    cf_array = greenwood_cf_array(CFMIN / b2.Hz, CFMAX / b2.Hz, num_neurons)   # shape: (num_neurons,)

    num_intervals = math.ceil(num_neurons / num_cells_per_interval)

    # Central CF per interval (for y-axis labels)
    cf_ids = np.zeros(num_intervals)
    for i in range(num_intervals):
        start_idx  = i * num_cells_per_interval
        end_idx    = min((i + 1) * num_cells_per_interval, num_neurons)
        mid_idx    = (start_idx + end_idx - 1) // 2
        cf_ids[i]  = cf_array[mid_idx] / b2.Hz   # store as plain float (Hz)

    # ------------------------------------------------------------------
    # 3.  Compute firing rates using calculate_firing_rates
    #     We iterate over intervals by passing a cf_interval to the
    #     shared helper — consistent with draw_rate_vs_cue.
    # ------------------------------------------------------------------
    rate_matrices = {side: np.zeros((num_intervals, len(cues))) for side in sides}

    for i in range(num_intervals):
        start_idx = i * num_cells_per_interval
        end_idx   = min((i + 1) * num_cells_per_interval, num_neurons)
        mid_idx   = (start_idx + end_idx - 1) // 2

        # CF boundaries for this interval (Hz, plain floats)
        cf_lo = cf_array[start_idx] / b2.Hz
        cf_hi = cf_array[end_idx - 1] / b2.Hz

        # Tiny guard: avoid a degenerate single-point interval
        if cf_lo == cf_hi:
            half_bin = (cf_array[1] / b2.Hz - cf_array[0] / b2.Hz) * 0.5
            cf_interval_i = [cf_lo - half_bin, cf_hi + half_bin]
        else:
            cf_interval_i = [cf_lo, cf_hi]

        # calculate_firing_rates returns (tot_spikes_dict, avg_rate_dict, n_neurons_dict)
        _, avg_rate, _ = calculate_firing_rates(
            cue_to_rate,
            pop,
            sides,
            cues,
            duration,
            cf_interval_i,
        )

        for side in sides:
            rate_matrices[side][i, :] = avg_rate[side]   # shape: (len(cues),)

    # ------------------------------------------------------------------
    # 4.  Optional row normalisation
    # ------------------------------------------------------------------
    if row_norm:
        for side in sides:
            for i in range(num_intervals):
                row_max = np.max(rate_matrices[side][i, :])
                if row_max > 0:
                    rate_matrices[side][i, :] /= row_max

    # ------------------------------------------------------------------
    # 5.  Difference matrix (L − R)
    # ------------------------------------------------------------------
    diff_matrix = rate_matrices['L'] - rate_matrices['R']

    # ------------------------------------------------------------------
    # 6.  Build figure
    # ------------------------------------------------------------------
    if show_sides:
        fig, axes = plt.subplots(1, 3, figsize=(figsize[0]*3, figsize[1]))

        im_left  = axes[0].imshow(rate_matrices['L'], cmap=cmap,
                                  aspect='auto', interpolation='none')
        im_right = axes[1].imshow(rate_matrices['R'], cmap=cmap,
                                  aspect='auto', interpolation='none')

        cbar_left  = plt.colorbar(im_left,  ax=axes[0])
        cbar_right = plt.colorbar(im_right, ax=axes[1])
        cbar_label = 'Normalized Firing Rate' if row_norm else 'Firing Rate [Hz]'
        cbar_left.set_label(cbar_label)
        cbar_right.set_label(cbar_label)

        axes[0].set_title('Left side')
        axes[1].set_title('Right side')
        diff_ax = axes[2]
        diff_ax.set_title('Difference (L − R)')
    else:
        fig, diff_ax = plt.subplots(1, 1, figsize=figsize)

    # Diverging colour scale, symmetric around zero
    norm_max = max(abs(np.nanmin(diff_matrix)), abs(np.nanmax(diff_matrix)))
    if norm_max_given is not None and norm_max_given >= norm_max:
        norm_max = norm_max_given
    diff_norm = Normalize(vmin=-norm_max, vmax=norm_max)

    im_diff  = diff_ax.imshow(diff_matrix, cmap=diff_cmap, aspect='auto',
                              interpolation='none', norm=diff_norm)
    cbar_diff = plt.colorbar(im_diff, ax=diff_ax)
    cbar_diff.set_label('Difference (L − R)')

    # ------------------------------------------------------------------
    # 7.  Axis formatting helper
    # ------------------------------------------------------------------
    def _format_cf(hz):
        if hz < 1000:
            return f"{int(round(hz))} Hz"
        else:
            return f"{hz / 1000:.1f} kHz"

    def setup_axis(ax):
        # --- x-axis ---
        ax.set_xticks(np.arange(len(cues)))
        if cue_type == "angle":
            ax.set_xticklabels([f"{int(c)}°" for c in cues])
            ax.set_xlabel("Azimuth angle [deg]")
        elif cue_type == "itd":
            ax.set_xticklabels([f"{round(c * 1e6)}" for c in cues],
                               rotation=45, ha='right')
            ax.set_xlabel("ITD [µs]")
        elif cue_type == "ild":
            ax.set_xticklabels([f"{c}" for c in cues])
            ax.set_xlabel("ILD [dB]")
        elif cue_type == "ild_exp":
            ax.set_xticklabels([f"{round(c)}" for c in cues])
            ax.set_xlabel("Contralateral Level [dB]")

        # --- y-axis ---
        if f_ticks is not None and y_axis == 'cf':
            y_positions, y_labels = [], []

            # Fixed edge ticks
            y_positions.append(-0.5);                  y_labels.append(_format_cf(cf_ids[0]))
            y_positions.append(num_intervals - 0.5);   y_labels.append(_format_cf(cf_ids[-1]))

            for freq in f_ticks:
                distances = np.abs(cf_ids - freq)
                closest_idx = int(np.argmin(distances))
                y_positions.append(closest_idx)
                y_labels.append(_format_cf(freq))

            ax.set_yticks(y_positions)
            ax.set_yticklabels(y_labels)

        else:
            ax.set_yticks(np.arange(num_intervals))
            if y_axis == 'cf':
                ax.set_yticklabels([_format_cf(f) for f in cf_ids])
            else:  # 'cells'
                labels = []
                for i in range(num_intervals):
                    s = i * num_cells_per_interval
                    e = min((i + 1) * num_cells_per_interval, num_neurons) - 1
                    labels.append(f"{s}–{e}")
                ax.set_yticklabels(labels)

        ax.invert_yaxis()   # low-CF (high-index in Greenwood ascending order) at bottom

    # Apply to all axes
    if show_sides:
        for ax in axes:
            setup_axis(ax)
        axes[0].set_ylabel(
            'Characteristic frequency' if y_axis == 'cf' else 'Cell indices'
        )
    else:
        setup_axis(diff_ax)
        diff_ax.set_ylabel(
            'Characteristic frequency' if y_axis == 'cf' else 'Cell indices'
        )

    plt.tight_layout()
    if title:
        plt.subplots_adjust(top=0.88)
        fig.suptitle(title, fontsize=16)

    return

def plot_rasterplot(
    spikes_series,
    y_ax='cf_custom',
    f_ticks=[125, 1000, 10000],
    cf_bin_size=50, #cells
    psth_bin_size=1, #ms
    hist_rate=True,
    figsize=(15, 8),
    color="b",
    xlim=None,
    ylim=None,
):
    """
    Raster + population histogram + PSTH from a Pandas Series of spike times.

    spikes_series:
        index = neuron id
        values = list of spike times (seconds)
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    gids = np.array(spikes_series.index)
    n_neurons = len(gids)

    # -------------------------------------------------------
    # flatten Series → times + senders
    # -------------------------------------------------------
    times = []
    senders = []

    for gid, ts in spikes_series.items():
        if len(ts) > 0:
            t = np.array(ts) * 1000.0
            times.append(t)
            senders.append(np.full(len(t), gid))

    if len(times):
        times = np.concatenate(times)
        senders = np.concatenate(senders)
    else:
        times = np.array([])
        senders = np.array([])

    # -------------------------------------------------------
    # xlim
    # -------------------------------------------------------
    if xlim is None:
        xmax = times.max() if len(times) else 1
        xlim = [0, xmax]

    # -------------------------------------------------------
    # CF mapping (same as your main function)
    # -------------------------------------------------------
    cf_full = greenwood_cf_array(CFMIN/b2.Hz, CFMAX/b2.Hz, n_neurons) / b2.Hz

    if ylim is None:
        ylim = [cf_full.min(), cf_full.max()]

    _, ymin_idx = take_closest(cf_full, ylim[0])
    _, ymax_idx = take_closest(cf_full, ylim[1])

    # -------------------------------------------------------
    # filtering
    # -------------------------------------------------------
    mask_t = (times >= xlim[0]) & (times <= xlim[1])
    times_f = times[mask_t]
    senders_f = senders[mask_t]
    local_ids_f = senders_f - gids[0]

    # -------------------------------------------------------
    # layout: raster + hist | psth
    # -------------------------------------------------------
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(
        2, 2,
        width_ratios=[5, 1.5],
        height_ratios=[3, 1],
        wspace=0.05,
        hspace=0.25
    )

    ax_raster = fig.add_subplot(gs[0, 0])
    ax_hist   = fig.add_subplot(gs[0, 1], sharey=ax_raster)
    ax_psth   = fig.add_subplot(gs[1, 0], sharex=ax_raster)

    # -------------------------------------------------------
    # Y axis modes
    # -------------------------------------------------------
    if y_ax in ["neurons", "cf_custom"]:
        y_values = senders_f
        ax_raster.set_ylim([gids[0] + ymin_idx, gids[0] + ymax_idx])
        ax_raster.set_ylabel("Neuron ID")

        if y_ax == "cf_custom":
            ax_raster.set_ylabel("CF [Hz]")
            tick_pos = []
            for f in f_ticks:
                _, idx = take_closest(cf_full, f)
                tick_pos.append(gids[0] + idx)
            ax_raster.set_yticks(tick_pos)
            ax_raster.set_yticklabels(f_ticks)

    elif y_ax == "cf":
        y_values = cf_full[local_ids_f]
        ax_raster.set_ylabel("CF [Hz]")
        ax_raster.set_ylim(ylim)

    else:
        raise ValueError("unknown y_ax mode")

    # -------------------------------------------------------
    # RASTER
    # -------------------------------------------------------
    ax_raster.plot(times_f, y_values, '.', color=color, markersize=1)
    ax_raster.set_xlim(xlim)


    # -------------------------------------------------------
    # POPULATION HISTOGRAM (same math as your first function)
    # -------------------------------------------------------
    spike_count = np.bincount(local_ids_f, minlength=n_neurons)

    bins_neurons = np.arange(0, n_neurons, cf_bin_size)

    grouped_counts = np.array([
        spike_count[i:i+cf_bin_size].sum()
        for i in bins_neurons
    ])

    grouped_y = np.array([
        np.arange(n_neurons)[i:i+cf_bin_size].mean()
        for i in bins_neurons
    ])

    mask_vis = (grouped_y >= ymin_idx) & (grouped_y <= ymax_idx)

    grouped_y_plot = gids[0] + grouped_y[mask_vis]
    grouped_counts = grouped_counts[mask_vis]

    if hist_rate:
        grouped_values = (grouped_counts / xlim[1]) * 1000.0 / cf_bin_size
        xlabel = "Avg rate [Hz]"
    else:
        grouped_values = grouped_counts
        xlabel = "Spike count"

    avg_value = grouped_values.mean() if len(grouped_values) else 0

    ax_hist.barh(grouped_y_plot, grouped_values,
                 height=0.8 * cf_bin_size,
                 color=color, alpha=0.4)

    ax_hist.axvline(avg_value, linestyle='--', linewidth=2, color=color)
    ax_hist.set_xlabel(xlabel)
    ax_hist.tick_params(axis='y', labelleft=False)

    # -------------------------------------------------------
    # PSTH  ✅ (this was missing before)
    # -------------------------------------------------------
    bins = np.arange(xlim[0], xlim[1] + psth_bin_size, psth_bin_size)
    counts, _ = np.histogram(times_f, bins=bins)

    if hist_rate:
        n_visible = ymax_idx - ymin_idx + 1
        rates = (counts * 1000.0) / (psth_bin_size * n_visible)
        avg_rate = rates.mean() if len(rates) else 0
        ax_psth.plot(bins[:-1], rates, color=color, alpha=0.8)
        ax_psth.axhline(avg_rate, linestyle='--', linewidth=2, color=color)
        ax_psth.set_ylabel("Rate [Hz]")
    else:
        ax_psth.bar(bins[:-1], counts, width=psth_bin_size, alpha=0.4, color=color)
        ax_psth.set_ylabel("Spike count")

    ax_psth.set_xlabel("Time (ms)")

    print(f"Avg firing rate: {avg_value:.2f} Hz")

    return fig, (ax_raster, ax_hist, ax_psth)

def plot_sound(
    sound,
    figsize=(6, 4),
    title=None,
    time_in_ms=True,
    xlim=None,
    ylim=None,
    colors=None,
    labels=None):
    """
    Plot one or multiple Brian2Hears Sound objects over time.
    Works for mono or stereo.
    sound: brian2hears.Sound, Tone/ToneBurst/etc., or a list/tuple of any of these.
    colors: single color string, or list of colors (one per sound)
    labels: single label string, or list of labels (one per sound)
    """

    def _to_wave_and_fs(s):
        snd = s if isinstance(s, b2h.Sound) else s.sound
        return np.asarray(snd), float(snd.samplerate)

    def _time_axis(wave, fs):
        t = np.arange(wave.shape[0]) / fs
        return t * 1000 if time_in_ms else t

    xlabel = "Time (ms)" if time_in_ms else "Time (s)"

    # ── Normalise input to a list ──────────────────────────────────────────
    if isinstance(sound, (list, tuple)):
        sounds = list(sound)
    else:
        sounds = [sound]

    n = len(sounds)

    # ── Normalise colors and labels to lists of length n ──────────────────
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    if colors is None:
        colors = default_colors[:n]
    elif isinstance(colors, str):
        colors = [colors] * n
    else:
        colors = list(colors)

    if labels is None:
        if n == 1:
            labels = [None]
        else:
            labels = [
                create_sound_key(s) if not isinstance(s, b2h.Sound) else f"Sound {i+1}"
                for i, s in enumerate(sounds)
            ]
    elif isinstance(labels, str):
        labels = [labels]
    else:
        labels = list(labels)

    # ── Plot ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)
    needs_legend = n > 1

    for i, s in enumerate(sounds):
        wave, fs = _to_wave_and_fs(s)
        t = _time_axis(wave, fs)
        color = colors[i % len(colors)]
        lbl = labels[i] if i < len(labels) else f"Sound {i+1}"

        is_mono = wave.ndim == 1 or (wave.ndim == 2 and wave.shape[1] == 1)
        w = wave[:, 0] if (wave.ndim == 2) else wave

        if is_mono:
            ax.plot(t, w, color=color, linewidth=0.8, label=lbl)
        else:
            # stereo: left/right as separate traces
            suffix = f" ({lbl})" if n > 1 else ""
            ax.plot(t, wave[:, 0], linewidth=0.8,
                    color=color, linestyle='-',  label=f"L{suffix}")
            ax.plot(t, wave[:, 1], linewidth=0.8,
                    color=color, linestyle='--', label=f"R{suffix}")
            needs_legend = True

    if needs_legend:
        ax.legend()

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Amplitude")
    ax.set_ylim(ylim)
    if xlim:
        ax.set_xlim(xlim)

    # ── Title ──────────────────────────────────────────────────────────────
    if title is None and n == 1 and not isinstance(sounds[0], b2h.Sound):
        title = create_sound_key(sounds[0])
    ax.set_title(title)

    plt.tight_layout()
    return fig, ax

def draw_spikes_and_psth_bothside(
    res,
    cue,
    pop,
    y_ax='cf_custom',
    f_ticks=[125, 1000, 10000],
    title=None,
    xlim=None,
    center_cf=None,
    bw_neurons=None,
    cf_interval=None,
    psth_bin_size=1,
    hist_rate=False,
    cf_bin_size=3,
    raster_dot_size=1,
    figsize=(14, 18)
):

    side_colors = {'L': 'm', 'R': 'g'}

    duration = res.get("simulation_time", res["sounds"]["base_sound"].sound.duration / b2.ms)
    if xlim is None:
        xlim = [0, duration]

    # ------------------------------------------------------------------
    # Resolve ylim — mirrors draw_rate_vs_cue logic exactly:
    #   1. center_cf + bw_neurons  → derive cf_interval from neuron indices
    #   2. cf_interval             → use directly as [ylim_min, ylim_max]
    #   3. neither                 → full frequency range
    # ------------------------------------------------------------------
    if center_cf is not None and bw_neurons is not None:
        _n_tmp = len(res["cue_to_rate"][cue]["L"][pop]["global_ids"])
        _cf_tmp = greenwood_cf_array(CFMIN / b2.Hz, CFMAX / b2.Hz, _n_tmp) / b2.Hz
        _, center_idx = take_closest(_cf_tmp, center_cf)
        low_idx  = max(0, center_idx - bw_neurons)
        high_idx = min(_n_tmp - 1, center_idx + bw_neurons)
        ylim = [_cf_tmp[low_idx], _cf_tmp[high_idx]]
        print(
            f"[draw_spikes_and_psth_bothside] center_cf={center_cf} Hz → "
            f"neuron idx [{low_idx}, {high_idx}] → "
            f"ylim=[{ylim[0]:.1f}, {ylim[1]:.1f}] Hz"
        )
    elif cf_interval is not None:
        ylim = list(cf_interval)          # [cf_min, cf_max] passed directly
    else:
        ylim = [CFMIN / Hz, CFMAX / Hz]  

    L_hrtf_sound = res["sounds"]["left_sounds"][cue]
    R_hrtf_sound = res["sounds"]["right_sounds"][cue]

    # -----------------------------------------------------------------------
    # 5-row LAYOUT
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(
        5, 2, figure=fig,
        width_ratios=[5, 1.5],
        height_ratios=[0.15, 1, 0.15, 1, 0.8],
        hspace=0.35,
        wspace=0.05,
    )

    ax_sound0 = fig.add_subplot(gs[0, 0])
    t0 = np.arange(len(R_hrtf_sound)) / R_hrtf_sound.samplerate * 1000
    ax_sound0.plot(t0, R_hrtf_sound, color='g', lw=2)
    ax_sound0.set_ylabel("R sound")
    ax_sound0.set_xlim(xlim)
    ax_sound0.grid(True, alpha=0.3)

    ax_raster_R = fig.add_subplot(gs[1, 0])
    ax_hist_R   = fig.add_subplot(gs[1, 1], sharey=ax_raster_R)

    ax_sound2 = fig.add_subplot(gs[2, 0])
    t2 = np.arange(len(L_hrtf_sound)) / L_hrtf_sound.samplerate * 1000
    ax_sound2.plot(t2, L_hrtf_sound, color='m', lw=2)
    ax_sound2.set_ylabel("L sound")
    ax_sound2.set_xlim(xlim)
    ax_sound2.grid(True, alpha=0.3)

    ax_raster_L = fig.add_subplot(gs[3, 0])
    ax_hist_L   = fig.add_subplot(gs[3, 1], sharey=ax_raster_L)

    ax_psth     = fig.add_subplot(gs[4, 0], sharex=ax_raster_L)

    # ===========================================================
    # Helper: Filter spikes
    # ===========================================================
    def filter_spikes(spikes, xlim, ylim):
        times = spikes["times"]
        senders = spikes["senders"]
        gids = spikes["global_ids"]
        n = len(gids)

        mask_t = (times >= xlim[0]) & (times <= xlim[1])
        times_t = times[mask_t]
        senders_t = senders[mask_t]

        cf_hz = greenwood_cf_array(CFMIN/ b2.Hz, CFMAX/ b2.Hz, n) / b2.Hz
        _, ymin_idx = take_closest(cf_hz, ylim[0])
        _, ymax_idx = take_closest(cf_hz, ylim[1])

        cf_min_id = gids[0] + ymin_idx
        cf_max_id = gids[0] + ymax_idx

        mask_cf = (senders_t >= cf_min_id) & (senders_t <= cf_max_id)

        return (
            times_t[mask_cf],
            senders_t[mask_cf],
            cf_hz,
            ymin_idx,
            ymax_idx,
            gids
        )

    # ===========================================================
    # Helper: Raster y-axis
    # ===========================================================
    def setup_raster_yaxis(ax, y_ax, senders_f, local_ids_f, cf_full, gids, ylim, ymin_idx, ymax_idx, pop):
        if y_ax in ["neurons", "cf_custom"]:
            y_values = senders_f
            if ylim is None:
                ax.set_ylim([gids[0], gids[-1]])
            else:
                ax.set_ylim([gids[0] + ymin_idx, gids[0] + ymax_idx])
            ax.set_ylabel(f"{pop} neuron ID")

            if y_ax == "cf_custom":
                ax.set_ylabel(f"{pop} CF [Hz]")
                tick_pos = []
                for f in f_ticks:
                    _, idx = take_closest(cf_full, f)
                    tick_pos.append(gids[0] + idx)
                ax.set_yticks(tick_pos)
                ax.set_yticklabels(f_ticks)

        elif y_ax == "cf":
            y_values = cf_full[local_ids_f]
            ax.set_ylabel(f"{pop} CF [Hz]")
            ax.set_ylim(ylim)
        else:
            raise ValueError(f"Unknown y_ax mode: {y_ax}")

        return y_values

    # ===========================================================
    # Helper: Population histogram
    # ===========================================================
    def compute_population_histogram(y_ax, local_ids_f, cf_full, n_neurons, ymin_idx, ymax_idx, gids, ylim, cf_bin_size):
        if y_ax in ["neurons", "cf_custom"]:
            spike_count = np.bincount(local_ids_f, minlength=n_neurons)
            bins_neurons = np.arange(0, n_neurons, cf_bin_size)
            grouped_counts = np.array([spike_count[i:i+cf_bin_size].sum() for i in bins_neurons])
            grouped_y = np.array([np.arange(n_neurons)[i:i+cf_bin_size].mean() for i in bins_neurons])
            mask_vis = (grouped_y >= ymin_idx) & (grouped_y <= ymax_idx)
            grouped_y_plot = gids[0] + grouped_y[mask_vis]
            grouped_counts = grouped_counts[mask_vis]
            bar_height = 0.8 * cf_bin_size
        elif y_ax == "cf":
            spike_cf = cf_full[local_ids_f]
            cf_bins = np.arange(ylim[0], ylim[1] + cf_bin_size, cf_bin_size)
            grouped_counts, _ = np.histogram(spike_cf, bins=cf_bins)
            grouped_y_plot = 0.5 * (cf_bins[:-1] + cf_bins[1:])
            bar_height = 0.8 * (cf_bins[1] - cf_bins[0])
        else:
            raise ValueError(f"Unknown y_ax mode: {y_ax}")
        return grouped_y_plot, grouped_counts, bar_height

    # ===========================================================
    # RASTERS + HISTOGRAMS
    # ===========================================================
    for side, ax_raster, ax_hist in [
        ("L", ax_raster_L, ax_hist_L),
        ("R", ax_raster_R, ax_hist_R),
    ]:
        spikes = res["cue_to_rate"][cue][side][pop]

        times_f, senders_f, cf_full, ymin_idx, ymax_idx, gids = \
            filter_spikes(spikes, xlim, ylim)

        n_neurons = len(gids)
        local_ids_f = senders_f - gids[0]

        # RASTER Y-axis
        y_values = setup_raster_yaxis(
            ax=ax_raster,
            y_ax=y_ax,
            senders_f=senders_f,
            local_ids_f=local_ids_f,
            cf_full=cf_full,
            gids=gids,
            ylim=ylim,
            ymin_idx=ymin_idx,
            ymax_idx=ymax_idx,
            pop=pop
        )

        ax_raster.plot(times_f, y_values, '.', color=side_colors[side], markersize=raster_dot_size)
        ax_raster.set_xlim(xlim)
        ax_raster.text(0.0, 1.05, f"{side} side",
                       transform=ax_raster.transAxes,
                       fontsize=12, fontweight='bold',
                       color=side_colors[side])

        # POPULATION HISTOGRAM
        grouped_y, grouped_counts, bar_height = compute_population_histogram(
            y_ax=y_ax,
            local_ids_f=local_ids_f,
            cf_full=cf_full,
            n_neurons=n_neurons,
            ymin_idx=ymin_idx,
            ymax_idx=ymax_idx,
            gids=gids,
            ylim=ylim,
            cf_bin_size=cf_bin_size
        )

        if hist_rate:
            grouped_values = (grouped_counts / xlim[1]) * 1000.0 / cf_bin_size
            avg_value = grouped_values.mean()
            #print(f"Avg firing rate ({side} side): {avg_value:.2f} Hz POP")
            xlabel = "Avg Firing rate [Hz]"
        else:
            grouped_values = grouped_counts
            avg_value = grouped_values.mean()
            xlabel = "Spike count"

        ax_hist.barh(grouped_y, grouped_values, height=bar_height,
                     color=side_colors[side], alpha=0.4)
        ax_hist.axvline(avg_value, linestyle='--', linewidth=2,
                color=side_colors[side], alpha=0.9)
        ax_hist.set_ylim(ax_raster.get_ylim())
        ax_hist.set_xlabel(xlabel)
        ax_hist.tick_params(axis='y', labelleft=False)
        x_max = max(ax_hist_L.get_xlim()[1], ax_hist_R.get_xlim()[1])
        ax_hist_L.set_xlim(0, x_max)
        ax_hist_R.set_xlim(0, x_max)

    # ===========================================================
    # PSTH
    # ===========================================================
    for side in ["L", "R"]:
        color = side_colors[side]
        spikes = res["cue_to_rate"][cue][side][pop]

        times_f, _, _, ymin_idx, ymax_idx, _ = filter_spikes(spikes, xlim, ylim)

        bins = np.arange(xlim[0], xlim[1] + psth_bin_size, psth_bin_size)
        counts, _ = np.histogram(times_f, bins=bins)

        if hist_rate:
            rates = (counts * 1000.0) / (psth_bin_size * (ymax_idx - ymin_idx + 1))
            avg_value = rates.mean()
            print(f"Avg firing rate ({side} side): {avg_value:.2f} Hz")
            ax_psth.plot(bins[:-1], rates, color=color, alpha=0.7, label=side)
            ax_psth.axhline(avg_value, linestyle='--', linewidth=2, color=side_colors[side], alpha=0.9)
        else:
            ax_psth.hist(times_f, bins=bins, alpha=0.4, color=color, label=side)

    ax_psth.set_xlabel("Time [ms]")
    ax_psth.set_ylabel("Avg Firing rate [Hz]" if hist_rate else "Spike count")
    ax_psth.legend()

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')

def draw_rate_vs_cue(
    data,
    pop='LSO',
    rate='avg', 
    side = None,         
    cf_interval=None,
    time_interval=None,
    target_cf_hz=None,
    center_cf=None,
    bw_neurons=None,
    color=None,
    figsize=[7, 4],
    title=None,
    ylim=None,
    label=None,
    error='sem',
    shaded=True,
    cue_type="angle",
    xlim=None,
    ipsi_level = None
):

    VALID_RATES = {'avg', 'pop', 'spk', 'spk_pn', 'mm_norm', 'max_norm',}
    if rate not in VALID_RATES:
        raise ValueError(f"rate must be one of {VALID_RATES}, got {rate!r}")

    # ------------------------------------------------------------------
    # Everything below is identical to your original up to _draw_single_pop_subplot
    # ------------------------------------------------------------------
    if isinstance(data, list):
        multi_data = data
        data = data[0]
        multi_mode = True
    else:
        multi_data = [data]
        multi_mode = False

    cue_to_rate = data["cue_to_rate"]
    default_duration = (
        data["basesound"].sound.duration / b2.ms
        if "basesound" in data
        else data["sounds"]["base_sound"].sound.duration / b2.ms
    )
    duration = data.get("simulation_time", default_duration) * b2.ms

    # CF interval resolution (target_cf_hz / center_cf+bw_neurons) — unchanged
    if target_cf_hz is not None:
        _first_cue = list(cue_to_rate.keys())[0]
        _pop_key   = pop if (isinstance(pop, str) and pop != "all") else "LSO"
        _n_tmp     = len(cue_to_rate[_first_cue]["L"][_pop_key]["global_ids"])
        _cf_tmp    = greenwood_cf_array(CFMIN / Hz, CFMAX / Hz, _n_tmp) / Hz
        _, cf_idx  = take_closest(_cf_tmp, target_cf_hz)
        half_bin   = (_cf_tmp[1] - _cf_tmp[0]) * 0.5
        cf_interval = [_cf_tmp[cf_idx] - half_bin, _cf_tmp[cf_idx] + half_bin]
        print(
            f"[draw_rate_vs_cue] target_cf_hz={target_cf_hz} Hz → "
            f"neuron idx {cf_idx} → "
            f"cf_interval=[{cf_interval[0]:.1f}, {cf_interval[1]:.1f}] Hz"
        )

    elif center_cf is not None and bw_neurons is not None:
        _first_cue = list(cue_to_rate.keys())[0]
        _pop_key   = pop if (isinstance(pop, str) and pop != "all") else "LSO"
        _n_tmp     = len(cue_to_rate[_first_cue]["L"][_pop_key]["global_ids"])
        _cf_tmp    = greenwood_cf_array(CFMIN / Hz, CFMAX / Hz, _n_tmp) / Hz
        _, center_idx = take_closest(_cf_tmp, center_cf)
        low_idx  = max(0, center_idx - bw_neurons)
        high_idx = min(_n_tmp - 1, center_idx + bw_neurons)
        if low_idx == high_idx:
            half_bin    = (_cf_tmp[1] - _cf_tmp[0]) * 0.5
            cf_interval = [_cf_tmp[low_idx] - half_bin, _cf_tmp[high_idx] + half_bin]
        else:
            cf_interval = [_cf_tmp[low_idx], _cf_tmp[high_idx]]
        print(
            f"[draw_rate_vs_cue] center_cf={center_cf} Hz → "
            f"neuron idx [{low_idx}, {high_idx}] → "
            f"cf_interval=[{cf_interval[0]:.1f}, {cf_interval[1]:.1f}] Hz"
        )

    def _filter_spike_dict(spike_dict, time_interval):
        times   = spike_dict["times"]
        senders = spike_dict["senders"]
        gids    = spike_dict["global_ids"]
        if time_interval is None:
            return spike_dict
        mask = (times >= time_interval[0]) & (times <= time_interval[1])
        return {"times": times[mask], "senders": senders[mask], "global_ids": gids}

    if time_interval is not None:
        effective_duration = (time_interval[1] - time_interval[0]) * b2.ms
    else:
        effective_duration = duration

    # ------------------------------------------------------------------
    def _draw_single_pop_subplot(ax, pop_name, side):

        cues = sorted(cue_to_rate.keys())
        if side is not None:
            sides_local = [side]
        else:
            sides_local = ["L", "R"]

        if isinstance(color, dict):
            side_colors = color
        elif isinstance(color, str):
            side_colors = {sides_local[0]: color}
        else:
            side_colors = {"L": "m", "R": "g"}

        all_pop   = {side: [] for side in sides_local}
        all_avg   = {side: [] for side in sides_local}
        all_count = {side: [] for side in sides_local}

        for d in multi_data:
            angle_to_rate_d = d["cue_to_rate"]

            if time_interval is not None:
                angle_to_rate_filtered = {}
                for cue in cues:
                    angle_to_rate_filtered[cue] = {}
                    for side in sides_local:
                        angle_to_rate_filtered[cue][side] = {}
                        for p in angle_to_rate_d[cue][side]:
                            if p == pop_name:
                                angle_to_rate_filtered[cue][side][p] = \
                                    _filter_spike_dict(
                                        angle_to_rate_d[cue][side][p],
                                        time_interval
                                    )
                            else:
                                angle_to_rate_filtered[cue][side][p] = \
                                    angle_to_rate_d[cue][side][p]
                atr_to_use = angle_to_rate_filtered
                dur_d = effective_duration
            else:
                atr_to_use = angle_to_rate_d
                dur_d = (
                    d.get(
                        "simulation_time",
                        data["sounds"]["base_sound"].sound.duration / b2.ms,
                    )
                    * b2.ms
                )

            pop_d, avg_d, cnt_d = calculate_firing_rates(
                atr_to_use, pop_name, sides_local, cues, dur_d, cf_interval
            )

            for side in sides_local:
                all_pop  [side].append(pop_d[side])
                all_avg  [side].append(avg_d[side])
                all_count[side].append(cnt_d[side])

        # Mean across datasets
        mean_pop   = {s: np.mean(all_pop  [s], axis=0) for s in sides_local}
        mean_avg   = {s: np.mean(all_avg  [s], axis=0) for s in sides_local}
        mean_count = {s: np.mean(all_count[s], axis=0) for s in sides_local}

        # Error (only meaningful in multi_mode)
        if multi_mode:
            if error == "sem":
                _err = lambda x: np.std(x, axis=0) / np.sqrt(len(multi_data))
            elif error == "std":
                _err = lambda x: np.std(x, axis=0)
            else:
                raise ValueError("error must be 'sem' or 'std'")
            err_pop   = {s: _err(all_pop  [s]) for s in sides_local}
            err_avg   = {s: _err(all_avg  [s]) for s in sides_local}
            err_count = {s: _err(all_count[s]) for s in sides_local}
        else:
            err_pop = err_avg = err_count = None

        # ------------------------------------------------------------------
        # Select what to plot — clean single block
        # ------------------------------------------------------------------
        if rate == 'avg':
            plotted_rate = mean_avg
            plotted_err  = err_avg
            ylabel_text  = "Avg Firing Rate [Hz]"

        elif rate == 'pop':
            plotted_rate = mean_pop
            plotted_err  = err_pop
            ylabel_text  = "Population Firing Rate [Hz]"

        elif rate == 'spk':
            plotted_rate = mean_count
            plotted_err  = err_count
            ylabel_text  = "Spike Count"

        elif rate == 'spk_pn':
            # Resolve n_neurons from cf_interval (same logic used in the title block)
            _first_cue = list(cue_to_rate.keys())[0]
            _pop_key   = pop if isinstance(pop, str) and pop != "all" else pop_name
            _n_tmp     = len(cue_to_rate[_first_cue]["L"][pop_name]["global_ids"])
            if cf_interval is not None:
                _cf_tmp      = greenwood_cf_array(CFMIN / Hz, CFMAX / Hz, _n_tmp) / Hz
                _, _ymin_idx = take_closest(_cf_tmp, cf_interval[0])
                _, _ymax_idx = take_closest(_cf_tmp, cf_interval[1])
                n_neurons    = _ymax_idx - _ymin_idx + 1
            else:
                n_neurons = _n_tmp

            plotted_rate = {
                s: np.array(mean_count[s]) / n_neurons
                for s in sides_local
            }
            plotted_err = (
                {s: np.array(err_count[s]) / n_neurons for s in sides_local}
                if err_count is not None else None
            )
            ylabel_text = "Spikes / Neuron"

        elif rate == 'mm_norm':
            plotted_rate, _ = normalize_rates(mean_avg, sides_local)
            plotted_err      = err_avg
            ylabel_text      = "Min-Max Normalized Rate"

        elif rate == 'max_norm':
            plotted_rate = {
                s: np.array(mean_avg[s]) / np.max(mean_avg[s])
                for s in sides_local
            }
            plotted_err = err_avg
            ylabel_text = "Max Normalized Rate"

        # ------------------------------------------------------------------
        # Plot — unchanged from your original
        # ------------------------------------------------------------------

        for side in sides_local:
            mean_curve = plotted_rate[side]
            ax.plot(
                cues, mean_curve, "o-",
                color=side_colors.get(side, "k"),
                label=label if label else side,
            )
            if multi_mode and plotted_err is not None:
                err_curve = plotted_err[side]
                if shaded:
                    ax.fill_between(
                        cues,
                        mean_curve - err_curve,
                        mean_curve + err_curve,
                        alpha=0.25,
                        color=side_colors.get(side, "k"),
                        linewidth=0,
                        label=f"±{error.upper()}",
                    )
                else:
                    ax.errorbar(
                        cues, mean_curve, yerr=err_curve,
                        fmt="none", capsize=3,
                        color=side_colors.get(side, "k"),
                    )
        if label is None:
            ax.legend()

        ax.set_ylabel(ylabel_text)

        # X-axis, ylim, title — completely unchanged from your original
        # [ ... paste your existing block here unchanged ... ]

    # ---- SINGLE POP / ALL POPS — unchanged ----
    # [ ... paste your existing dispatch block here unchanged ... ]

        # ------------------------------------------------------------------
        # X-axis ticks, labels, limits
        # ------------------------------------------------------------------
        if cue_type == "angle":
            ax.set_xticks(cues)
            ax.set_xticklabels([f"{int(c)}°" for c in cues])
            ax.set_xlabel("Azimuth Angle [deg]")
            if xlim:
                ax.set_xlim(xlim)

        elif cue_type == "itd":
            if xlim:
                xlim_s = [xlim[0]/1e6, xlim[1]/1e6]
                ax.set_xlim(xlim_s)
                visible_cues = [c for c in cues if xlim_s[0] <= c <= xlim_s[1]]
            else:
                visible_cues = cues

            max_ticks = 8
            if len(visible_cues) > max_ticks:
                visible_cues = np.concatenate([np.linspace(-0.005, -0.001,  4, endpoint=False),np.linspace(-0.001,  0.001, 5),np.linspace(0.001,  0.005,  5)[1:]])

            ax.set_xticks(visible_cues)
            ax.set_xticklabels(
                [f"{round(c * 1e6)}" for c in visible_cues],
                rotation=45,
                ha='right'
            )
            ax.set_xlabel("ITD [µs]")
            ax.axvline(0, color = 'k', linewidth = 0.5)

        elif cue_type == "ild":
            visible_cues = cues
            if len(visible_cues) > 11:
                visible_cues = visible_cues[::2]
            ax.set_xticks(visible_cues)
            ax.set_xticklabels([f"{round(ipsi_level - c)}" for c in visible_cues])
            ax.set_xlabel("ILD [dB]")
            if xlim:
                ax.set_xlim(xlim)

        elif cue_type == "contra_level":
            visible_cues = cues
            if len(visible_cues) > 11:
                visible_cues = visible_cues[::2]
            ax.set_xticks(visible_cues)
            ax.set_xticklabels([f"{c}" for c in visible_cues])
            ax.set_xlabel("Contralateral Level [dB]")
            if xlim:
                ax.set_xlim(xlim)

        if ylim:
            ax.set_ylim(ylim)

        # ------------------------------------------------------------------
        # Title — always show resolved cf_interval + neuron count
        # ------------------------------------------------------------------
        base_title = (
            f"{pop_name} ({len(multi_data)} recordings)" if multi_mode else pop_name
        )

        filter_parts = []

        # Time interval
        if time_interval is not None:
            filter_parts.append(f"t=[{time_interval[0]},{time_interval[1]}] ms")

        # CF band — always derived from the resolved cf_interval
        if cf_interval is not None:
            # Resolve neuron count from the resolved cf_interval
            _first_cue = list(cue_to_rate.keys())[0]
            _n_tmp     = len(cue_to_rate[_first_cue]["L"][pop_name]["global_ids"])
            _cf_tmp    = greenwood_cf_array(CFMIN / Hz, CFMAX / Hz, _n_tmp) / Hz
            _, _ymin_idx = take_closest(_cf_tmp, cf_interval[0])
            _, _ymax_idx = take_closest(_cf_tmp, cf_interval[1])
            _n_band      = _ymax_idx - _ymin_idx + 1

            if target_cf_hz is not None:
                # Single neuron path
                filter_parts.append(
                    f"CF={_cf_tmp[_ymin_idx]:.0f} Hz (1 neuron)"
                )
            elif center_cf is not None and bw_neurons is not None:
                # center + bandwidth path — show original request + resolved band
                filter_parts.append(
                    f"CF={center_cf:.0f} ±{bw_neurons} neurons "
                    f"→ [{cf_interval[0]:.0f},{cf_interval[1]:.0f}] Hz "
                    f"({_n_band} neurons)"
                )
            else:
                # Direct cf_interval path
                filter_parts.append(
                    f"CF=[{cf_interval[0]:.0f},{cf_interval[1]:.0f}] Hz "
                    f"({_n_band} neurons)"
                )
        else:
            # No CF filter — full population
            _first_cue = list(cue_to_rate.keys())[0]
            _n_tmp     = len(cue_to_rate[_first_cue]["L"][pop_name]["global_ids"])
            filter_parts.append(f"CF=full ({_n_tmp} neurons)")

        ax.set_title(
            base_title + ("  |  " + ", ".join(filter_parts) if filter_parts else "")
        )

    # ---- SINGLE POP ----
    if isinstance(pop, str) and pop != "all":
        fig, ax = plt.subplots(figsize=figsize)
        _draw_single_pop_subplot(ax, pop, side)

        if title:
            ax.set_title(title)

        plt.tight_layout()
        plt.show()
        return ax

    # ---- ALL POPS ----
    pops = ["ANF", "SBC", "GBC", "LNTBC", "MNTBC", "MSO", "LSO"] if pop == "all" else list(pop)

    n_rows = math.ceil(len(pops) / 3)
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 4 * n_rows))
    axes = np.array(axes).flatten()

    for ax, p in zip(axes, pops):
        _draw_single_pop_subplot(ax, p, side)

    for j in range(len(pops), len(axes)):
        axes[j].axis("off")

    if title:
        fig.suptitle(title)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    return axes

def draw_rate_vs_cue_multidata(
    data_list,          # list of: single result dict OR list of result dicts
    pop='LSO',
    rate='avg',         # same valid set as draw_rate_vs_cue
    cf_interval=None,
    time_interval=None,
    target_cf_hz=None,  # scalar OR list/array with one value per group in data_list
    center_cf=None,
    bw_neurons=None,
    side='L',
    colors=None,
    labels=None,
    figsize=(7, 4),
    title=None,
    ylim=None,
    error='sem',
    shaded=True,
    cue_type="angle",
    xlim=None,
    alpha=0.8,
    lw=1.5,
    ipsi_level = None
):
    """
    Plot cue vs firing rate comparing groups of datasets.

    data_list : list of (single result dict OR list of result dicts)
        Each entry is one group. If an entry is a list of dicts, the group
        is averaged (mean ± SEM/STD shading). If a single dict, no shading.
    pop : str
        Population name (e.g. 'LSO', 'MSO').
    rate : str
        One of {'avg', 'pop', 'spk', 'spk_pn', 'mm_norm', 'max_norm'}.
    side : str
        'L' or 'R'.
    colors : list of str, optional
        One color per group. Defaults to tab10 cycle.
    labels : list of str, optional
        One label per group. Defaults to 'group 0', 'group 1', ...
    target_cf_hz : float OR list/array, optional
        If a single float, the same target CF is applied to every group
        (same behavior as before). If a list/array, it must have the same
        length as data_list — element i gives the target CF (Hz) for
        group i, so each group can be evaluated at its own single
        characteristic-frequency neuron.
    error : str
        'sem' or 'std' — controls error band when group has multiple runs.
    shaded : bool
        True → fill_between; False → errorbar caps.
    All other CF selection, time_interval, cue_type, xlim params work
    identically to draw_rate_vs_cue.
    """
    VALID_RATES = {'avg', 'pop', 'spk', 'spk_pn', 'mm_norm', 'max_norm'}
    if rate not in VALID_RATES:
        raise ValueError(f"rate must be one of {VALID_RATES}, got {rate!r}")

    # ------------------------------------------------------------------
    # Normalize: each entry may be a single dict or a list of dicts
    # ------------------------------------------------------------------
    groups = []
    for entry in data_list:
        if isinstance(entry, list):
            groups.append(entry)
        else:
            groups.append([entry])
    n_groups = len(groups)

    # ------------------------------------------------------------------
    # Colors and labels
    # ------------------------------------------------------------------
    if colors is None:
        cmap   = plt.get_cmap("tab10")
        colors = [cmap(i % 10) for i in range(n_groups)]
    if labels is None:
        labels = [f"group {i}" for i in range(n_groups)]

    # ------------------------------------------------------------------
    # Normalize target_cf_hz into a per-group list.
    # Scalar (or None) → broadcast to every group (old behavior).
    # list/array → must match n_groups, one target CF per group.
    # ------------------------------------------------------------------
    if target_cf_hz is None or np.isscalar(target_cf_hz):
        target_cf_hz_list = [target_cf_hz] * n_groups
    else:
        target_cf_hz_list = list(target_cf_hz)
        if len(target_cf_hz_list) != n_groups:
            raise ValueError(
                f"target_cf_hz list must have the same length as data_list "
                f"({n_groups} groups), got {len(target_cf_hz_list)}."
            )
        
    # ------------------------------------------------------------------
    # Normalize center_cf into a per-group list.
    # Scalar (or None) → broadcast to every group (old behavior).
    # list/array → must match n_groups, one center CF per group.
    # ------------------------------------------------------------------
    if center_cf is None or np.isscalar(center_cf):
        center_cf_list = [center_cf] * n_groups
    else:
        center_cf_list = list(center_cf)
        if len(center_cf_list) != n_groups:
            raise ValueError(
                f"center_cf list must have the same length as data_list "
                f"({n_groups} groups), got {len(center_cf_list)}."
            )

    # ------------------------------------------------------------------
    # Helper: filter spike dict by time window
    # ------------------------------------------------------------------
    def _filter_spike_dict(spike_dict, time_interval):
        if time_interval is None:
            return spike_dict
        times   = spike_dict["times"]
        senders = spike_dict["senders"]
        gids    = spike_dict["global_ids"]
        mask    = (times >= time_interval[0]) & (times <= time_interval[1])
        return {"times": times[mask], "senders": senders[mask], "global_ids": gids}

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)
    ylabel  = "Firing Rate [Hz]"  # overwritten inside loop
    cf_infos = []  # per-group CF description, used in the title

    for gi, (group, color, label) in enumerate(zip(groups, colors, labels)):
        this_target_cf = target_cf_hz_list[gi]
        this_center_cf = center_cf_list[gi]

        # --------------------------------------------------------------
        # Resolve CF interval for THIS group, using its own reference
        # dataset (safe even if groups have different population sizes).
        # --------------------------------------------------------------
        _ref_data    = group[0]
        _cue_to_rate = _ref_data["cue_to_rate"]
        _first_cue   = list(_cue_to_rate.keys())[0]
        _n_tmp       = len(_cue_to_rate[_first_cue][side][pop]["global_ids"])
        _cf_tmp      = greenwood_cf_array(CFMIN / Hz, CFMAX / Hz, _n_tmp) / Hz

        resolved_cf_interval = cf_interval  # may stay None
        cf_idx = None

        if this_target_cf is not None:
            _, cf_idx             = take_closest(_cf_tmp, this_target_cf)
            half_bin               = (_cf_tmp[1] - _cf_tmp[0]) * 0.5
            resolved_cf_interval   = [_cf_tmp[cf_idx] - half_bin, _cf_tmp[cf_idx] + half_bin]
            print(
                f"[draw_rate_vs_cue_multidata] group '{label}': "
                f"target_cf_hz={this_target_cf} Hz → neuron idx {cf_idx} → "
                f"cf_interval=[{resolved_cf_interval[0]:.1f}, {resolved_cf_interval[1]:.1f}] Hz"
            )
        elif this_center_cf is not None and bw_neurons is not None:
            _, center_idx = take_closest(_cf_tmp, this_center_cf)
            low_idx  = max(0, center_idx - bw_neurons)
            high_idx = min(_n_tmp - 1, center_idx + bw_neurons)
            if low_idx == high_idx:
                half_bin             = (_cf_tmp[1] - _cf_tmp[0]) * 0.5
                resolved_cf_interval = [_cf_tmp[low_idx] - half_bin, _cf_tmp[high_idx] + half_bin]
            else:
                resolved_cf_interval = [_cf_tmp[low_idx], _cf_tmp[high_idx]]
            print(
                f"[draw_rate_vs_cue_multidata] group '{label}': "
                f"center_cf={this_center_cf} Hz → neuron idx [{low_idx}, {high_idx}] → "
                f"cf_interval=[{resolved_cf_interval[0]:.1f}, {resolved_cf_interval[1]:.1f}] Hz"
            )

        # Neuron count for spk_pn and title, resolved per group
        if resolved_cf_interval is not None:
            _, _ymin_idx = take_closest(_cf_tmp, resolved_cf_interval[0])
            _, _ymax_idx = take_closest(_cf_tmp, resolved_cf_interval[1])
            n_band = _ymax_idx - _ymin_idx + 1
        else:
            n_band = _n_tmp

        if this_target_cf is not None:
            cf_infos.append(f"{label}: CF={_cf_tmp[cf_idx]:.0f} Hz (1 neuron)")
        elif this_center_cf is not None and bw_neurons is not None:
            cf_infos.append(
                f"{label}: CF={this_center_cf:.0f}±{bw_neurons}n → "
                f"[{resolved_cf_interval[0]:.0f},{resolved_cf_interval[1]:.0f}] Hz "
                f"({n_band} neurons)"
            )
        elif resolved_cf_interval is not None:
            cf_infos.append(
                f"{label}: CF=[{resolved_cf_interval[0]:.0f},{resolved_cf_interval[1]:.0f}] Hz "
                f"({n_band} neurons)"
            )
        else:
            cf_infos.append(f"{label}: CF=full ({n_band} neurons)")

        # --------------------------------------------------------------
        # Per-dataset firing rate extraction (unchanged apart from using
        # this group's resolved_cf_interval)
        # --------------------------------------------------------------
        all_avg   = []
        all_pop   = []
        all_count = []

        for d in group:
            cue_to_rate = d["cue_to_rate"]
            cues        = sorted(cue_to_rate.keys())
            default_duration = (
                d["basesound"].sound.duration / b2.ms
                if "basesound" in d
                else d["sounds"]["base_sound"].sound.duration / b2.ms
            )
            duration = d.get("simulation_time", default_duration) * b2.ms

            if time_interval is not None:
                effective_duration = (time_interval[1] - time_interval[0]) * b2.ms
                atr_filtered = {}
                for cue in cues:
                    atr_filtered[cue] = {side: {}}
                    for p in cue_to_rate[cue][side]:
                        atr_filtered[cue][side][p] = (
                            _filter_spike_dict(cue_to_rate[cue][side][p], time_interval)
                            if p == pop else cue_to_rate[cue][side][p]
                        )
                atr_to_use = atr_filtered
                dur_to_use = effective_duration
            else:
                atr_to_use = cue_to_rate
                dur_to_use = duration

            tot_d, avg_d, cnt_d = calculate_firing_rates(
                atr_to_use, pop, [side], cues, dur_to_use, resolved_cf_interval,
            )
            all_avg.append(avg_d[side])
            all_pop.append(tot_d[side])
            all_count.append(cnt_d[side])

        # Aggregate across runs in this group
        mean_avg   = np.mean(all_avg,   axis=0)
        mean_pop   = np.mean(all_pop,   axis=0)
        mean_count = np.mean(all_count, axis=0)

        multi = len(group) > 1
        if multi:
            if error == 'sem':
                _err = lambda x: np.std(x, axis=0) / np.sqrt(len(group))
            elif error == 'std':
                _err = lambda x: np.std(x, axis=0)
            else:
                raise ValueError("error must be 'sem' or 'std'")
            err_avg   = _err(all_avg)
            err_pop   = _err(all_pop)
            err_count = _err(all_count)
        else:
            err_avg = err_pop = err_count = None

        # --------------------------------------------------------------
        # Select rate mode — mirrors draw_rate_vs_cue exactly
        # --------------------------------------------------------------
        if rate == 'avg':
            y_vals, y_err = mean_avg,   err_avg
            ylabel        = "Avg Firing Rate [Hz]"
        elif rate == 'pop':
            y_vals, y_err = mean_pop,   err_pop
            ylabel        = "Population Firing Rate [Hz]"
        elif rate == 'spk':
            y_vals, y_err = mean_count, err_count
            ylabel        = "Spike Count"
        elif rate == 'spk_pn':
            y_vals = np.array(mean_count) / n_band
            y_err  = np.array(err_count)  / n_band if err_count is not None else None
            ylabel = "Spikes / Neuron"
        elif rate == 'mm_norm':
            y_min, y_max = np.array(mean_avg).min(), np.array(mean_avg).max()
            y_vals = (np.array(mean_avg) - y_min) / (y_max - y_min + 1e-12)
            y_err  = err_avg
            ylabel = "Min-Max Normalized Rate"
        elif rate == 'max_norm':
            y_vals = np.array(mean_avg) / (np.array(mean_avg).max() + 1e-12)
            y_err  = err_avg
            ylabel = "Max Normalized Rate"

        y_vals = np.array(y_vals)
        ax.plot(cues, y_vals, "o-", color=color, label=label, alpha=alpha, lw=lw)

        if multi and y_err is not None:
            y_err = np.array(y_err)
            if shaded:
                ax.fill_between(
                    cues, y_vals - y_err, y_vals + y_err,
                    alpha=0.2, color=color, linewidth=0,
                )
            else:
                ax.errorbar(
                    cues, y_vals, yerr=y_err,
                    fmt='none', capsize=3, color=color,
                )

    # ------------------------------------------------------------------
    # X-axis
    # ------------------------------------------------------------------
    if cue_type == "angle":
        ax.set_xticks(cues)
        ax.set_xticklabels([f"{int(c)}°" for c in cues])
        ax.set_xlabel("Azimuth Angle [deg]")
        if xlim:
            ax.set_xlim(xlim)
    elif cue_type == "itd":
        if xlim:
            xlim_s = [xlim[0]/1e6, xlim[1]/1e6]
            ax.set_xlim(xlim_s)
            visible_cues = [c for c in cues if xlim_s[0] <= c <= xlim_s[1]]
        else:
            visible_cues = cues

        max_ticks = 11
        if len(visible_cues) > max_ticks:
            step = max(1, len(visible_cues) // max_ticks)
            subsampled = visible_cues[::step]
            # Ensure 0 is always included
            if 0.0 not in subsampled:
                zero_idx = np.argmin(np.abs(np.array(visible_cues)))
                zero_val = visible_cues[zero_idx]
                subsampled = sorted(set(subsampled) | {zero_val})
            visible_cues = subsampled

        ax.set_xticks(visible_cues)
        ax.set_xticklabels(
            [f"{round(c * 1e6)}" for c in visible_cues],
            rotation=45,
            ha='right'
        )
        ax.set_xlabel("ITD [µs]")
    elif cue_type == "ild":
        visible_cues = cues
        if len(visible_cues) > 11:
            visible_cues = visible_cues[::2]
        ax.set_xticks(visible_cues)
        ax.set_xticklabels([f"{round(ipsi_level - c)}" for c in visible_cues])
        ax.set_xlabel("ILD [dB]")
        if xlim:
            ax.set_xlim(xlim)
    elif cue_type == "contra_level":
        visible_cues = cues
        if len(visible_cues) > 11:
            visible_cues = visible_cues[::2]
        ax.set_xticks(visible_cues)
        ax.set_xticklabels([f"{c}" for c in visible_cues])
        ax.set_xlabel("Contra Level [dB]")
        if xlim:
            ax.set_xlim(xlim)

    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(ylim)
    ax.legend(fontsize=8, ncol=max(1, n_groups // 10))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ------------------------------------------------------------------
    # Title — CF info is now per-group since each group may target a
    # different characteristic frequency.
    # ------------------------------------------------------------------
    header_parts = []
    if time_interval is not None:
        header_parts.append(f"t=[{time_interval[0]},{time_interval[1]}] ms")

    base_title = f"{pop} — side {side} ({n_groups} groups)"
    subtitle   = "  |  ".join(header_parts + cf_infos) if (header_parts or cf_infos) else ""
    # ax.set_title(base_title + ("\n" + subtitle if subtitle else ""), fontsize=9)
    ax.set_title(base_title)

    if title:
        fig.suptitle(title, fontsize=13, fontweight='bold')

    plt.tight_layout()
    return fig, ax

def draw_single_neuron_raster(
    data_list,
    pop,
    side,
    cue,
    target_cf_hz=None,
    center_cf=None,
    bw_neurons=0,
    xlim=None,
    psth_bin_size=1,
    hist_rate=True,
    dot_size=2,
    color=None,
    labels=None,
    title=None,
    figsize=(10, 7),
):
    if not data_list:
        raise ValueError("data_list is empty")

    if color is None:
        color = 'm' if side == 'L' else 'g'

    n_reps = len(data_list)
    if labels is None:
        labels = [f"rep {i}" for i in range(n_reps)]

    # ------------------------------------------------------------------
    # Resolve target neuron index from the first dataset
    # ------------------------------------------------------------------
    _ref       = data_list[0]
    _ctr       = _ref["cue_to_rate"]
    _spikes0   = _ctr[cue][side][pop]
    _gids      = _spikes0["global_ids"]
    _n         = len(_gids)
    _cf_arr    = greenwood_cf_array(CFMIN / b2.Hz, CFMAX / b2.Hz, _n) / b2.Hz

    if target_cf_hz is not None:
        _, neuron_idx = take_closest(_cf_arr, target_cf_hz)
    elif center_cf is not None:
        _, center_idx = take_closest(_cf_arr, center_cf)
        neuron_idx    = int(np.clip(center_idx + bw_neurons, 0, _n - 1))
    else:
        raise ValueError("Provide either target_cf_hz or center_cf.")

    neuron_gid = int(_gids[0]) + neuron_idx
    neuron_cf  = float(_cf_arr[neuron_idx])

    print(
        f"[draw_single_neuron_raster] pop={pop} side={side} "
        f"→ neuron idx={neuron_idx}, GID={neuron_gid}, CF={neuron_cf:.1f} Hz"
    )

    cues = sorted(_ctr.keys())

    if xlim is None:
        _default_dur = (
            _ref["basesound"].sound.duration / b2.ms
            if "basesound" in _ref
            else _ref["sounds"]["base_sound"].sound.duration / b2.ms
        )
        _dur = _ref.get("simulation_time", _default_dur)
        xlim = [0.0, float(_dur)]

    # ------------------------------------------------------------------
    # Retrieve sound waveform (from first dataset, first cue)
    # ------------------------------------------------------------------
    _sounds    = _ref["sounds"]
    _side_key  = "left_sounds" if side == "L" else "right_sounds"
    _hrtf_sound = _sounds[_side_key][cue]
    _t_sound    = np.arange(len(_hrtf_sound)) / _hrtf_sound.samplerate * 1000  # ms

    # ------------------------------------------------------------------
    # Layout: sound | raster | psth
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=figsize)
    gs  = GridSpec(
        3, 1, figure=fig,
        height_ratios=[0.5, 3, 1],
        hspace=0.35,
    )
    ax_sound  = fig.add_subplot(gs[0])
    ax_raster = fig.add_subplot(gs[1], sharex=ax_sound)
    ax_psth   = fig.add_subplot(gs[2], sharex=ax_sound)

    # ------------------------------------------------------------------
    # Sound panel
    # ------------------------------------------------------------------
    ax_sound.plot(_t_sound, _hrtf_sound, color=color, lw=0.8)
    ax_sound.set_xlim(xlim)
    ax_sound.set_ylabel(f"{'L' if side == 'L' else 'R'} sound")
    ax_sound.set_yticks([])
    ax_sound.spines["top"].set_visible(False)
    ax_sound.spines["right"].set_visible(False)
    ax_sound.spines["left"].set_visible(False)
    ax_sound.tick_params(labelbottom=False)

    # ------------------------------------------------------------------
    # Raster
    # ------------------------------------------------------------------
    all_spike_times = []

    for rep_idx, d in enumerate(data_list):
        spikes  = d["cue_to_rate"][cue][side][pop]
        times   = spikes["times"]
        senders = spikes["senders"]
        mask    = (
            (senders == neuron_gid) &
            (times   >= xlim[0])   &
            (times   <= xlim[1])
        )
        rep_times = times[mask]
        all_spike_times.append(rep_times)

        y_vals = np.full(len(rep_times), rep_idx)
        ax_raster.plot(
            rep_times, y_vals, '|',
            color=color,
            markersize=dot_size * 4,
            markeredgewidth=dot_size * 0.6,
        )

    ax_raster.set_ylim(-0.5, n_reps - 0.5)
    ax_raster.invert_yaxis()
    ax_raster.set_yticks(range(n_reps))
    ax_raster.set_yticklabels(labels, fontsize=8)
    ax_raster.set_ylabel("Repetition")
    ax_raster.set_title(
        f"{pop} | side {side} | cue = {cue} | CF = {neuron_cf:.0f} Hz "
        f"(GID {neuron_gid}, idx {neuron_idx})"
    )
    ax_raster.spines["top"].set_visible(False)
    ax_raster.spines["right"].set_visible(False)
    ax_raster.tick_params(labelbottom=False)

    # ------------------------------------------------------------------
    # PSTH
    # ------------------------------------------------------------------
    bins = np.arange(xlim[0], xlim[1] + psth_bin_size, psth_bin_size)
    all_times_pooled = np.concatenate(all_spike_times) if all_spike_times else np.array([])
    counts, _ = np.histogram(all_times_pooled, bins=bins)

    if hist_rate:
        rates = counts * (1000.0 / psth_bin_size) / n_reps
        ax_psth.bar(bins[:-1], rates, width=psth_bin_size * 0.9,
                    color=color, alpha=0.55, align='edge')
        ax_psth.axhline(rates.mean(), color=color, lw=1.5, ls='--', alpha=0.85)
        ax_psth.set_ylabel("Rate [Hz]")
    else:
        ax_psth.bar(bins[:-1], counts, width=psth_bin_size * 0.9,
                    color=color, alpha=0.55, align='edge')
        ax_psth.set_ylabel("Spike count")

    ax_psth.set_xlabel("Time [ms]")
    ax_psth.spines["top"].set_visible(False)
    ax_psth.spines["right"].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=13, fontweight='bold')

    plt.tight_layout()
    return fig, (ax_sound, ax_raster, ax_psth)# ─────────────────────────────────────────────────────────────────────────────

def draw_single_neuron_raster_by_cue(
    data,
    pop,
    side,
    target_cf_hz=None,
    center_cf=None,
    bw_neurons=0,
    xlim=None,
    psth_bin_size=1,
    hist_rate=True,
    dot_size=2,
    color=None,
    cmap_name='viridis',
    cues_to_plot=None,
    title=None,
    figsize=(10, 7),
):
    """
    Raster plot for a SINGLE result (one seed/recording): each row on the
    y-axis is a different cue (e.g. azimuth/ITD/ILD), for one selected
    neuron — as opposed to draw_single_neuron_raster, where rows are
    repetitions of the same cue.

    The sound panel shows one waveform per cue (since, unlike the
    repetitions case, the stimulus differs across rows), colored to match
    the cue ordering on the raster below.
    """
    if color is None:
        color = 'm' if side == 'L' else 'g'

    _ctr = data["cue_to_rate"]
    all_cues = sorted(_ctr.keys())
    cues = cues_to_plot if cues_to_plot is not None else all_cues
    n_cues = len(cues)

    # ---- Resolve target neuron (same logic, using first cue as reference) ----
    _ref_cue = cues[0]
    _spikes0 = _ctr[_ref_cue][side][pop]
    _gids    = _spikes0["global_ids"]
    _n       = len(_gids)
    _cf_arr  = greenwood_cf_array(CFMIN / b2.Hz, CFMAX / b2.Hz, _n) / b2.Hz

    if target_cf_hz is not None:
        _, neuron_idx = take_closest(_cf_arr, target_cf_hz)
    elif center_cf is not None:
        _, center_idx = take_closest(_cf_arr, center_cf)
        neuron_idx = int(np.clip(center_idx + bw_neurons, 0, _n - 1))
    else:
        raise ValueError("Provide either target_cf_hz or center_cf.")

    neuron_gid = int(_gids[0]) + neuron_idx
    neuron_cf  = float(_cf_arr[neuron_idx])

    print(
        f"[draw_single_neuron_raster_by_cue] pop={pop} side={side} "
        f"→ neuron idx={neuron_idx}, GID={neuron_gid}, CF={neuron_cf:.1f} Hz"
    )

    if xlim is None:
        _default_dur = (
            data["basesound"].sound.duration / b2.ms
            if "basesound" in data
            else data["sounds"]["base_sound"].sound.duration / b2.ms
        )
        _dur = data.get("simulation_time", _default_dur)
        xlim = [0.0, float(_dur)]

    # colormap across cues, shared between sound panel and raster row labels

    # ------------------------------------------------------------------
    # Layout: sound | raster | psth
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=figsize)
    gs  = GridSpec(
        3, 1, figure=fig,
        height_ratios=[0.5, 3, 1],
        hspace=0.35,
    )
    ax_sound  = fig.add_subplot(gs[0])
    ax_raster = fig.add_subplot(gs[1], sharex=ax_sound)
    ax_psth   = fig.add_subplot(gs[2], sharex=ax_sound)

    # ------------------------------------------------------------------
    # Sound panel — one waveform per cue
    # ------------------------------------------------------------------
    _sounds     = data["sounds"]
    _side_key   = "left_sounds" if side == "L" else "right_sounds"
    _hrtf_sound = _sounds[_side_key][0]
    _t_sound    = np.arange(len(_hrtf_sound)) / _hrtf_sound.samplerate * 1000  # ms

    ax_sound.plot(_t_sound, _hrtf_sound, color=color, lw=0.8)
    ax_sound.set_xlim(xlim)
    ax_sound.set_ylabel(f"{'L' if side == 'L' else 'R'} sound (cue=0)")
    ax_sound.set_yticks([])
    ax_sound.spines["top"].set_visible(False)
    ax_sound.spines["right"].set_visible(False)
    ax_sound.spines["left"].set_visible(False)
    ax_sound.tick_params(labelbottom=False)
    # ------------------------------------------------------------------
    # Raster
    # ------------------------------------------------------------------
    all_spike_times = []

    for row_idx, cue in enumerate(cues):
        spikes  = _ctr[cue][side][pop]
        times   = spikes["times"]
        senders = spikes["senders"]
        mask = (
            (senders == neuron_gid) &
            (times   >= xlim[0]) &
            (times   <= xlim[1])
        )
        cue_times = times[mask]
        all_spike_times.append(cue_times)

        y_vals = np.full(len(cue_times), row_idx)
        ax_raster.plot(
            cue_times, y_vals, '|',
            color=color,
            markersize=dot_size * 4,
            markeredgewidth=dot_size * 0.6,
        )

    ax_raster.set_ylim(-0.5, n_cues - 0.5)
    ax_raster.invert_yaxis()
    ax_raster.set_yticks(range(round(n_cues,2)))
    ax_raster.set_yticklabels([f"{c:.6f}" for c in cues], fontsize=8)


    ax_raster.set_ylabel("Cue")
    ax_raster.set_title(
        f"{pop} | side {side} | CF = {neuron_cf:.0f} Hz "
        f"(GID {neuron_gid}, idx {neuron_idx})"
    )
    ax_raster.spines["top"].set_visible(False)
    ax_raster.spines["right"].set_visible(False)
    ax_raster.tick_params(labelbottom=False)

    # ---- PSTH pooled across all plotted cues ----
    bins = np.arange(xlim[0], xlim[1] + psth_bin_size, psth_bin_size)
    all_times_pooled = np.concatenate(all_spike_times) if all_spike_times else np.array([])
    counts, _ = np.histogram(all_times_pooled, bins=bins)

    if hist_rate:
        rates = counts * (1000.0 / psth_bin_size) / n_cues
        ax_psth.bar(bins[:-1], rates, width=psth_bin_size * 0.9,
                    color=color, alpha=0.55, align='edge')
        ax_psth.axhline(rates.mean(), color=color, lw=1.5, ls='--', alpha=0.85)
        ax_psth.set_ylabel("Rate [Hz]")
    else:
        ax_psth.bar(bins[:-1], counts, width=psth_bin_size * 0.9,
                    color=color, alpha=0.55, align='edge')
        ax_psth.set_ylabel("Spike count")

    ax_psth.set_xlabel("Time [ms]")
    ax_psth.spines["top"].set_visible(False)
    ax_psth.spines["right"].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=13, fontweight='bold')

    plt.tight_layout()
    return fig, (ax_sound, ax_raster, ax_psth)# ─────────────────────────────────────────────────────────────────────────────

# METRIC COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_single_neuron_psth(
    spikes_file,
    target_cf_hz=None,
    cf_interval=None,
    center_cf=None,
    bw_neurons=None,
    xlim=None,
    ylim=None,
    bin_size=1.0,
    hist_rate=True,
    figsize=(10, 4),
    color='b'
):
    """
    Plot PSTH of one ANF neuron or a small CF neighborhood.

    spikes_file : list of dicts, length = n_reps
        Each element is a NEST spike dict:
            {"times": np.ndarray [ms], "senders": np.ndarray, "global_ids": np.ndarray}
        Single dict is auto-wrapped.

    CF selection (same priority as report_firing_rates_and_vs):
        target_cf_hz  → single neuron (n_neighbors=0 equivalent)
        center_cf + bw_neurons → band around center
        cf_interval   → direct [cf_min, cf_max] in Hz
        none          → full population
    """

    # ------------------------------------------------------------------
    # Auto-wrap single dict
    # ------------------------------------------------------------------
    if isinstance(spikes_file, dict):
        spikes_file = [spikes_file]

    n_reps    = len(spikes_file)
    n_neurons = len(spikes_file[0]["global_ids"])

    # ------------------------------------------------------------------
    # CF selection → (idx_min, idx_max)
    # ------------------------------------------------------------------
    cf_array = greenwood_cf_array(CFMIN / Hz, CFMAX / Hz, n_neurons) / Hz

    if target_cf_hz is not None:
        _, cf_idx = take_closest(cf_array, target_cf_hz)
        idx_min   = cf_idx
        idx_max   = cf_idx
        band_label = f"CF={cf_array[cf_idx]:.0f} Hz (single neuron)"

    elif center_cf is not None and bw_neurons is not None:
        _, center_idx = take_closest(cf_array, center_cf)
        idx_min = max(0, center_idx - bw_neurons)
        idx_max = min(n_neurons - 1, center_idx + bw_neurons)
        band_label = (
            f"center_cf={center_cf} Hz ±{bw_neurons} neurons → "
            f"[{cf_array[idx_min]:.1f}, {cf_array[idx_max]:.1f}] Hz"
        )

    elif cf_interval is not None:
        _, idx_min = take_closest(cf_array, cf_interval[0])
        _, idx_max = take_closest(cf_array, cf_interval[1])
        band_label = (
            f"cf_interval=[{cf_interval[0]:.0f}, {cf_interval[1]:.0f}] Hz → "
            f"neurons [{idx_min}, {idx_max}]"
        )

    else:
        idx_min = 0
        idx_max = n_neurons - 1
        band_label = "full population"

    sel_indices = np.arange(idx_min, idx_max + 1)
    n_sel       = len(sel_indices)

    print(f"  CF band    : {band_label}")
    print(f"  Neurons    : {n_sel}  |  Repetitions : {n_reps}")
    print(f"  Neurons ID range: {sel_indices[0]} to {sel_indices[-1]}")

    # ------------------------------------------------------------------
    # Collect spike times [ms] for selected neurons across all reps
    # ------------------------------------------------------------------
    pooled_times_ms = []

    for rep_dict in spikes_file:
        gids    = rep_dict["global_ids"]
        times_ms  = rep_dict["times"]
        senders   = rep_dict["senders"]

        # Global ID bounds for the selected band
        gid_min = gids[idx_min]
        gid_max = gids[idx_max]

        mask = (senders >= gid_min) & (senders <= gid_max)
        pooled_times_ms.append(times_ms[mask])

    if len(pooled_times_ms) == 0 or all(len(t) == 0 for t in pooled_times_ms):
        raise RuntimeError("No spikes found for selected neurons.")

    pooled_times_ms = np.concatenate(pooled_times_ms)

    if xlim is None:
        xlim = (0, pooled_times_ms.max())

    # ------------------------------------------------------------------
    # PSTH
    # ------------------------------------------------------------------
    bins   = np.arange(xlim[0], xlim[1] + bin_size, bin_size)
    counts, _ = np.histogram(pooled_times_ms, bins=bins)

    if hist_rate:
        y      = counts * 1000.0 / (bin_size * n_reps * n_sel)
        ylabel = "Firing rate [Hz]"
    else:
        y      = counts
        ylabel = "Spike count"

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(bins[:-1], y, lw=2, color=color)
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(alpha=0.3)
    ax.set_title(f"PSTH — {band_label}\n{n_reps} rep(s)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.show()

    return fig, ax

def calculate_single_neuron_vector_strength(
    spikes_file,
    target_cf_hz=None,
    cf_interval=None,
    center_cf=None,
    bw_neurons=None,
    n_neighbors=0,          # legacy alias for bw_neurons (single-neuron calls)
    n_bins=7,
    x_ax="phase",
    y_ax="percent",
    center_at_peak=False,
    figsize=(7, 5),
    color="b",
    ylim=None,
    display=True
):
    """
    Vector strength + phase/time histogram for a selected CF band
    across repeated simulations.

    spikes_file : list of dicts, length = n_reps
        Each element is a NEST spike dict:
            {"times": np.ndarray [ms], "senders": np.ndarray, "global_ids": np.ndarray}
        Single dict is auto-wrapped.

    CF selection (same priority as report_firing_rates_and_vs):
        target_cf_hz              → single neuron (or ± n_neighbors)
        center_cf + bw_neurons    → band around center
        cf_interval               → direct [cf_min, cf_max] in Hz
        none                      → full population
    """

    # ------------------------------------------------------------------
    # Auto-wrap single dict
    # ------------------------------------------------------------------
    if isinstance(spikes_file, dict):
        spikes_file = [spikes_file]

    n_reps    = len(spikes_file)
    n_neurons = len(spikes_file[0]["global_ids"])

    # ------------------------------------------------------------------
    # CF selection → (idx_min, idx_max)
    # ------------------------------------------------------------------
    cf_array = greenwood_cf_array(CFMIN / Hz, CFMAX / Hz, n_neurons) / Hz

    if target_cf_hz is not None:
        _, cf_idx = take_closest(cf_array, target_cf_hz)
        # n_neighbors kept for backward compat; maps to bw_neurons logic
        effective_bw = bw_neurons if bw_neurons is not None else n_neighbors
        idx_min = max(0, cf_idx - effective_bw)
        idx_max = min(n_neurons - 1, cf_idx + effective_bw)
        band_label = (
            f"CF={cf_array[cf_idx]:.0f} Hz"
            + (f" ±{effective_bw} neurons" if effective_bw > 0 else " (single neuron)")
        )
        ref_cf = cf_array[cf_idx]   # phase reference always at target

    elif center_cf is not None and bw_neurons is not None:
        _, center_idx = take_closest(cf_array, center_cf)
        idx_min = max(0, center_idx - bw_neurons)
        idx_max = min(n_neurons - 1, center_idx + bw_neurons)
        band_label = (
            f"center_cf={center_cf} Hz ±{bw_neurons} neurons → "
            f"[{cf_array[idx_min]:.1f}, {cf_array[idx_max]:.1f}] Hz"
        )
        ref_cf = cf_array[center_idx]

    elif cf_interval is not None:
        _, idx_min = take_closest(cf_array, cf_interval[0])
        _, idx_max = take_closest(cf_array, cf_interval[1])
        band_label = (
            f"cf_interval=[{cf_interval[0]:.0f}, {cf_interval[1]:.0f}] Hz → "
            f"neurons [{idx_min}, {idx_max}]"
        )
        ref_cf = cf_array[(idx_min + idx_max) // 2]   # midpoint CF

    else:
        idx_min = 0
        idx_max = n_neurons - 1
        band_label = "full population"
        ref_cf = cf_array[n_neurons // 2]

    sel_indices = np.arange(idx_min, idx_max + 1)
    n_sel       = len(sel_indices)

    # ------------------------------------------------------------------
    # Pool spike times [seconds] across repetitions and selected neurons
    # ------------------------------------------------------------------
    pooled_spike_times = []

    for rep_dict in spikes_file:
        gids     = rep_dict["global_ids"]
        times_ms = rep_dict["times"]
        senders  = rep_dict["senders"]

        gid_min = gids[idx_min]
        gid_max = gids[idx_max]

        mask = (senders >= gid_min) & (senders <= gid_max)
        if mask.any():
            pooled_spike_times.append(times_ms[mask] / 1000.0)   # ms → s

    if len(pooled_spike_times) == 0:
        return (0, None) if display else 0

    spike_times_array = np.concatenate(pooled_spike_times)
    total_spikes      = len(spike_times_array)

    # ------------------------------------------------------------------
    # Vector strength
    # ------------------------------------------------------------------
    phases = get_spike_phases(spike_times_array, ref_cf)
    vs     = calculate_vector_strength(spike_times_array, ref_cf)

    if not display:
        return vs

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # =========================
    # PHASE AXIS
    # =========================
    if x_ax == "phase":

        orig_bins = np.linspace(0, 2 * np.pi, n_bins + 1)
        hist_raw, _ = np.histogram(phases, bins=orig_bins)
        peak_bin_idx = np.argmax(hist_raw)

        if center_at_peak:
            bin_centers_orig = (orig_bins[:-1] + orig_bins[1:]) / 2
            peak_center      = bin_centers_orig[peak_bin_idx]
            values           = np.angle(np.exp(1j * (phases - peak_center)))  # fixed np.cue → np.angle
            bins             = np.linspace(-np.pi, np.pi, n_bins + 1)
            bin_centers      = (bins[:-1] + bins[1:]) / 2
        else:
            values      = phases
            bins        = orig_bins
            bin_centers = (bins[:-1] + bins[1:]) / 2

        hist, _   = np.histogram(values, bins=bins)
        bin_width = bins[1] - bins[0]

        if y_ax == "percent":
            y      = hist / total_spikes * 100
            ylabel = "Spikes / bin (% of total)"
        elif y_ax == "ashida":
            y      = hist / (total_spikes * bin_width)
            ylabel = "Probability density (rad$^{-1}$)"
        else:
            raise ValueError("y_ax must be 'percent' or 'ashida'")

        ax.bar(bin_centers, y, width=bin_width, alpha=0.7, color=color)
        ax.set_xlabel("Phase (cycles)")
        ax.set_xticks(
            np.array([0, 0.5, 1, 1.5, 2]) * np.pi if not center_at_peak
            else np.array([-1, -0.5, 0, 0.5, 1]) * np.pi
        )
        ax.set_xticklabels(
            ['0', '', '0.5', '', '1'] if not center_at_peak
            else ['-0.5', '', '0', '', '0.5']
        )

    # =========================
    # TIME AXIS
    # =========================
    elif x_ax == "time":

        period_ms        = 1000 / ref_cf
        time_values      = (phases / (2 * np.pi)) * period_ms

        orig_bins        = np.linspace(0, period_ms, n_bins + 1)
        hist_raw, _      = np.histogram(time_values, bins=orig_bins)
        peak_bin_idx     = np.argmax(hist_raw)

        if center_at_peak:
            bin_centers_orig = (orig_bins[:-1] + orig_bins[1:]) / 2
            peak_center      = bin_centers_orig[peak_bin_idx]
            values           = np.mod(
                time_values - peak_center + period_ms / 2, period_ms
            ) - period_ms / 2
            bins        = np.linspace(-period_ms / 2, period_ms / 2, n_bins + 1)
            bin_centers = (bins[:-1] + bins[1:]) / 2
        else:
            values      = time_values
            bins        = orig_bins
            bin_centers = (bins[:-1] + bins[1:]) / 2

        hist, _   = np.histogram(values, bins=bins)
        bin_width = bins[1] - bins[0]

        if y_ax == "percent":
            y      = hist / total_spikes * 100
            ylabel = "Spikes / bin (% of total)"
        elif y_ax == "ashida":
            y      = hist / (total_spikes * bin_width)
            ylabel = "Probability density (ms$^{-1}$)"
        else:
            raise ValueError("y_ax must be 'percent' or 'ashida'")

        ax.bar(bin_centers, y, width=bin_width, alpha=0.7, color=color)
        ax.set_xlabel("Time [ms]")

    ax.set_ylabel(ylabel)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_title(
        f"VS = {vs:.3f}  |  ref CF = {ref_cf:.0f} Hz\n"
        f"{band_label}  |  {n_reps} rep(s)  |  {total_spikes} spikes"
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.show()

    return vs, fig

def report_firing_rates_and_vs(
    spikes_file,
    pop,
    target_cf_hz=None,
    cf_interval=None,
    center_cf=None,
    bw_neurons=None,
    n_bins=7,
    x_ax="phase",
    y_ax="percent",
    center_at_peak=False,
    figsize=(7, 5),
    color="b",
    ylim=None,
    display_vs=True,
    error="sem",            # "sem" or "std"
    n_bootstrap=1000,       # bootstrap resamples for VS uncertainty
):
    """
    Report firing rates in three canonical windows and vector strength
    for a selected CF band.

    spikes_file : list of dicts, length = n_reps
        Each element is a NEST spike dict:
            {"times": np.ndarray [ms], "senders": np.ndarray, "global_ids": np.ndarray}
        Single dict is auto-wrapped.

    CF selection priority:
        target_cf_hz           → single neuron
        center_cf + bw_neurons → band around center
        cf_interval            → direct [cf_min, cf_max] in Hz
        none                   → full population

    Uncertainty:
        FR  → mean ± SEM (or STD) across repetitions
        VS  → mean ± SEM via bootstrap resampling across repetitions
              (only meaningful when n_reps > 1; reported but flagged otherwise)

    Windows (seconds)
    -----------------
    spontaneous : 0.150 – 0.300
    onset       : 0.005 – 0.015
    plateau     : 0.025 – 0.100
    """

    WINDOWS = {
        "spontaneous": (0.150, 0.300),
        "onset":       (0.005, 0.015),
        "plateau":     (0.025, 0.100),
    }

    # ------------------------------------------------------------------
    # Auto-wrap single dict
    # ------------------------------------------------------------------
    if isinstance(spikes_file, dict):
        spikes_file = [spikes_file]

    n_reps    = len(spikes_file)
    n_neurons = len(spikes_file[0]["global_ids"])

    # ------------------------------------------------------------------
    # CF selection → (idx_min, idx_max, ref_cf)
    # ------------------------------------------------------------------
    cf_array = greenwood_cf_array(CFMIN / Hz, CFMAX / Hz, n_neurons) / Hz

    if target_cf_hz is not None:
        _, cf_idx  = take_closest(cf_array, target_cf_hz)
        idx_min    = cf_idx
        idx_max    = cf_idx
        ref_cf     = cf_array[cf_idx]
        band_label = f"CF={ref_cf:.0f} Hz (single neuron)"

    elif center_cf is not None and bw_neurons is not None:
        _, center_idx = take_closest(cf_array, center_cf)
        idx_min    = max(0, center_idx - bw_neurons)
        idx_max    = min(n_neurons - 1, center_idx + bw_neurons)
        ref_cf     = cf_array[center_idx]
        band_label = (
            f"center_cf={center_cf} Hz ±{bw_neurons} neurons → "
            f"[{cf_array[idx_min]:.1f}, {cf_array[idx_max]:.1f}] Hz"
        )

    elif cf_interval is not None:
        _, idx_min = take_closest(cf_array, cf_interval[0])
        _, idx_max = take_closest(cf_array, cf_interval[1])
        ref_cf     = cf_array[(idx_min + idx_max) // 2]
        band_label = (
            f"cf_interval=[{cf_interval[0]:.0f}, {cf_interval[1]:.0f}] Hz → "
            f"neurons [{idx_min}, {idx_max}]"
        )

    else:
        idx_min    = 0
        idx_max    = n_neurons - 1
        ref_cf     = cf_array[n_neurons // 2]
        band_label = "full population"

    n_sel = idx_max - idx_min + 1

    print(f"\n{'='*60}")
    print(f"  Population : {pop}")
    print(f"  CF band    : {band_label}")
    print(f"  Neurons    : {n_sel}  |  Repetitions : {n_reps}")
    print(f"  Uncertainty: ±{error.upper()}" + (" (n=1: no uncertainty)" if n_reps == 1 else ""))
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # Extract spike times [seconds] per rep
    # ------------------------------------------------------------------
    spikes_by_rep = []

    for rep_dict in spikes_file:
        gids     = rep_dict["global_ids"]
        times_ms = rep_dict["times"]
        senders  = rep_dict["senders"]

        gid_min = gids[idx_min]
        gid_max = gids[idx_max]

        mask = (senders >= gid_min) & (senders <= gid_max)
        spikes_by_rep.append(times_ms[mask] / 1000.0)   # ms → s

    # ------------------------------------------------------------------
    # Helper: error metric
    # ------------------------------------------------------------------
    def _err(values):
        """values: 1-D array of per-rep measurements."""
        if len(values) < 2:
            return float("nan")
        if error == "sem":
            return np.std(values, ddof=1) / np.sqrt(len(values))
        elif error == "std":
            return np.std(values, ddof=1)
        else:
            raise ValueError("error must be 'sem' or 'std'")

    # ------------------------------------------------------------------
    # Firing rates per window  (per-rep → mean ± error)
    # ------------------------------------------------------------------
    col_w = 10
    err_label = error.upper()
    print(
        f"\n  {'Window':<14}  {'Duration':>10}  "
        f"{'FR [Hz]':>{col_w}}  {'±'+err_label:>{col_w}}"
    )
    print(f"  {'-'*14}  {'-'*10}  {'-'*col_w}  {'-'*col_w}")

    fr_results = {}

    for win_name, (t_start, t_end) in WINDOWS.items():
        win_dur = t_end - t_start

        # Per-rep firing rate
        per_rep_fr = np.array([
            np.sum((rep >= t_start) & (rep < t_end)) / (n_sel * win_dur)
            for rep in spikes_by_rep
        ])

        mean_fr = per_rep_fr.mean()
        err_fr  = _err(per_rep_fr)

        fr_results[win_name]              = mean_fr
        fr_results[f"{win_name}_{error}"] = err_fr

        err_str = f"{err_fr:.2f}" if not np.isnan(err_fr) else "  n/a"
        print(
            f"  {win_name:<14}  "
            f"{win_dur*1000:>8.0f} ms  "
            f"{mean_fr:>{col_w}.2f}  "
            f"{err_str:>{col_w}}"
        )

    # Ratios (propagate error via standard ratio error formula)
    def _ratio_err(a, sa, b, sb):
        """Relative error propagation for ratio a/b."""
        r = a / (b + 1e-12)
        if np.isnan(sa) or np.isnan(sb):
            return float("nan")
        return r * np.sqrt((sa / (a + 1e-12))**2 + (sb / (b + 1e-12))**2)

    onset_plateau_ratio = fr_results["onset"] / (fr_results["plateau"] + 1e-12)
    onset_plateau_err   = _ratio_err(
        fr_results["onset"],   fr_results[f"onset_{error}"],
        fr_results["plateau"], fr_results[f"plateau_{error}"]
    )

    plateau_spont_ratio = fr_results["plateau"] / (fr_results["spontaneous"] + 1e-12)
    plateau_spont_err   = _ratio_err(
        fr_results["plateau"],     fr_results[f"plateau_{error}"],
        fr_results["spontaneous"], fr_results[f"spontaneous_{error}"]
    )

    def _fmt_ratio(r, e):
        e_str = f"{e:.2f}" if not np.isnan(e) else "n/a"
        return f"{r:.2f} ± {e_str}"

    print(f"\n  Onset / Plateau ratio       : {_fmt_ratio(onset_plateau_ratio, onset_plateau_err)}")
    print(f"  Plateau / Spontaneous ratio : {_fmt_ratio(plateau_spont_ratio, plateau_spont_err)}")

    fr_results["onset_plateau_ratio"]  = onset_plateau_ratio
    fr_results["plateau_spont_ratio"]  = plateau_spont_ratio

    # ------------------------------------------------------------------
    # Vector Strength — plateau window, bootstrap uncertainty
    # ------------------------------------------------------------------
    t_vs_start, t_vs_end = WINDOWS["plateau"]

    plateau_by_rep = [
        rep[(rep >= t_vs_start) & (rep < t_vs_end)]
        for rep in spikes_by_rep
    ]
    pooled_plateau = np.concatenate(plateau_by_rep)

    print(f"\n  {'─'*56}")

    if len(pooled_plateau) == 0:
        print("  No spikes in plateau window — VS not computed.")
    else:
        vs = calculate_vector_strength(pooled_plateau, ref_cf)
        fr_results["vs"] = vs

        # Bootstrap VS uncertainty: resample reps with replacement
        if n_reps > 1:
            rng = np.random.default_rng(seed=42)
            boot_vs = np.array([
                calculate_vector_strength(
                    np.concatenate([
                        plateau_by_rep[i]
                        for i in rng.integers(0, n_reps, size=n_reps)
                    ]),
                    ref_cf
                )
                for _ in range(n_bootstrap)
            ])
            vs_err = (
                boot_vs.std(ddof=1) / np.sqrt(n_bootstrap)
                if error == "sem"
                else boot_vs.std(ddof=1)
            )
            vs_ci_lo, vs_ci_hi = np.percentile(boot_vs, [2.5, 97.5])
            fr_results[f"vs_{error}"] = vs_err
            fr_results["vs_ci95"]     = (vs_ci_lo, vs_ci_hi)

            vs_str = (
                f"{vs:.4f} ± {vs_err:.4f}  "
                f"[95% CI: {vs_ci_lo:.4f} – {vs_ci_hi:.4f}]"
            )
        else:
            vs_str = f"{vs:.4f}  (n=1 rep — no uncertainty estimate)"

        neuron_str = "single neuron" if n_sel == 1 else f"pooled {n_sel} neurons"
        print(
            f"  VS (plateau, {neuron_str} × {n_reps} reps, "
            f"ref CF={ref_cf:.0f} Hz)"
        )
        print(f"  = {vs_str}")
        if n_sel > 1:
            print(f"  [pooled VS — reflects population phase-locking]")

        if display_vs:
            calculate_single_neuron_vector_strength(
                spikes_file=spikes_file,
                target_cf_hz=ref_cf,
                bw_neurons=idx_max - idx_min,
                n_bins=n_bins,
                x_ax=x_ax,
                y_ax=y_ax,
                center_at_peak=center_at_peak,
                figsize=figsize,
                color=color,
                ylim=ylim,
                display=True,
            )

    print(f"{'='*60}\n")
    return fr_results

def get_avg_rate_vs_cue(
    data,
    pop='LSO',
    side='L',
    rate='avg',
    cf_interval=None,
    time_interval=None,
    target_cf_hz=None,
    center_cf=None,
    bw_neurons=None,
):
    """
    Extract the raw (cues, rate_curve) for one population/side, reusing the
    exact CF-resolution and rate-calculation logic used in draw_rate_vs_cue.
 
    `rate` selects which quantity is returned — same options as
    draw_rate_vs_cue: 'avg', 'pop', 'spk', 'spk_pn', 'mm_norm', 'max_norm'.
    Default ('avg') is unchanged from before, so existing calls (e.g. from
    extract_ild50_metrics) keep working without edits.
    """
    VALID_RATES = {'avg', 'pop', 'spk', 'spk_pn', 'mm_norm', 'max_norm'}
    if rate not in VALID_RATES:
        raise ValueError(f"rate must be one of {VALID_RATES}, got {rate!r}")
 
    if isinstance(data, list):
        multi_data = data
        data = data[0]
    else:
        multi_data = [data]
 
    cue_to_rate = data["cue_to_rate"]
    default_duration = (
        data["basesound"].sound.duration / b2.ms
        if "basesound" in data
        else data["sounds"]["base_sound"].sound.duration / b2.ms
    )
    duration = data.get("simulation_time", default_duration) * b2.ms
 
    # --- CF interval resolution (identical to draw_rate_vs_cue) ---
    if target_cf_hz is not None:
        _first_cue = list(cue_to_rate.keys())[0]
        _n_tmp = len(cue_to_rate[_first_cue]["L"][pop]["global_ids"])
        _cf_tmp = greenwood_cf_array(CFMIN / Hz, CFMAX / Hz, _n_tmp) / Hz
        _, cf_idx = take_closest(_cf_tmp, target_cf_hz)
        half_bin = (_cf_tmp[1] - _cf_tmp[0]) * 0.5
        cf_interval = [_cf_tmp[cf_idx] - half_bin, _cf_tmp[cf_idx] + half_bin]
    elif center_cf is not None and bw_neurons is not None:
        _first_cue = list(cue_to_rate.keys())[0]
        _n_tmp = len(cue_to_rate[_first_cue]["L"][pop]["global_ids"])
        _cf_tmp = greenwood_cf_array(CFMIN / Hz, CFMAX / Hz, _n_tmp) / Hz
        _, center_idx = take_closest(_cf_tmp, center_cf)
        low_idx = max(0, center_idx - bw_neurons)
        high_idx = min(_n_tmp - 1, center_idx + bw_neurons)
        if low_idx == high_idx:
            half_bin = (_cf_tmp[1] - _cf_tmp[0]) * 0.5
            cf_interval = [_cf_tmp[low_idx] - half_bin, _cf_tmp[high_idx] + half_bin]
        else:
            cf_interval = [_cf_tmp[low_idx], _cf_tmp[high_idx]]
 
    def _filter_spike_dict(spike_dict, time_interval):
        times, gids = spike_dict["times"], spike_dict["global_ids"]
        senders = spike_dict["senders"]
        if time_interval is None:
            return spike_dict
        mask = (times >= time_interval[0]) & (times <= time_interval[1])
        return {"times": times[mask], "senders": senders[mask], "global_ids": gids}
 
    effective_duration = (
        (time_interval[1] - time_interval[0]) * b2.ms
        if time_interval is not None else duration
    )
 
    cues = sorted(cue_to_rate.keys())
    all_pop, all_avg, all_count = [], [], []
 
    for d in multi_data:
        angle_to_rate_d = d["cue_to_rate"]
        if time_interval is not None:
            atr = {}
            for cue in cues:
                atr[cue] = {side: {}}
                for p, sd in angle_to_rate_d[cue][side].items():
                    atr[cue][side][p] = (
                        _filter_spike_dict(sd, time_interval) if p == pop else sd
                    )
            dur_d = effective_duration
        else:
            atr = angle_to_rate_d
            dur_d = d.get(
                "simulation_time",
                data["sounds"]["base_sound"].sound.duration / b2.ms,
            ) * b2.ms
 
        pop_d, avg_d, cnt_d = calculate_firing_rates(atr, pop, [side], cues, dur_d, cf_interval)
        all_pop.append(pop_d[side])
        all_avg.append(avg_d[side])
        all_count.append(cnt_d[side])
 
    mean_pop = np.mean(all_pop, axis=0)
    mean_avg = np.mean(all_avg, axis=0)
    mean_count = np.mean(all_count, axis=0)
 
    n_reps = len(multi_data)
    _sem = lambda arr: (np.std(arr, axis=0) / np.sqrt(n_reps)) if n_reps > 1 else None
    sem_pop, sem_avg, sem_count = _sem(all_pop), _sem(all_avg), _sem(all_count)
 
    # neuron count in the resolved CF band, needed for spk_pn
    _first_cue = list(cue_to_rate.keys())[0]
    _n_tmp = len(cue_to_rate[_first_cue]["L"][pop]["global_ids"])
    if cf_interval is not None:
        _cf_tmp = greenwood_cf_array(CFMIN / Hz, CFMAX / Hz, _n_tmp) / Hz
        _, _ymin_idx = take_closest(_cf_tmp, cf_interval[0])
        _, _ymax_idx = take_closest(_cf_tmp, cf_interval[1])
        n_neurons = _ymax_idx - _ymin_idx + 1
    else:
        n_neurons = _n_tmp
 
    if rate == 'avg':
        curve, curve_sem = mean_avg, sem_avg
    elif rate == 'pop':
        curve, curve_sem = mean_pop, sem_pop
    elif rate == 'spk':
        curve, curve_sem = mean_count, sem_count
    elif rate == 'spk_pn':
        curve = np.array(mean_count) / n_neurons
        curve_sem = (np.array(sem_count) / n_neurons) if sem_count is not None else None
    elif rate == 'mm_norm':
        normed, _ = normalize_rates({side: mean_avg}, [side])
        curve = normed[side]
        curve_sem = sem_avg  # error stays in original avg-rate units; rescale if you plot it normalized
    elif rate == 'max_norm':
        curve = np.array(mean_avg) / np.max(mean_avg)
        curve_sem = sem_avg
 
    return {
        "cues": np.array(cues, dtype=float),
        "rate": np.array(curve, dtype=float),
        "sem": curve_sem,
        "rate_mode": rate,
        "cf_interval": cf_interval,
        "n_neurons": n_neurons,
    }
 
def four_param_sigmoid(x, bottom, top, ild50, slope):
    """4-parameter logistic. slope>0 -> decreasing curve, slope<0 -> increasing."""
    return bottom + (top - bottom) / (1.0 + np.exp((x - ild50) / slope))

def extract_ild50_metrics(
    cues,
    rate,
    ipsi_ref='min',
    normalize=True,
    p0=None,
    maxfev=20000,
):
    """
    Fit a 4-parameter sigmoid to an avg-rate vs contralateral-level ('ild_exp')
    curve and extract max rate, min rate, and ILD50 (half-maximal contralateral
    level), following the normalize-to-monaural-ipsi -> sigmoid-fit approach.

    Parameters
    ----------
    cues : array-like
        Contralateral levels (dB) — x-axis of the ild_exp curve.
    rate : array-like
        Avg firing rate (Hz), same order as cues.
    ipsi_ref : 'min' | 'max' | float
        Which cue represents monaural ipsilateral stimulation (weakest/no
        contralateral drive), used as the 100% reference. Pick 'min' if the
        smallest contra level in your sweep is effectively "off", 'max' if
        it's the other end, or pass an explicit cue value.
    normalize : bool
        If True, rescale so rate at ipsi_ref == 100 before fitting (this
        matches the IID50 definition). If False, fits raw Hz.
    p0 : tuple, optional
        Initial guess (bottom, top, ild50, slope).

    Returns
    -------
    dict: max_rate, min_rate, ild50, slope, fit_params, fit_curve_fn,
          r_squared, cues, rate (post-normalization)
    """
    cues = np.asarray(cues, dtype=float)
    rate = np.asarray(rate, dtype=float)
    order = np.argsort(cues)
    cues, rate = cues[order], rate[order]

    if normalize:
        if ipsi_ref == 'min':
            ref_val = rate[0]
        elif ipsi_ref == 'max':
            ref_val = rate[-1]
        else:
            _, idx = take_closest(cues, ipsi_ref)
            ref_val = rate[idx]
        if ref_val == 0:
            raise ValueError("Reference (monaural ipsi) rate is 0 — cannot normalize.")
        y = 1 * rate / ref_val
    else:
        y = rate

    # auto-detect direction (LSO-type curves fall as contra level rises)
    corr = np.corrcoef(cues, y)[0, 1]
    sign = 1.0 if corr < 0 else -1.0

    if p0 is None:
        bottom0 = float(np.min(y))
        top0 = float(np.max(y))
        ild500 = cues[np.argmin(np.abs(y - (top0 + bottom0) / 2))]
        slope0 = sign * (cues[-1] - cues[0]) / 10.0
        p0 = (bottom0, top0, ild500, slope0)

    popt, pcov = curve_fit(four_param_sigmoid, cues, y, p0=p0, maxfev=maxfev)
    bottom, top, ild50, slope = popt

    y_fit = four_param_sigmoid(cues, *popt)
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "max_rate": float(np.max(y)),
        "min_rate": float(np.min(y)),
        "ild50": float(ild50),
        "slope": float(slope),
        "fit_params": {"bottom": bottom, "top": top, "ild50": ild50, "slope": slope},
        "fit_curve_fn": lambda x, p=popt: four_param_sigmoid(np.asarray(x, dtype=float), *p),
        "r_squared": float(r_squared),
        "normalized": normalize,
        "cues": cues,
        "rate": y,
    }

def plot_ild50_fit(
    metrics,
    ax=None,
    figsize=(6, 4.5),
    data_color='k',
    fit_color='C3',
    show_sem=True,
    sem=None,
    title=None,
    xlabel="Contralateral Level [dB]",
    ylabel=None,
    n_fit_points=200,
):
    """
    Plot the averaged data points (from extract_ild50_metrics) together with
    the fitted sigmoid, marking max, min, and ILD50.

    Parameters
    ----------
    metrics : dict
        Output of extract_ild50_metrics.
    ax : matplotlib Axes, optional
        Existing axes to draw on. Creates new figure if None.
    show_sem : bool
        If True and `sem` is provided, draws error bars on the data points.
    sem : array-like, optional
        Standard error per cue (e.g. curve["sem"] from get_avg_rate_vs_cue).
        Only meaningful if metrics was NOT normalized (normalize=False),
        since sem here is in raw Hz, not on the normalized scale.
    """
    cues = metrics["cues"]
    y = metrics["rate"]
    fit_fn = metrics["fit_curve_fn"]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # raw / normalized data points
    if show_sem and sem is not None and not metrics["normalized"]:
        ax.errorbar(
            cues, y, yerr=sem, fmt='o', color=data_color,
            capsize=3, label='data (mean ± SEM)', zorder=3,
        )
    else:
        ax.plot(cues, y, 'o', color=data_color, label='data', zorder=3)

    # smooth fitted sigmoid
    x_fit = np.linspace(cues.min(), cues.max(), n_fit_points)
    y_fit = fit_fn(x_fit)
    ax.plot(x_fit, y_fit, '-', color=fit_color, lw=2, label='sigmoid fit', zorder=2)

    # mark max, min, ILD50
    ax.axhline(metrics["max_rate"], color=fit_color, ls=':', lw=1, alpha=0.6)
    ax.axhline(metrics["min_rate"], color=fit_color, ls=':', lw=1, alpha=0.6)

    ild50 = metrics["ild50"]
    y_at_ild50 = fit_fn(np.array([ild50]))[0]
    ax.axvline(ild50, color=fit_color, ls='--', lw=1, alpha=0.7)
    ax.plot(ild50, y_at_ild50, 'D', color=fit_color, ms=7, zorder=4,
            label=f'ILD50 = {ild50:.1f} dB')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or ("Normalized Rate [% ipsi]" if metrics["normalized"] else "Avg Firing Rate [Hz]"))
    ax.set_title(title or f"R² = {metrics['r_squared']:.3f}")
    ax.legend(loc='best', fontsize=8)

    return ax

def compute_metrics_for_angle(res, cue, pop,
                               xlim_peak=(0, 20), bin_size=0.5,
                               center_cf=None, bw_neurons=None,
                               psth_filtering=False, stim_fs=None,
                               smooth_cutoff_hz=None, prominence=50.0):
    n_peaks_list = [2, 5, 10]
    out = {}

    psth_kwargs = dict(
        xlim_peak        = xlim_peak,
        bin_size         = bin_size,
        center_cf        = center_cf,
        bw_neurons       = bw_neurons,
        psth_filtering   = psth_filtering,
        stim_fs          = stim_fs,
        smooth_cutoff_hz = smooth_cutoff_hz,
    )

    rates_L, centres_L = _get_psth_rates(
        res["cue_to_rate"][cue]["L"][pop], **psth_kwargs)
    rates_R, centres_R = _get_psth_rates(
        res["cue_to_rate"][cue]["R"][pop], **psth_kwargs)

    # ── onset peak ────────────────────────────────────────────────────────────
    t_on_L, fr_on_L = _onset_peak(rates_L, centres_L, bin_size, prominence)
    t_on_R, fr_on_R = _onset_peak(rates_R, centres_R, bin_size, prominence)

    out["fr_onset_L"]    = fr_on_L
    out["fr_onset_R"]    = fr_on_R
    out["fr_onset_diff"] = fr_on_L - fr_on_R
    out["t_onset_L"]     = t_on_L
    out["t_onset_R"]     = t_on_R
    out["t_onset_diff"]  = t_on_L - t_on_R

    # ── period: exact from stimulus frequency, no estimation ─────────────────
    if stim_fs is not None:
        period = 1000.0 / stim_fs          # ms
    else:
        # fallback to IPI estimate if stim_fs not provided
        all_t_L, _ = _get_sorted_peaks(rates_L, centres_L, bin_size, prominence=prominence)
        all_t_R, _ = _get_sorted_peaks(rates_R, centres_R, bin_size, prominence=prominence)
        ipi = []
        if len(all_t_L) > 1: ipi.append(np.diff(all_t_L))
        if len(all_t_R) > 1: ipi.append(np.diff(all_t_R))
        period = np.mean(np.concatenate(ipi)) if ipi else np.nan
    out["period"] = period

    # ── phase: relative to onset peak, using exact period ────────────────────
    # t_on gives the phase reference: the onset peak defines cycle position 0
    # all subsequent peak times are expressed relative to that reference
    if not np.isnan(period) and period > 0:
        ph_on_L = ((t_on_L - t_on_L) % period) / period * 2 * np.pi   # always 0
        ph_on_R = ((t_on_R - t_on_L) % period) / period * 2 * np.pi   # shift of R vs L onset
        out["phase_onset_diff"] = (ph_on_L - ph_on_R + np.pi) % (2 * np.pi) - np.pi
    else:
        out["phase_onset_diff"] = np.nan

    # ── multi-peak averages ───────────────────────────────────────────────────
    for n in n_peaks_list:
        t_L, fr_L = _get_sorted_peaks(rates_L, centres_L, bin_size,
                                       n_peaks=n, prominence=prominence)
        t_R, fr_R = _get_sorted_peaks(rates_R, centres_R, bin_size,
                                       n_peaks=n, prominence=prominence)

        avg_fr_L = fr_L.mean() if len(fr_L) else np.nan
        avg_fr_R = fr_R.mean() if len(fr_R) else np.nan
        out[f"fr_avg_{n}_L"]    = avg_fr_L
        out[f"fr_avg_{n}_R"]    = avg_fr_R
        out[f"fr_avg_{n}_diff"] = avg_fr_L - avg_fr_R

        avg_t_L = t_L.mean() if len(t_L) else np.nan
        avg_t_R = t_R.mean() if len(t_R) else np.nan
        out[f"t_avg_{n}_L"]     = avg_t_L
        out[f"t_avg_{n}_R"]     = avg_t_R
        out[f"t_avg_{n}_diff"]  = avg_t_L - avg_t_R

        if not np.isnan(period) and period > 0:
            # phase of each peak relative to the L onset reference
            ph_L = np.mean(((t_L - t_on_L) % period) / period * 2 * np.pi) if len(t_L) else np.nan
            ph_R = np.mean(((t_R - t_on_L) % period) / period * 2 * np.pi) if len(t_R) else np.nan
            out[f"phase_avg_{n}_diff"] = (ph_L - ph_R + np.pi) % (2 * np.pi) - np.pi
        else:
            out[f"phase_avg_{n}_diff"] = np.nan

    return out

def compute_lateralization_metrics(res, pop, cues,
                                    xlim_peak=(0, 20), bin_size=0.5,
                                    center_cf=None, bw_neurons=None,
                                    psth_filtering=False, stim_fs=None,
                                    smooth_cutoff_hz=None, prominence=50.0):
    """Sweep cues and stack results into arrays."""
    keys = None
    rows = []
    for cue in cues:
        m = compute_metrics_for_angle(
            res, cue, pop,
            xlim_peak=xlim_peak, bin_size=bin_size,
            center_cf=center_cf, bw_neurons=bw_neurons,
            psth_filtering=psth_filtering, stim_fs=stim_fs,
            smooth_cutoff_hz=smooth_cutoff_hz, prominence=prominence,
        )
        if keys is None:
            keys = list(m.keys())
        rows.append(m)

    results = {"cues": np.array(cues)}
    for k in keys:
        results[k] = np.array([r[k] for r in rows])
    return results

def extract_first_spike_latency_metrics(
    data_list,
    pop,
    side,
    cue,
    target_cf_hz=None,
    center_cf=None,
    bw_neurons=0,
    onset_time=0.0,
    window=None,
):
    """
    First-spike latency and jitter for a single neuron across repetitions:
    "Median latency and jitter, defined as the 25-75% quartile range, were
    computed for the first spikes after stimulus onset, averaged over n
    repetitions."

    Parameters
    ----------
    data_list : list of result dicts, one per repetition
        (same structure/order as draw_single_neuron_raster's data_list)
    pop, side, cue : population, hemisphere, cue value to select.
    target_cf_hz / center_cf : neuron selection, same semantics as
        draw_single_neuron_raster.
    onset_time : float
        Stimulus onset (ms); latency is measured relative to this.
    window : [start, end] ms, optional
        Absolute time window to search for the first spike. Defaults to
        [onset_time, trial end] inferred from the first dataset.

    Returns
    -------
    dict: median_latency, jitter_iqr, q25, q75, latencies (per responding
    rep), n_reps, n_responses, response_rate, neuron_gid, neuron_cf.
    """
    if not data_list:
        raise ValueError("data_list is empty")

    n_reps = len(data_list)

    # ---- Resolve target neuron (identical logic to draw_single_neuron_raster) ----
    _ref     = data_list[0]
    _ctr     = _ref["cue_to_rate"]
    _spikes0 = _ctr[cue][side][pop]
    _gids    = _spikes0["global_ids"]
    _n       = len(_gids)
    _cf_arr  = greenwood_cf_array(CFMIN / b2.Hz, CFMAX / b2.Hz, _n) / b2.Hz

    if target_cf_hz is not None:
        _, neuron_idx = take_closest(_cf_arr, target_cf_hz)
    elif center_cf is not None:
        _, center_idx = take_closest(_cf_arr, center_cf)
        neuron_idx = int(np.clip(center_idx + bw_neurons, 0, _n - 1))
    else:
        raise ValueError("Provide either target_cf_hz or center_cf.")

    neuron_gid = int(_gids[0]) + neuron_idx
    neuron_cf  = float(_cf_arr[neuron_idx])

    if window is None:
        _default_dur = (
            _ref["basesound"].sound.duration / b2.ms
            if "basesound" in _ref
            else _ref["sounds"]["base_sound"].sound.duration / b2.ms
        )
        _dur = _ref.get("simulation_time", _default_dur)
        window = [onset_time, float(_dur)]

    # ---- First-spike latency per repetition ----
    latencies = []
    for d in data_list:
        spikes  = d["cue_to_rate"][cue][side][pop]
        times   = spikes["times"]
        senders = spikes["senders"]
        mask = (
            (senders == neuron_gid) &
            (times   >= window[0]) &
            (times   <= window[1])
        )
        rep_times = np.sort(times[mask])
        if rep_times.size > 0:
            latencies.append(rep_times[0] - onset_time)
        # non-responding reps are excluded, not zero-padded

    latencies = np.array(latencies, dtype=float)
    n_responses = latencies.size

    if n_responses == 0:
        return {
            "median_latency": np.nan, "jitter_iqr": np.nan,
            "q25": np.nan, "q75": np.nan, "latencies": latencies,
            "n_reps": n_reps, "n_responses": 0, "response_rate": 0.0,
            "neuron_gid": neuron_gid, "neuron_cf": neuron_cf,
        }

    q25 = float(np.percentile(latencies, 25))
    q75 = float(np.percentile(latencies, 75))

    return {
        "median_latency": float(np.median(latencies)),
        "jitter_iqr": q75 - q25,
        "q25": q25, "q75": q75,
        "latencies": latencies,
        "n_reps": n_reps, "n_responses": n_responses,
        "response_rate": n_responses / n_reps,
        "neuron_gid": neuron_gid, "neuron_cf": neuron_cf,
    }

def extract_mso_itd_peak_metrics(
    data_by_freq,
    pop='MSO',
    side='L',
    cf_interval=None,
    target_cf_hz=None,
    center_cf=None,
    bw_neurons=None,
    time_interval=None,
    fit_gaussian=True,
    gaussian_window=2,
):
    """
    Per-neuron ITD-tuning peak extraction for MSO. Complementary to the
    CD/CP pipeline (which derives best IPD from vector analysis + Rayleigh
    test): here the "best ITD" is read directly off the rate-vs-ITD curve.
    Useful as a sanity check, and as a fallback for neurons that don't hit
    3+ significant frequencies for CD/CP regression.
 
    data_by_freq: {freq_hz: dataset}, same convention as the CD/CP function:
    dataset["cue_to_rate"][itd_seconds][side][pop] holds the spike dict
    {times, senders, global_ids} for an ITD sweep at that frequency.
    (If your actual layout differs — e.g. a single dataset keyed by
    (freq, itd) tuples instead of one dict per frequency — the CF-band
    resolution and _per_neuron_curve extraction below are the two spots
    to adapt; everything downstream operates on plain (itds, curve) arrays.)
 
    For each neuron and each frequency:
      1. builds the raw rate-vs-ITD curve (spike count / duration, per neuron)
      2. argmax → raw peak ITD
      3. 3-point parabolic (Lagrange) interpolation around the argmax for
         sub-bin peak precision — this is the main "find the peak" step
      4. optionally fits offset + amp*exp(-(x-mu)^2/2sigma^2) on a small
         window around the peak, as a smoother alternative to (3)
 
    Returns {neuron_id: {...}} with per-frequency curves/peaks plus a
    frequency-independent summary: best_freq (frequency giving the highest
    peak rate for that neuron) and best_itd (parabolic peak ITD at that
    frequency).
    """
    freqs = sorted(data_by_freq.keys())
 
    # --- resolve which neurons fall in the CF band, from the first frequency ---
    _first_data = data_by_freq[freqs[0]]
    _first_cue_to_rate = _first_data["cue_to_rate"]
    _first_cue = list(_first_cue_to_rate.keys())[0]
    all_gids = np.array(_first_cue_to_rate[_first_cue][side][pop]["global_ids"])
    n_total = len(all_gids)
 
    if target_cf_hz is not None:
        _cf_tmp = greenwood_cf_array(CFMIN / Hz, CFMAX / Hz, n_total) / Hz
        _, cf_idx = take_closest(_cf_tmp, target_cf_hz)
        half_bin = (_cf_tmp[1] - _cf_tmp[0]) * 0.5
        cf_interval = [_cf_tmp[cf_idx] - half_bin, _cf_tmp[cf_idx] + half_bin]
    elif center_cf is not None and bw_neurons is not None:
        _cf_tmp = greenwood_cf_array(CFMIN / Hz, CFMAX / Hz, n_total) / Hz
        _, center_idx = take_closest(_cf_tmp, center_cf)
        low_idx = max(0, center_idx - bw_neurons)
        high_idx = min(n_total - 1, center_idx + bw_neurons)
        if low_idx == high_idx:
            half_bin = (_cf_tmp[1] - _cf_tmp[0]) * 0.5
            cf_interval = [_cf_tmp[low_idx] - half_bin, _cf_tmp[high_idx] + half_bin]
        else:
            cf_interval = [_cf_tmp[low_idx], _cf_tmp[high_idx]]
 
    if cf_interval is not None:
        _cf_tmp = greenwood_cf_array(CFMIN / Hz, CFMAX / Hz, n_total) / Hz
        _, lo = take_closest(_cf_tmp, cf_interval[0])
        _, hi = take_closest(_cf_tmp, cf_interval[1])
        neuron_ids = all_gids[lo:hi + 1]
    else:
        neuron_ids = all_gids
 
    def _filter_spike_dict(spike_dict, time_interval):
        times, senders, gids = spike_dict["times"], spike_dict["senders"], spike_dict["global_ids"]
        if time_interval is None:
            return spike_dict
        mask = (times >= time_interval[0]) & (times <= time_interval[1])
        return {"times": times[mask], "senders": senders[mask], "global_ids": gids}
 
    def _per_neuron_curve(dataset):
        cue_to_rate = dataset["cue_to_rate"]
        default_duration = (
            dataset["basesound"].sound.duration / b2.ms
            if "basesound" in dataset
            else dataset["sounds"]["base_sound"].sound.duration / b2.ms
        )
        duration_ms = dataset.get("simulation_time", default_duration)
        if time_interval is not None:
            duration_ms = time_interval[1] - time_interval[0]
        duration_s = duration_ms / 1000.0
 
        itds = sorted(cue_to_rate.keys())
        curves = {nid: np.zeros(len(itds)) for nid in neuron_ids}
        for ci, itd in enumerate(itds):
            sd = cue_to_rate[itd][side][pop]
            if time_interval is not None:
                sd = _filter_spike_dict(sd, time_interval)
            senders = np.asarray(sd["senders"])
            for nid in neuron_ids:
                curves[nid][ci] = np.sum(senders == nid) / duration_s
        return np.array(itds, dtype=float), curves
 
    def _parabolic_refine(x, y, idx):
        """3-point Lagrange parabola through the peak and its two neighbors."""
        if idx == 0 or idx == len(x) - 1:
            return x[idx], y[idx]
        x0, x1, x2 = x[idx - 1], x[idx], x[idx + 1]
        y0, y1, y2 = y[idx - 1], y[idx], y[idx + 1]
        denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
        if denom == 0:
            return x1, y1
        A = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
        B = (x2**2 * (y0 - y1) + x1**2 * (y2 - y0) + x0**2 * (y1 - y2)) / denom
        if A == 0:
            return x1, y1
        C = y1 - A * x1**2 - B * x1
        x_peak = -B / (2 * A)
        y_peak = A * x_peak**2 + B * x_peak + C
        return x_peak, y_peak
 
    def _gaussian_refine(x, y, idx, window):
        lo = max(0, idx - window)
        hi = min(len(x), idx + window + 1)
        xw, yw = np.asarray(x[lo:hi]), np.asarray(y[lo:hi])
        if len(xw) < 4:
            return None
        def gauss(t, offset, amp, mu, sigma):
            return offset + amp * np.exp(-((t - mu) ** 2) / (2 * sigma**2))
        span = (xw[-1] - xw[0]) or 1e-9
        p0 = [np.min(yw), np.max(yw) - np.min(yw), x[idx], span / 4]
        try:
            popt, _ = curve_fit(gauss, xw, yw, p0=p0, maxfev=5000)
            return popt[2]  # mu
        except Exception:
            return None
 
    results = {}
    for nid in neuron_ids:
        per_freq = {}
        best_freq, best_rate = None, -np.inf
 
        for f in freqs:
            itds, curves = _per_neuron_curve(data_by_freq[f])
            curve = curves[nid]
            idx = int(np.argmax(curve))
            raw_itd, raw_rate = itds[idx], curve[idx]
            par_itd, par_rate = _parabolic_refine(itds, curve, idx)
            gauss_itd = (
                _gaussian_refine(itds, curve, idx, gaussian_window)
                if fit_gaussian else None
            )
 
            per_freq[f] = {
                "itds": itds,
                "curve": curve,
                "raw_peak_itd": raw_itd,
                "raw_peak_rate": raw_rate,
                "parabolic_peak_itd": par_itd,
                "parabolic_peak_rate": par_rate,
                "gaussian_peak_itd": gauss_itd,
            }
            if raw_rate > best_rate:
                best_rate, best_freq = raw_rate, f
 
        results[nid] = {
            "by_freq": per_freq,
            "best_freq": best_freq,
            "best_itd": per_freq[best_freq]["parabolic_peak_itd"],
            "best_itd_raw": per_freq[best_freq]["raw_peak_itd"],
            "best_itd_gaussian": per_freq[best_freq]["gaussian_peak_itd"],
        }
 
    return results
 
def extract_best_itd_ipd(curve, freq_hz):
    """
    Given one population-level curve dict from get_avg_rate_vs_cue
    (curve['cues'] = ITDs in seconds, curve['rate'] = rate vs ITD) and the
    tone frequency it was recorded at, find the peak of the curve (best
    ITD) and convert it to best IPD (in cycles) via IPD = ITD * freq.
 
    Returns one dict per call -- accumulate these across your frequency
    loop, then pass the list to plot_best_itd_ipd_vs_freq.
    """
    itds = np.asarray(curve["cues"])
    rates = np.asarray(curve["rate"])
 
    idx = int(np.argmax(rates))
    raw_itd, raw_rate = itds[idx], rates[idx]
 
    # 3-point parabolic refine for sub-bin precision
    if 0 < idx < len(itds) - 1:
        x0, x1, x2 = itds[idx - 1], itds[idx], itds[idx + 1]
        y0, y1, y2 = rates[idx - 1], rates[idx], rates[idx + 1]
        denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
        A = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom if denom else 0
        B = (x2**2 * (y0 - y1) + x1**2 * (y2 - y0) + x0**2 * (y1 - y2)) / denom if denom else 0
        if A != 0:
            best_itd = -B / (2 * A)
            best_rate = A * best_itd**2 + B * best_itd + (y1 - A * x1**2 - B * x1)
        else:
            best_itd, best_rate = raw_itd, raw_rate
    else:
        best_itd, best_rate = raw_itd, raw_rate
 
    best_ipd_cycles = best_itd * freq_hz  # ITD in seconds * freq in Hz -> cycles
    best_ipd_deg = best_ipd_cycles * 360.0
 
    return {
        "freq_hz": freq_hz,
        "best_itd": best_itd,          # seconds
        "best_rate": best_rate,
        "best_ipd_cycles": best_ipd_cycles,
        "best_ipd_deg": best_ipd_deg,
    }
 
def plot_best_itd_ipd_vs_freq(results, figsize=(10, 4)):
    """
    results: list of dicts as returned by extract_best_itd_ipd, one per
    frequency. Plots best ITD (µs) and best IPD (cycles) vs frequency.
    """
    results = sorted(results, key=lambda r: r["freq_hz"])
    freqs = [r["freq_hz"] for r in results]
    itds_us = [r["best_itd"] * 1e6 for r in results]
    ipds_cyc = [r["best_ipd_cycles"] for r in results]
 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
 
    ax1.plot(freqs, itds_us, "o-", color="m")
    ax1.set_xlabel("Frequency [Hz]")
    ax1.set_ylabel("Best ITD [µs]")
    ax1.set_title("Best ITD vs Frequency")
 
    ax2.plot(freqs, ipds_cyc, "o-", color="g")
    ax2.set_xlabel("Frequency [Hz]")
    ax2.set_ylabel("Best IPD [cycles]")
    ax2.set_title("Best IPD vs Frequency")
 
    plt.tight_layout()
    plt.show()
    return fig, (ax1, ax2)
 
# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────
def plot_lateralization_metrics(metrics,
                                 show_onset_only=False,
                                 show_lr_on_onset=True):
    cues       = metrics["cues"]
    n_peaks_list = [2, 5, 10]
    avg_styles   = {2: ('--', 'o'), 5: ('-.', 's'), 10: (':', '^')}
    avg_colors   = {2: '#e07b00', 5: '#9400d3', 10: '#007090'}

    fig, axes = plt.subplots(3, 1, figsize=(7, 15), sharex=True)

    # ── Panel 0: FR difference ────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(cues, metrics["fr_onset_diff"], 'o-', color='k',
            linewidth=2, label="onset peak")
    ax.axhline(0, color='gray', linestyle=':', linewidth=1)
    ax.axvline(0, color='gray', linestyle=':', linewidth=1)
    ax.set_ylabel("ΔFR  L−R  [Hz]")
    ax.grid(True, alpha=0.3)
    if not show_onset_only:
        for n in n_peaks_list:
            ls, mk = avg_styles[n]
            ax.plot(cues, metrics[f"fr_avg_{n}_diff"],
                    linestyle=ls, marker=mk, color=avg_colors[n],
                    linewidth=1.5, label=f"avg first {n} peaks")
    if show_onset_only and show_lr_on_onset:
        ax2 = ax.twinx()
        ax2.plot(cues, metrics["fr_onset_L"], 's--', color='m',
                 linewidth=1.5, alpha=0.7, label="L onset FR")
        ax2.plot(cues, metrics["fr_onset_R"], 's--', color='g',
                 linewidth=1.5, alpha=0.7, label="R onset FR")
        ax2.set_ylabel("Individual FR [Hz]", color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')
        ax2.legend()
    ax.legend()

    # ── Panel 1: Timing difference ────────────────────────────────────────────
    ax = axes[1]
    ax.plot(cues, metrics["t_onset_diff"], 'o-', color='k',
            linewidth=2, label="onset peak")
    ax.axhline(0, color='gray', linestyle=':', linewidth=1)
    ax.axvline(0, color='gray', linestyle=':', linewidth=1)
    ax.set_ylabel("Δt  L−R  [ms]")
    ax.grid(True, alpha=0.3)
    if not show_onset_only:
        for n in n_peaks_list:
            ls, mk = avg_styles[n]
            ax.plot(cues, metrics[f"t_avg_{n}_diff"],
                    linestyle=ls, marker=mk, color=avg_colors[n],
                    linewidth=1.5, label=f"avg first {n} peaks")
    ax.legend()

    # ── Panel 2: Phase difference ─────────────────────────────────────────────
    ax = axes[2]
    phase_onset_deg = np.degrees(metrics["phase_onset_diff"])
    ax.plot(cues, phase_onset_deg, 'o-', color='k',
            linewidth=2, label="onset peak")
    ax.axhline(0, color='gray', linestyle=':', linewidth=1)
    ax.axvline(0, color='gray', linestyle=':', linewidth=1)
    ax.set_ylabel("ΔPhase  L−R  [deg]")
    ax.grid(True, alpha=0.3)
    if not show_onset_only:
        for n in n_peaks_list:
            ls, mk = avg_styles[n]
            phase_avg_deg = np.degrees(metrics[f"phase_avg_{n}_diff"])
            ax.plot(cues, phase_avg_deg,
                    linestyle=ls, marker=mk, color=avg_colors[n],
                    linewidth=1.5, label=f"avg first {n} peaks")
    ax.legend()
    ax.set_xticks(cues)

    plt.tight_layout()
    plt.show()

def plot_psth_per_angle(res, pop, cues,
                        
                         xlim_peak=(0, 20), bin_size=0.5,
                         center_cf=None, bw_neurons=None,
                         show_onset_only=False,
                         psth_filtering=False, stim_fs=None,
                         smooth_cutoff_hz=None, prominence=20.0):
    n_angles = len(cues)
    n_cols   = 1
    n_rows   = int(np.ceil(n_angles / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(7 * n_cols, 4 * n_rows),
                              sharex=True, sharey=False)
    axes_flat = axes.flatten()

    psth_kwargs = dict(
        xlim_peak        = xlim_peak,
        bin_size         = bin_size,
        center_cf        = center_cf,
        bw_neurons       = bw_neurons,
        psth_filtering   = psth_filtering,
        stim_fs          = stim_fs,
        smooth_cutoff_hz = smooth_cutoff_hz,
    )

    side_colors = {'L': 'm', 'R': 'g'}

    for ax_idx, cue in enumerate(cues):
        ax = axes_flat[ax_idx]

        for side, color in side_colors.items():
            spikes = res["cue_to_rate"][cue][side][pop]
            rates, centres = _get_psth_rates(spikes, **psth_kwargs)

            ax.plot(centres, rates, color=color, alpha=0.8,
                    linewidth=1.5, label=side)

            if show_onset_only:
                t_on, fr_on = _onset_peak(rates, centres, bin_size, prominence)
                if not np.isnan(t_on):
                    ax.plot(t_on, fr_on, 'o', color='red',
                            markerfacecolor='none', markeredgewidth=2, zorder=5)
            else:
                t_peaks, fr_peaks = _get_sorted_peaks(rates, centres, bin_size,
                                                       prominence=prominence)
                if len(t_peaks):
                    ax.plot(t_peaks, fr_peaks, 'o', color='red',
                            markerfacecolor='none', markeredgewidth=2, zorder=5)

                # overplot onset peak in a distinct colour on top
                t_on, fr_on = _onset_peak(rates, centres, bin_size, prominence)
                if not np.isnan(t_on):
                    ax.plot(t_on, fr_on, 'o', color='blue',
                            markerfacecolor='none', markeredgewidth=2.5, zorder=6)

        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("FR [Hz]")
        ax.set_title(f"{cue}°")
        ax.legend()
        ax.grid(True, alpha=0.3)

    for ax_idx in range(n_angles, len(axes_flat)):
        axes_flat[ax_idx].set_visible(False)

    plt.tight_layout()
    plt.show()

#### for paper:
def draw_single_neuron_raster_xpaper(
    data_list,
    pop,
    side,
    cue,
    target_cf_hz=None,
    center_cf=None,
    bw_neurons=0,
    xlim=None,
    dot_size=1,
    color=None,
    labels=None,
    figsize=(10, 4),
):
    if not data_list:
        raise ValueError("data_list is empty")

    if color is None:
        color = 'm' if side == 'L' else 'g'

    n_reps = len(data_list)
    if labels is None:
        labels = [range(n_reps)]

    # ------------------------------------------------------------------
    # Resolve target neuron index from the first dataset
    # ------------------------------------------------------------------
    _ref       = data_list[0]
    _ctr       = _ref["cue_to_rate"]
    _spikes0   = _ctr[cue][side][pop]
    _gids      = _spikes0["global_ids"]
    _n         = len(_gids)
    _cf_arr    = greenwood_cf_array(CFMIN / b2.Hz, CFMAX / b2.Hz, _n) / b2.Hz

    if target_cf_hz is not None:
        _, neuron_idx = take_closest(_cf_arr, target_cf_hz)
    elif center_cf is not None:
        _, center_idx = take_closest(_cf_arr, center_cf)
        neuron_idx    = int(np.clip(center_idx + bw_neurons, 0, _n - 1))
    else:
        raise ValueError("Provide either target_cf_hz or center_cf.")

    neuron_gid = int(_gids[0]) + neuron_idx
    neuron_cf  = float(_cf_arr[neuron_idx])

    print(
        f"[draw_single_neuron_raster] pop={pop} side={side} "
        f"→ neuron idx={neuron_idx}, GID={neuron_gid}, CF={neuron_cf:.1f} Hz"
    )

    cues = sorted(_ctr.keys())

    if xlim is None:
        _default_dur = (
            _ref["basesound"].sound.duration / b2.ms
            if "basesound" in _ref
            else _ref["sounds"]["base_sound"].sound.duration / b2.ms
        )
        _dur = _ref.get("simulation_time", _default_dur)
        xlim = [0.0, float(_dur)]

    # ------------------------------------------------------------------
    # Layout: raster only
    # ------------------------------------------------------------------
    fig, ax_raster = plt.subplots(figsize=figsize)

    # ------------------------------------------------------------------
    # Raster
    # ------------------------------------------------------------------
    for rep_idx, d in enumerate(data_list):
        spikes  = d["cue_to_rate"][cue][side][pop]
        times   = spikes["times"]
        senders = spikes["senders"]
        mask    = (
            (senders == neuron_gid) &
            (times   >= xlim[0])   &
            (times   <= xlim[1])
        )
        rep_times = times[mask]

        y_vals = np.full(len(rep_times), rep_idx)
        ax_raster.plot(
            rep_times, y_vals, '.',
            color=color,
            markersize=dot_size * 4,
            markeredgewidth=dot_size * 0.6,
        )

    ax_raster.set_xlim(xlim)
    ax_raster.set_ylim(-0.5, n_reps - 0.5)
    ax_raster.invert_yaxis()
    ax_raster.set_yticks(range(n_reps))
    ax_raster.set_yticklabels([])
    ax_raster.set_xlabel("Time [ms]")
    ax_raster.spines["top"].set_visible(False)
    ax_raster.spines["right"].set_visible(False)

    plt.tight_layout()
    return fig, ax_raster

def draw_single_neuron_raster_by_cue_xpaper(
    data,
    pop,
    side,
    target_cf_hz=None,
    center_cf=None,
    bw_neurons=0,
    xlim=None,
    ylim=None,
    dot_size=2,
    color=None,
    cues_to_plot=None,
    title=None,
    figsize=(10, 5),
):
    """
    Raster plot for a SINGLE result (one seed/recording): each row on the
    y-axis is a different cue (e.g. azimuth/ITD/ILD), for one selected
    neuron — as opposed to draw_single_neuron_raster, where rows are
    repetitions of the same cue.
    """
    if color is None:
        color = 'm' if side == 'L' else 'g'

    _ctr = data["cue_to_rate"]
    all_cues = sorted(_ctr.keys())
    cues = cues_to_plot if cues_to_plot is not None else all_cues
    n_cues = len(cues)

    # ---- Resolve target neuron (same logic, using first cue as reference) ----
    _ref_cue = cues[0]
    _spikes0 = _ctr[_ref_cue][side][pop]
    _gids    = _spikes0["global_ids"]
    _n       = len(_gids)
    _cf_arr  = greenwood_cf_array(CFMIN / b2.Hz, CFMAX / b2.Hz, _n) / b2.Hz

    if target_cf_hz is not None:
        _, neuron_idx = take_closest(_cf_arr, target_cf_hz)
    elif center_cf is not None:
        _, center_idx = take_closest(_cf_arr, center_cf)
        neuron_idx = int(np.clip(center_idx + bw_neurons, 0, _n - 1))
    else:
        raise ValueError("Provide either target_cf_hz or center_cf.")

    neuron_gid = int(_gids[0]) + neuron_idx
    neuron_cf  = float(_cf_arr[neuron_idx])

    print(
        f"[draw_single_neuron_raster_by_cue] pop={pop} side={side} "
        f"→ neuron idx={neuron_idx}, GID={neuron_gid}, CF={neuron_cf:.1f} Hz"
    )

    if xlim is None:
        _default_dur = (
            data["basesound"].sound.duration / b2.ms
            if "basesound" in data
            else data["sounds"]["base_sound"].sound.duration / b2.ms
        )
        _dur = data.get("simulation_time", _default_dur)
        xlim = [0.0, float(_dur)]

    # ------------------------------------------------------------------
    # Raster only
    # ------------------------------------------------------------------
    fig, ax_raster = plt.subplots(figsize=figsize)

    for row_idx, cue in enumerate(cues):
        spikes  = _ctr[cue][side][pop]
        times   = spikes["times"]
        senders = spikes["senders"]
        mask = (
            (senders == neuron_gid) &
            (times   >= xlim[0]) &
            (times   <= xlim[1])
        )
        cue_times = times[mask]

        y_vals = np.full(len(cue_times), row_idx)
        ax_raster.plot(
            cue_times, y_vals, '.',
            color=color,
            markersize=dot_size * 4,
        )

    ax_raster.set_xlim(xlim)
    ax_raster.set_ylim(ylim if ylim is not None else (-0.5, n_cues - 0.5))
    ax_raster.set_yticks(range(n_cues))
    ax_raster.set_yticklabels([f"{int(round(c))}" for c in cues])

    ax_raster.set_xlabel("Time [ms]")
    ax_raster.set_ylabel("Cue")

    ax_raster.spines["top"].set_visible(False)
    ax_raster.spines["right"].set_visible(False)

    plt.tight_layout()
    return fig, ax_raster

def plot_ild50_fit_xpaper(
    metrics,
    ipsi_level,
    ax=None,
    figsize=(6, 4.5),
    data_color='k',
    fit_color='C3',
    show_sem=True,
    sem=None,
    title=None,
    xlabel="ILD (dB)",
    ylabel="Firing Rate (%)",
    n_fit_points=200,
):
    """
    Plot the averaged data points (from extract_ild50_metrics) together with
    the fitted sigmoid, expressed as ILD (contra - ipsi level), marking
    ILD50 with a horizontal line from the y-axis to x=0, and dropping
    vertical reference lines from ILD50 and from (0, 50) down to the x-axis.

    Parameters
    ----------
    metrics : dict
        Output of extract_ild50_metrics.
    ipsi_level : float
        Fixed ipsilateral sound level (dB), constant across all cues.
        Used to convert contralateral levels (and ILD50) into ILD.
    ax : matplotlib Axes, optional
        Existing axes to draw on. Creates new figure if None.
    show_sem : bool
        If True and `sem` is provided, draws error bars on the data points.
    sem : array-like, optional
        Standard error per cue (e.g. curve["sem"] from get_avg_rate_vs_cue).
        Only meaningful if metrics was NOT normalized (normalize=False),
        since sem here is in raw Hz, not on the normalized scale.
    """
    # convert contralateral cues -> ILD (contra - ipsi)
    ild = ipsi_level - metrics["cues"]
    y = metrics["rate"]
    fit_fn = metrics["fit_curve_fn"]

    # rescale y to percentage if not already normalized 0-100
    if metrics["normalized"]:
        y_pct = y
        def fit_fn_pct(x_contra):
            return fit_fn(x_contra)
    else:
        rate_range = metrics["max_rate"] - metrics["min_rate"]
        y_pct = 100 * (y - metrics["min_rate"]) / rate_range
        def fit_fn_pct(x_contra):
            return 100 * (fit_fn(x_contra) - metrics["min_rate"]) / rate_range

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # raw / normalized data points, plotted against ILD
    sem_pct = None
    if show_sem and sem is not None and not metrics["normalized"]:
        sem_pct = 100 * np.asarray(sem) / (metrics["max_rate"] - metrics["min_rate"])
        ax.errorbar(
            ild, y_pct, yerr=sem_pct, fmt='o', color=data_color,
            capsize=3, label='data (mean ± SEM)', zorder=3,
        )
    else:
        ax.plot(ild, y_pct, 'o', color=data_color, label='data', zorder=3)

    # smooth fitted sigmoid, evaluated in original contra-level space
    x_fit_contra = np.linspace(metrics["cues"].min(), metrics["cues"].max(), n_fit_points)
    x_fit_ild = ipsi_level - x_fit_contra
    y_fit_pct = fit_fn_pct(x_fit_contra)
    ax.plot(x_fit_ild, y_fit_pct, '-', color=fit_color, lw=2, label='sigmoid fit', zorder=2)

    # ILD50: convert from contra level to ILD
    ild50 = ipsi_level - metrics["ild50"]
    y_at_ild50 = 50

    # determine y-limits so error bars (if any) are fully visible
    y_lo, y_hi = 0.0, 100.0
    if sem_pct is not None:
        y_lo = min(y_lo, np.min(y_pct - sem_pct))
        y_hi = max(y_hi, np.max(y_pct + sem_pct))
        pad = 0.05 * (y_hi - y_lo)
        y_lo -= pad
        y_hi += pad

    ax.set_ylim(y_lo, y_hi)
    # keep ticks at the standard 0-100 steps regardless of padded limits
    ax.set_yticks(range(0, 125, 25))

    ax.invert_xaxis()
    xmin, xmax = ax.get_xlim()
  

    ax.plot(ild50, y_at_ild50, '*', color='b', ms=10, zorder=4,
            label=f'ILD_50 = {abs(ild50):.1f} dB')

    # horizontal line: from y-axis (xmin) to x = 0, at y = 50
    ax.plot([xmin, ild50], [y_at_ild50, y_at_ild50],
            color=fit_color, ls='--', lw=1, alpha=0.7)

    # vertical line: from ILD50 down to bottom of axes (y_lo)
    ax.plot([ild50, ild50], [y_lo, y_at_ild50],
            color=fit_color, ls='--', lw=1, alpha=0.7)


    ax.set_xlim(xmin, xmax)
    ax.set_ylim(y_lo, y_hi)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or f"R² = {metrics['r_squared']:.3f}")
    ax.legend(loc='best', fontsize=8)

    return ax