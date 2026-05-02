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
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
simulate_repo = PROJECT_ROOT + '/simulate'
sys.path.insert(0, simulate_repo)
from utils.anf_utils import create_sound_key

plt.rcParams["axes.grid"] = True
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.titleweight']= 'bold'
plt.rcParams['axes.spines.top']= False
plt.rcParams['axes.spines.right']= False
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 12   # Size of x-axis tick labels
plt.rcParams['ytick.labelsize'] = 12   # Size of y-axis tick labels
plt.rcParams['legend.fontsize'] = 10   # Size of the legend text
# Make axis labels bold
plt.rcParams['axes.labelweight'] = 'bold'  # Makes x and y axis labels bold

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

def firing_neurons_distribution(x):
    "returns {neuron_id: num_spikes}.keys()"
    n2s = {id: 0 for id in x["global_ids"]}
    for sender in x["senders"]:
        n2s[sender] += 1
    return n2s.values()

def shift_senders(x, hist_logscale=False):
    "returns list of 'senders' with ids shifted to [0,num_neurons]. optionally ids are CFs"
    if hist_logscale:
        cf = greenwood_cf_array(CFMIN/ b2.Hz, CFMAX/ b2.Hz, len(x["global_ids"])) 
        old2newid = {oldid: cf[i] for i, oldid in enumerate(x["global_ids"])}
    else:
        old2newid = {oldid: i for i, oldid in enumerate(x["global_ids"])}
    return [old2newid[i] for i in x["senders"]]

def draw_hist(
    ax,
    senders_renamed,
    cues,
    num_neurons,
    max_spikes_single_neuron,
    logscale=True,
    freq=None,
):
    """draws a low opacity horizontal histogram for each cue position

    includes a secondary y-axis, optionally logarithmic.
    if logscale, expects senders to be renamed to CFs
    if freq, include a horizontal line at corresponding frequency
    """
    max_histogram_height = 0.25
    bin_count = 50
    alpha = 0.5
    freqlinestyle = {
        "color": "black",
        "linestyle": ":",
        "label": "freq_in",
        "alpha": 0.2,
    }
    if logscale:
        bins = greenwood_cf_array(CFMIN/ b2.Hz, CFMAX/ b2.Hz, bin_count) / b2.Hz

        for j, cue in enumerate(cues):
            left_data = senders_renamed["L"][j]
            right_data = senders_renamed["R"][j]

            left_hist, _ = np.histogram(left_data, bins=bins)
            right_hist, _ = np.histogram(right_data, bins=bins)
            max_value = max(max(left_hist), max(right_hist))
            left_hist_normalized = left_hist / (max_value * max_histogram_height)
            right_hist_normalized = right_hist / (max_value * max_histogram_height)

            ax.barh(
                bins[:-1],
                -left_hist_normalized,
                height=np.diff(bins),  # bins have different sizes
                left=cue,
                color="m",
                alpha=alpha,
                align="edge",
            )
            ax.barh(
                bins[:-1],
                right_hist_normalized,
                height=np.diff(bins),
                left=cue,
                color="g",
                alpha=alpha,
                align="edge",
            )
        ax.set_yscale("log")
        ax.set_ylim(CFMIN, CFMAX)
        yticks = [125, 1000, 10000, 20000]
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{freq} Hz" for freq in yticks])
        if freq is not None:
            ax.axhline(y=freq / b2.Hz, **freqlinestyle)
        ax.set_ylabel("CF [Hz]")
    else:
        bins = np.linspace(0, num_neurons, bin_count)

        for j, cue in enumerate(cues):
            left_data = senders_renamed["L"][j]
            right_data = senders_renamed["R"][j]

            left_hist, _ = np.histogram(left_data, bins=bins)
            right_hist, _ = np.histogram(right_data, bins=bins)
            left_hist_normalized = (
                left_hist / max_spikes_single_neuron * max_histogram_height
            )
            right_hist_normalized = (
                right_hist / max_spikes_single_neuron * max_histogram_height
            )

            ax.barh(
                bins[:-1],
                -left_hist_normalized,
                height=num_neurons / bin_count,
                left=cue,
                color="C0",
                alpha=alpha,
                align="edge",
            )
            ax.barh(
                bins[:-1],
                right_hist_normalized,
                height=num_neurons / bin_count,
                left=cue,
                color="C1",
                alpha=alpha,
                align="edge",
            )
        ax.set_ylabel("neuron id")
        ax.set_ylim(0, num_neurons)

        if freq is not None:
            cf = greenwood_cf_array(CFMIN/ b2.Hz, CFMAX/ b2.Hz, num_neurons) * b2.Hz
            freq, neur_n = take_closest(cf, freq)
            ax.axhline(y=neur_n)
    ax.yaxis.set_minor_locator(plt.NullLocator())  # remove minor ticks

def draw_single_cue_histogram(data, cue, population="SBC", fontsize=16, alpha=0.8):
    """
    Draw horizontal histograms of spike distributions across frequencies for a single cue,
    with left population growing downward and right population growing upward from a central axis.

    Parameters:
    -----------
    data : dict
        The full dataset containing cue_to_rate information
    cue : float
        The specific cue to visualize
    population : str
        Name of the neural population to visualize
    fontsize : int
        Base fontsize for the plot. Other elements will scale relative to this.

    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure
    """
    # Constants
    bin_count = 50

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 2.42))

    # Get data for this cue and population
    pop_data = {
        "L": data["cue_to_rate"][cue]["L"][population],
        "R": data["cue_to_rate"][cue]["R"][population],
    }

    # Create logarithmic bins for frequency
    bins = greenwood_cf_array(CFMIN/ b2.Hz, CFMAX/ b2.Hz, bin_count) / b2.Hz

    # Process data for histograms
    senders_renamed = {
        side: shift_senders(pop_data[side], True)  # True for logscale
        for side in ["L", "R"]
    }

    # Create histograms
    left_hist, _ = np.histogram(senders_renamed["L"], bins=bins)
    right_hist, _ = np.histogram(senders_renamed["R"], bins=bins)

    # Normalize histograms
    max_value = max(max(left_hist), max(right_hist))
    if max_value > 0:  # Avoid division by zero
        left_hist = left_hist / max_value
        right_hist = right_hist / max_value
    print(bins[0])
    # Plot histograms - note the negative values for left histogram
    ax.bar(
        bins[:-1],
        -left_hist,
        width=np.diff(bins),
        color="C0",
        alpha=alpha,
        label="Left",
        align="edge",
    )
    ax.bar(
        bins[:-1],
        right_hist,
        width=np.diff(bins),
        color="C1",
        alpha=alpha,
        label="Right",
        align="edge",
    )

    # Configure axes
    ax.set_xscale("log")
    ax.set_xlim(CFMIN, CFMAX)
    xticks = [20, 100, 500, 1000, 5000, 20000]
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{freq} Hz" for freq in xticks], fontsize=fontsize)

    # Set y-axis limits symmetrically around zero
    ylim = 1.1  # Slightly larger than 1 to give some padding
    ax.set_ylim(-ylim, ylim)
    ax.tick_params(axis="y", labelsize=fontsize)  # Set y-tick font size

    # Add horizontal line at y=0
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

    # Set font sizes for labels and title
    ax.set_xlabel("Characteristic Frequency (Hz)", fontsize=fontsize)
    ax.set_ylabel("Normalized spikes", fontsize=fontsize)
    # ax.legend(fontsize=fontsize)

    # plt.title(
    #     f"{population} population response at {cue}° azimuth\n"
    #     f'Sound: {data["conf"]["sound_key"]}',
    #     fontsize=fontsize * 1.2,
    # )  # Title slightly larger

    plt.tight_layout()

    return fig

def synthetic_cue_to_itd(cue, w_head: int = 22, v_sound: int = 33000):
    delta_x = w_head * np.sin(np.deg2rad(cue))
    return round(1000 * delta_x / v_sound, 2)

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

def draw_spikes_pop(
    res,
    cue,
    side,
    pop,
    y_ax = 'cf_custom',
    f_ticks = [125,1000,10000],
    title=None,
    plot_sound = False,
    xlim=None,
    ylim=None,
    color = None,
    figsize = (7,4)
):
    spikes = res["cue_to_rate"][cue][side][pop]  
    num_neurons = len(spikes["global_ids"])
    cf = greenwood_cf_array(CFMIN/ b2.Hz, CFMAX/ b2.Hz, num_neurons)*b2.Hz
    neuron_to_cf = {global_id: freq for global_id, freq in zip(spikes["global_ids"], cf)}
    duration = res.get("simulation_time", res["basesound"].sound.duration / b2.ms)

    if color == None:
        if side == 'L': color = 'm'
        elif side == 'R': color = 'g'

    if xlim == None: 
        xlim_array = [0,duration]
    else: xlim_array = xlim

    if y_ax == 'global_ids':
        y_values = spikes['senders']
        ylabel = f"{pop} global IDS"
    elif y_ax == 'cf':
        y_values = np.array([neuron_to_cf[sender] for sender in spikes["senders"]])
        ylabel = f"{pop} CF [Hz]"
    elif y_ax == 'ids':
        y_values = spikes['senders'] - spikes['global_ids'][0]
        ylabel = f"{pop} IDS"
    elif y_ax == 'cf_custom':
        y_values = spikes['senders'] - spikes['global_ids'][0]
        ylabel = f"{pop} CF [Hz]"
        label_indexes = np.zeros_like(f_ticks)
        for i, f in enumerate(f_ticks):
            _, label_indexes[i] = take_closest(cf, f*Hz)
    else:
        raise ValueError("Invalid value for 'ax'. Choose 'cf_custom', 'ids', 'cf', or 'global_ids'.")

    if(plot_sound):
        # Create figure with gridspec to control subplot sizes
        fig = plt.figure(figsize=(figsize[0], figsize[1]+1))
        # Create smaller subplot for sound (25% of height) and keep full width for spikes plot
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 7])
        
        # Sound plot - smaller and without axes/grid
        ax0 = fig.add_subplot(gs[0])
        x_sound = create_xax_time_sound(res)
        if len(x_sound) != len(res['basesound'].sound):
            x_sound = x_sound[0:len(res['basesound'].sound)]
        ax0.plot(x_sound, res['basesound'].sound, 'k', linewidth = '1')
        ax0.set_xlim(xlim_array)
        # Remove axes and grid
        ax0.axis('off')
        
        # Spikes plot - same size as when plot_sound is False
        ax1 = fig.add_subplot(gs[1])
        if ylim != None:
            if y_ax == 'cf':
                ax1.set_ylim(ylim)
            else:
                _, y0 = take_closest(cf, ylim[0]*Hz)
                _, y1 = take_closest(cf, ylim[1]*Hz)
                ax1.set_ylim([y0,y1])
        else:
            ax1.set_yticks(label_indexes)
            ax1.set_yticklabels(f_ticks*Hz) 
        
        ax1.plot(spikes['times'], y_values, '.', color=color, markersize=1)
        ax1.set_ylabel(ylabel)
        ax1.set_xlabel("Time [ms]")
        ax1.set_xlim(xlim_array)
        
        # Adjust layout to prevent overlapping
        plt.tight_layout()
    else:
        fig, ax = plt.subplots(1, figsize=figsize)
        if ylim != None:
            if y_ax == 'cf':
                ax.set_ylim(ylim)
            else:
                _, y0 = take_closest(cf, ylim[0]*Hz)
                _, y1 = take_closest(cf, ylim[1]*Hz)
                ax.set_ylim([y0,y1])
        else:
            ax.set_yticks(label_indexes)
            ax.set_yticklabels(f_ticks*Hz) 
        
        ax.plot(spikes['times'], y_values, '.', color=color, markersize=1)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Time [ms]")
        ax.set_xlim(xlim_array)
    
    plt.show()
    return

def draw_spikes_pop_bothside(
    res,
    cue,
    pop,
    y_ax = 'cf_custom',
    f_ticks = [100,1000,10000],
    title=None,
    plot_sound = False,
    xlim=None,
    ylim=None,
    color = None,
    figsize = (7,8)
):
    fig, ax = plt.subplots(2, figsize = figsize, sharex = True)
    for i, side in enumerate(['L', 'R']):
        spikes = res["cue_to_rate"][cue][side][pop]  
        num_neurons = len(spikes["global_ids"])
        cf = greenwood_cf_array(CFMIN/ b2.Hz, CFMAX/ b2.Hz, num_neurons)*b2.Hz
        neuron_to_cf = {global_id: freq for global_id, freq in zip(spikes["global_ids"], cf)}
        duration = res.get("simulation_time", res["basesound"].sound.duration / b2.ms)

        if xlim == None: 
            xlim_array = [0,duration]
        else: xlim_array = xlim
            
        if ylim is None:
            ylim = [CFMIN/Hz, CFMAX/Hz]

        if side == 'L': color = 'm'
        elif side == 'R': color = 'g'

        if y_ax == 'global_ids':
            y_values = spikes['senders']
            ylabel = f"{pop} global IDS"
        elif y_ax == 'cf':
            y_values = np.array([neuron_to_cf[sender] for sender in spikes["senders"]])
            ylabel = f"{pop} CF [Hz]"
        elif y_ax == 'ids':
            y_values = spikes['senders'] - spikes['global_ids'][0]
            ylabel = f"{pop} IDS"
        elif y_ax == 'cf_custom':
            y_values = spikes['senders'] - spikes['global_ids'][0]
            ylabel = f"{pop} CF [Hz]"
            label_indexes = np.zeros_like(f_ticks)
            for i, f in enumerate(f_ticks):
                _, label_indexes[i] = take_closest(cf, f*Hz)
        else:
            raise ValueError("Invalid value for 'ax'. Choose 'cf_custom', 'ids', 'cf', or 'global_ids'.")
  
        if ylim != None:
            if y_ax == 'cf':
                ax[i].set_ylim(ylim)
            else:
                _, y0 = take_closest(cf, ylim[0]*Hz)
                _, y1 = take_closest(cf, ylim[1]*Hz)
                ax[i].set_ylim([y0,y1])
        else:
            ax[i].set_yticks(label_indexes)
            ax[i].set_yticklabels(f_ticks*Hz) 
        
        ax[i].plot(spikes['times'], y_values, '.', color=color, markersize=1)
        ax[i].set_ylabel(ylabel)
        ax[i].set_xlim(xlim_array)
    ax[1].set_xlabel("Time [ms]")
    plt.show()
    return

def draw_psth_pop_bothside(
    res,
    cue,
    pop,
    title=None,
    xlim=None,
    ylim=None,
    bin_size = 1, #ms
    color = None,
    figsize = (7,4)
):
    fig, ax = plt.subplots(1, figsize = figsize)
    for color, side in zip(['m', 'g'],['L', 'R']):
        spikes = res["cue_to_rate"][cue][side][pop] 
        spike_times = spikes['times']
        spike_senders = spikes['senders']
        num_neurons = len(spikes["global_ids"])
        cf = greenwood_cf_array(CFMIN/ b2.Hz, CFMAX/ b2.Hz, num_neurons)*b2.Hz
        duration = res.get("simulation_time", res["basesound"].sound.duration / b2.ms)

        if xlim == None: 
            xlim_array = [0,duration]
        else: xlim_array = xlim
        
        if ylim is None:
            ylim = [CFMIN/Hz, CFMAX/Hz]

        time_mask = (spike_times >= xlim_array[0]) & (spike_times <= xlim_array[1])
        filtered_times = spike_times[time_mask]
        filtered_senders = spike_senders[time_mask]
        # Fix for the error - get indices properly
        _, ymin_idx = take_closest(cf, ylim[0]*Hz)
        _, ymax_idx = take_closest(cf, ylim[1]*Hz)
    
        # Convert indices to actual IDs
        ymin = spikes["global_ids"][0] + ymin_idx
        ymax = spikes["global_ids"][0] + ymax_idx
        cluster_mask = (filtered_senders >= ymin) & (filtered_senders <= ymax)
        cluster_times = filtered_times[cluster_mask]
        bins = np.arange(xlim_array[0], xlim_array[1] + bin_size, bin_size)
        ax.hist(cluster_times, bins=bins, alpha=0.7, color = color)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Spike Count')
        ax.grid(True, alpha=0.3)

    # You can add more customization to the axes if needed
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()
    return

def draw_psth_pop(
    res,
    cue,
    side,
    pop,
    title=None,
    xlim=None,
    ylim=None,        # CF interval in Hz
    bin_size=1,       # ms
    color=None,
    psth_rate=False,
    figsize=(7, 4)
):
    import matplotlib.pyplot as plt

    side_colors = {'L': 'b', 'R': 'g'}
    color = color if color is not None else side_colors.get(side, 'b')

    duration = res.get("simulation_time", res["sounds"]["base_sound"].sound.duration / b2.ms)
    if xlim is None:
        xlim = [0, duration]
    if ylim is None:
        ylim = [CFMIN / Hz, CFMAX / Hz]  # entire population by default

    spikes = res["cue_to_rate"][cue][side][pop]

    # -----------------------------
    # Helper: filter spikes in time & CF interval
    # -----------------------------
    def filter_spikes(spikes, xlim, ylim):
        times = spikes["times"]
        senders = spikes["senders"]
        gids = spikes["global_ids"]
        n_neurons = len(gids)

        # Time filter
        mask_t = (times >= xlim[0]) & (times <= xlim[1])
        times_t = times[mask_t]
        senders_t = senders[mask_t]

        # Map CF interval to neuron indices
        cf_hz = greenwood_cf_array(CFMIN / b2.Hz, CFMAX / b2.Hz, n_neurons) / b2.Hz
        _, ymin_idx = take_closest(cf_hz, ylim[0])
        _, ymax_idx = take_closest(cf_hz, ylim[1])

        cf_min_id = gids[0] + ymin_idx
        cf_max_id = gids[0] + ymax_idx

        # CF filter
        mask_cf = (senders_t >= cf_min_id) & (senders_t <= cf_max_id)

        return times_t[mask_cf], senders_t[mask_cf]

    times_f, senders_f = filter_spikes(spikes, xlim, ylim)

    bins = np.arange(xlim[0], xlim[1] + bin_size, bin_size)

    fig, ax = plt.subplots(1, figsize=figsize)

    if psth_rate:
        counts, _ = np.histogram(times_f, bins=bins)
        rates = counts / bin_size * 1000  # convert to Hz
        ax.plot(bins[:-1], rates, color=color, alpha=0.7)
        ax.set_ylabel("Firing rate [Hz]")
    else:
        ax.hist(times_f, bins=bins, alpha=0.7, color=color)
        ax.set_ylabel("Spike count")

    ax.set_xlabel("Time [ms]")
    ax.set_xlim(xlim)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if title:
        ax.set_title(title)
    ax.set_yticks([])
    plt.show()

def calculate_firing_rates(cue_to_rate, pop, sides, cues, duration, cf_interval=None):
    """
    Calculate firing rates for different sides and cues.
    
    Parameters:
    - cue_to_rate: Dictionary mapping cues to rate data
    - pop: Population name (e.g., 'LSO')
    - sides: List of sides ('L', 'R', or both)
    - cues: List of cues to process
    - duration: Duration of the simulation
    - cf_interval: Optional frequency interval for filtering
    
    Returns:
    - tot_spikes: Dictionary of total spike rates by side
    - active_neuron_rate: Dictionary of active neuron rates by side
    """
    
    # Get frequency space
    num_neurons = len(cue_to_rate[0]['L'][pop]["global_ids"])
    cf = greenwood_cf_array(CFMIN/ b2.Hz, CFMAX/ b2.Hz, num_neurons)/b2.Hz
    
    if cf_interval is None:
        # Simple case - use all neurons
        tot_spikes = {
            side: [
                len(cue_to_rate[cue][side][pop]["times"]) / duration
                for cue in cues
            ]
            for side in sides
        }
        avg_neuron_rate = {
            side: [
                len(cue_to_rate[cue][side][pop]["times"]) / (duration*num_neurons)
                for cue in cues
            ]
            for side in sides
        }

        active_neuron_rate = {
            side: [
                avg_fire_rate_actv_neurons(cue_to_rate[cue][side][pop])
                * (1 * b2.second)
                / duration
                for cue in cues
            ]
            for side in sides
        }
        return tot_spikes, avg_neuron_rate, active_neuron_rate
    
    # Case with CF interval filtering
    tot_spikes = {}
    avg_neuron_rate = {}
    cluster_numerosity = {}
    
    
    for side in sides:
        tot_spikes[side] = []
        avg_neuron_rate[side] = []
        
        for cue in cues:
            # Find indices in CF array corresponding to interval bounds
            _, ymin_idx = take_closest(cf, cf_interval[0])
            _, ymax_idx = take_closest(cf, cf_interval[1])

            # Calculate actual neuron IDs from global_ids and indices
            base_id = cue_to_rate[cue][side][pop]["global_ids"][0]
            ymin = base_id + ymin_idx
            ymax = base_id + ymax_idx

            # Filter spikes within the specified range
            cluster_mask = (cue_to_rate[cue][side][pop]['senders'] >= ymin) & (cue_to_rate[cue][side][pop]['senders'] <= ymax)
            cluster_times = cue_to_rate[cue][side][pop]['times'][cluster_mask]

            # Calculate rate for this cue and side
            tot_spikes[side].append(len(cluster_times) / duration)
            avg_neuron_rate[side].append(len(cluster_times)/((ymax - ymin)*duration))

            # Compute cluster numerosity (unique senders)
            unique_senders = np.unique(cue_to_rate[cue][side][pop]['senders'][cluster_mask])
        
        cluster_numerosity[side] = len(unique_senders)
    
    return tot_spikes, avg_neuron_rate, avg_neuron_rate #unique set not computed for a cf_interval

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

def add_rate_annotations(ax, original_values, normalized_rates, cues, sides, side_colors):
    """
    Add annotations showing original rate values at min/max points.
    
    Parameters:
    - ax: Matplotlib axis to plot on
    - original_values: Dictionary with original min/max values and indices
    - normalized_rates: Dictionary of normalized rates by side
    - cues: List of cues
    - sides: List of sides
    - side_colors: Dictionary mapping sides to colors
    """
    # Get current axis limits to avoid placing annotations outside visible area
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    y_margin = (ymax - ymin) * 0.05  # 5% margin from top/bottom
    
    # Track used positions to avoid overlaps
    used_positions = {}
    
    for side_idx, side in enumerate(sides):
        min_value = original_values[side]['min_value']/Hz
        max_value = original_values[side]['max_value']/Hz
        min_cue_idx = original_values[side]['min_cue_idx']
        max_cue_idx = original_values[side]['max_cue_idx']
        
        # Format values as Hz or kHz
        if max_value < 1000:
            max_text = f"{max_value:.2f} Hz"
        else:
            max_text = f"{max_value/1000:.2f} kHz"
            
        if min_value < 1000:
            min_text = f"{min_value:.2f} Hz"
        else:
            min_text = f"{min_value/1000:.2f} kHz"
        
        # Calculate positions to avoid overlapping
        min_x = cues[min_cue_idx]
        min_y = normalized_rates[side][min_cue_idx]
        max_x = cues[max_cue_idx]
        max_y = normalized_rates[side][max_cue_idx]
        
        # Offset for max annotation - try to position near the top of the plot
        # but avoid overlapping with other annotations
        max_y_pos = max_y
        max_offset_y = 15 + (side_idx * 15)  # Vertical offset increases with each side
        
        # Check if we're close to another annotation
        position_key = f"{max_x}_{max_y}"
        if position_key in used_positions:
            # Adjust the offset to avoid overlap
            max_offset_y += 15
        used_positions[position_key] = True
        
        # For min annotation, position below the point
        min_y_pos = min_y
        min_offset_y = -15 - (side_idx * 15)  # Vertical offset decreases with each side
        
        # Check if we're close to another annotation
        position_key = f"{min_x}_{min_y}"
        if position_key in used_positions:
            # Adjust the offset to avoid overlap
            min_offset_y -= 15
        used_positions[position_key] = True
        
        # Ensure annotations are within plot boundaries
        if max_y + max_offset_y/72 > ymax - y_margin:
            # If annotation would be outside top, place it below the point instead
            max_offset_y = -15
        
        if min_y + min_offset_y/72 < ymin + y_margin:
            # If annotation would be outside bottom, place it above the point instead
            min_offset_y = 15
            
        # Add annotations (text only, no boxes)
        ax.annotate(
            max_text,
            (max_x, max_y),
            xytext=(10, max_offset_y),
            textcoords='offset points',
            color=side_colors[side],
            fontsize=9,
            fontweight='bold',
            horizontalalignment='left',
            verticalalignment='center'
        )
        
        ax.annotate(
            min_text,
            (min_x, min_y),
            xytext=(10, min_offset_y),
            textcoords='offset points',
            color=side_colors[side],
            fontsize=9,
            fontweight='bold',
            horizontalalignment='left',
            verticalalignment='center'
        )

def plot_cf_intervals_grid(data, intervals, pop='MSO', rate=False):
    # Calculate number of rows and columns for the subplots
    n_intervals = len(intervals)
    n_cols = 2
    n_rows = (n_intervals + n_cols - 1) // n_cols  # Ceiling division
    
    # Create a figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5*n_rows))
    axes = axes.flatten()  # Flatten to easily iterate through all axes
    
    # Loop through intervals and create plots
    for i, interval in enumerate(intervals):
        if i < n_intervals:  # Make sure we don't exceed the number of intervals
            # Create a title for this subplot
            title = f"{interval[0]}-{interval[1]} Hz"
            
            # Plot in the corresponding subplot
            draw_rate_vs_cue_pop(
                data=data,
                pop=pop,
                rate=rate,
                cf_interval=interval,
                ax=axes[i],
                title=title
            )
    
    # Hide any unused subplots
    for i in range(n_intervals, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_tonotopic_heatmaps(
    data,
    pop='LSO',
    num_cells_per_interval=50,
    row_norm = True,
    title=None,
    figsize=(30, 18),
    cmap='viridis',
    diff_cmap='coolwarm',
    norm_max_given=None,
    y_axis='cf',
    f_ticks=None,
    show_sides=True
):
    """
    Generates heatmaps for auditory neural responses across cues and frequency bands.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing simulation data with cue_to_rate information
    pop : str, default='LSO'
        Population name to analyze (e.g., 'LSO', 'MSO')
    num_cells_per_interval : int, default=50
        Number of cells to include in each frequency interval
    title : str, optional
        Overall title for the plot
    figsize : tuple, default=(18, 12)
        Figure size (width, height) in inches
    cmap : str, default='viridis'
        Colormap for the left and right heatmaps
    diff_cmap : str, default='coolwarm'
        Colormap for the difference heatmap (should be diverging)
    norm_max_given : float, optional
        Maximum value for difference colormap normalization
    y_axis : str, default='cf'
        Type of y-axis to display: 'cf' for characteristic frequency or 'cells' for cell numbers
    f_ticks : list, optional
        List of frequency values to display as ticks (if provided, only these ticks will be shown)
    show_sides : bool, default=True
        Whether to show individual left/right heatmaps along with difference map
        
    Returns:
    --------
    fig : matplotlib Figure
        The generated figure with heatmaps
    """
    # Extract required data
    cue_to_rate = data["cue_to_rate"]
    duration = (data.get("simulation_time", data["sounds"]['base_sound'].sound.duration / b2.ms) * b2.ms)
    cues = list(cue_to_rate.keys())
    sides = ["L", "R"]
    
    # Get total number of neurons
    num_neurons = len(cue_to_rate[cues[0]]['L'][pop]["global_ids"])
    
    # Generate full CF array using greenwood
    cf = greenwood_cf_array(CFMIN/ b2.Hz, CFMAX/ b2.Hz, num_neurons)*b2.Hz
    
    # Calculate number of intervals based on total neurons and interval size
    num_intervals = num_neurons // num_cells_per_interval
    if num_neurons % num_cells_per_interval > 0:
        num_intervals += 1  # Add one more interval for remaining cells
    
    # Initialize matrices to store firing rates
    rate_matrices = {side: np.zeros((num_intervals, len(cues))) for side in sides}
    
    # Store characteristic frequencies for each interval
    cf_ids = np.zeros(num_intervals)
    
    # Calculate the base neuron ID for each side
    base_ids = {
        side: cue_to_rate[cues[0]][side][pop]["global_ids"][0]
        for side in sides
    }
    
    # Calculate firing rates for each interval and cue
    for side in sides:
        neuron_to_cf = {global_id: freq for global_id, freq in zip(cue_to_rate[0][side][pop]["global_ids"], cf)}
        for i in range(num_intervals):
            # Define the range of neurons for this interval
            start_idx = i * num_cells_per_interval
            end_idx = min((i + 1) * num_cells_per_interval, num_neurons)
            
            # Calculate actual neuron IDs
            ymin = base_ids[side] + start_idx
            ymax = base_ids[side] + end_idx - 1  # -1 because end_idx is exclusive
            
            # Calculate central neuron ID and its characteristic frequency
            ycentral = int((ymin + ymax)/2)
            cf_ids[i] = neuron_to_cf[ycentral]
            
            for j, cue in enumerate(cues):
                # Filter spikes within the specified range
                cluster_mask = (
                    (cue_to_rate[cue][side][pop]['senders'] >= ymin) & 
                    (cue_to_rate[cue][side][pop]['senders'] <= ymax)
                )
                cluster_times = cue_to_rate[cue][side][pop]['times'][cluster_mask]
                
                # Calculate average firing rate for this interval and cue
                num_cells = end_idx - start_idx
                firing_rate = len(cluster_times) / (duration * num_cells)
                rate_matrices[side][i, j] = firing_rate
    
    # Normalize each row by its maximum value
    if(row_norm):
        for side in sides:
            for i in range(num_intervals):
                max_val = np.max(rate_matrices[side][i, :])
                if max_val > 0:  # Avoid division by zero
                    rate_matrices[side][i, :] = rate_matrices[side][i, :] / max_val
    
    # Calculate difference matrix (L - R)
    diff_matrix = rate_matrices['L'] - rate_matrices['R']
    
    # Create figure based on display mode
    if show_sides:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Generate side heatmaps
        im_left = axes[0].imshow(rate_matrices['L'], cmap=cmap, aspect='auto', interpolation='none')
        im_right = axes[1].imshow(rate_matrices['R'], cmap=cmap, aspect='auto', interpolation='none')
        
        # Add colorbars for side plots
        cbar_left = plt.colorbar(im_left, ax=axes[0])
        cbar_right = plt.colorbar(im_right, ax=axes[1])
        if(row_norm):
            cbar_right.set_label('Normalized Firing Rate')
        else:
            cbar_right.set_label('Firing Rate')
        
        # Set titles for side plots
        axes[0].set_title('Left Side')
        axes[1].set_title('Right Side')
        
        # Set up diff plot (ax=axes[2])
        diff_ax = axes[2]
        diff_ax.set_title('Difference (Left - Right)')
    else:
        fig, diff_ax = plt.subplots(1, figsize=figsize)
    
    # For difference heatmap, use a diverging colormap centered at zero
    norm_max = max(abs(np.min(diff_matrix)), abs(np.max(diff_matrix)))
    if norm_max_given is not None and norm_max_given >= norm_max:
        norm_max = norm_max_given
    norm = Normalize(vmin=-norm_max, vmax=norm_max)
    
    # Create difference heatmap
    im_diff = diff_ax.imshow(diff_matrix, cmap=diff_cmap, aspect='auto', 
                       interpolation='none', norm=norm)
    cbar_diff = plt.colorbar(im_diff, ax=diff_ax)
    cbar_diff.set_label('Difference (L - R)')
    
    # Define function to set up axes formatting
    def setup_axis(ax):
        # Set x-axis labels (cues)
        ax.set_xticks(np.arange(len(cues)))
        ax.set_xticklabels([f"{cue}°" for cue in cues])
        ax.set_xlabel('Azimuth Angle')
        
        # Set y-axis formatting
        if f_ticks is not None and y_axis == 'cf':
            # Use only the specified frequency ticks
            y_positions = []
            y_labels = []
            
            # Add fixed 20 Hz tick at the bottom edge (not centered)
            y_positions.append(num_intervals - 0.5)  # Bottom edge
            y_labels.append("20 kHz")

            # Add fixed 20 kHz tick at the top edge (not centered)
            y_positions.append(-0.5)  # Top edge
            y_labels.append("125 Hz")

            for freq in f_ticks:
                # Format frequency label
                if freq < 1000:
                    label = f"{int(round(freq))} Hz"
                else:
                    label = f"{int(freq/1000)} kHz"
                
                # Find approximate position (closest interval) for this frequency
                distances = np.abs(cf_ids - freq)
                closest_idx = np.argmin(distances)
                
                y_positions.append(closest_idx)
                y_labels.append(label)
            
            ax.set_yticks(y_positions)
            ax.set_yticklabels(y_labels)
        else:
            # Show all intervals
            ax.set_yticks(np.arange(num_intervals))
            
            if y_axis == 'cf':
                # Format frequency labels
                freq_labels = []
                for freq in cf_ids:
                    if freq < 1000:
                        freq_labels.append(f"{int(round(freq))} Hz")
                    else:
                        freq_labels.append(f"{freq/1000:.1f}k Hz")
                
                ax.set_yticklabels(freq_labels)
            else:  # y_axis == 'cells'
                # Show cell number ranges
                cell_labels = []
                for i in range(num_intervals):
                    start_idx = i * num_cells_per_interval
                    end_idx = min((i + 1) * num_cells_per_interval, num_neurons) - 1
                    cell_labels.append(f"{start_idx}-{end_idx}")
                
                ax.set_yticklabels(cell_labels)
        
        ax.invert_yaxis()  # Ensure low frequencies are at the bottom
    
    # Apply axis formatting to all axes
    if show_sides:
        for ax in axes:
            setup_axis(ax)
        axes[0].set_ylabel('Characteristic Frequency' if y_axis == 'cf' else 'Cell Indices')
    else:
        setup_axis(diff_ax)
    
    # Adjust layout
    plt.tight_layout()
    if title:
        plt.subplots_adjust(top=0.9)  # Make room for the overall title
        fig.suptitle(title, fontsize=16)
    
    return

def draw_rate_vs_cue_pop_multi(
    
    data_list,                      # list of datasets
    title=None,
    pop='LSO',
    rate=True,                      # True = avg, False = population
    cf_interval=None,
    sides=None,
    color=None,
    norm=False,                     # False, True, "zscore", "minmax"
    figsize=[7,4],
    ax=None,
    ylim=None,
    label=None,
    error_type="sem"                # "sem" or "std"
):
    # -----------------------------------------------------
    # Setup
    # -----------------------------------------------------
    if sides is None:
        sides = ["L","R"]

    cues = list(data_list[0]["cue_to_rate"].keys())

    # Colors
    if isinstance(color, dict):
        side_colors = color
    elif isinstance(color, str):
        side_colors = {sides[0]: color}
    else:
        side_colors = {"L": "m", "R": "g"}

    # Accumulators
    avg_accum = {s: [] for s in sides}
    pop_accum = {s: [] for s in sides}

    # -----------------------------------------------------
    # Extract rate curves from each dataset
    # -----------------------------------------------------
    for data in data_list:
        cue_to_rate = data["cue_to_rate"]
        duration = data.get(
            "simulation_time",
            data["basesound"].sound.duration / b2.ms
        ) * b2.ms

        tot_spikes, avg_neuron_rate, _ = calculate_firing_rates(
            cue_to_rate, pop, sides, cues, duration, cf_interval
        )

        for s in sides:
            avg_accum[s].append(np.array(avg_neuron_rate[s]))
            pop_accum[s].append(np.array(tot_spikes[s]))

    # -----------------------------------------------------
    # Compute mean + error
    # -----------------------------------------------------
    def compute_mean_and_error(accum):
        arr = np.vstack(accum)
        mean = arr.mean(axis=0)

        if error_type == "std":
            err = arr.std(axis=0)
        else:
            err = arr.std(axis=0) / np.sqrt(arr.shape[0])  # SEM

        return mean, err

    avg_mean, avg_err = {}, {}
    pop_mean, pop_err = {}, {}

    for s in sides:
        avg_mean[s], avg_err[s] = compute_mean_and_error(avg_accum[s])
        pop_mean[s], pop_err[s] = compute_mean_and_error(pop_accum[s])

    # -----------------------------------------------------
    # Choose rate type
    # -----------------------------------------------------
    if rate is True:
        mean_used = avg_mean
        err_used = avg_err
        ylabel_text = "Avg Firing Rate [Hz]"
    else:
        mean_used = pop_mean
        err_used = pop_err
        ylabel_text = "Population Firing Rate [Hz]"

    # -----------------------------------------------------
    # NORMALIZATION OPTIONS
    # norm = False | True | "zscore" | "minmax"
    # -----------------------------------------------------
    original_values = None

    if norm:
        original_values = {s: mean_used[s].copy() for s in sides}

        for s in sides:
            x = mean_used[s]
            err = err_used[s]

            if norm is True:
                # Normalize by maximum
                max_val = x.max() if x.max() != 0 else 1e-9
                mean_used[s] = x / max_val
                err_used[s] = err / max_val

                

            elif norm == "zscore":
                mu = x.mean()
                sigma = x.std() if x.std() > 1e-12 else 1e-12
                mean_used[s] = (x - mu) / sigma
                err_used[s] = err / sigma


            elif norm == "minmax":
                mn = x.min()
                mx = x.max()
                rng = mx - mn if mx != mn else 1e-9
                mean_used[s] = (x - mn) / rng
                err_used[s] = err / rng



            else:
                raise ValueError("norm must be False, True, 'zscore', or 'minmax'")
            
        ylabel_text = "Normalized " + ylabel_text

    # -----------------------------------------------------
    # Plotting
    # -----------------------------------------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    for s in sides:
        pl = label if label else f"{s} (mean)"

        ax.errorbar(
            cues,
            mean_used[s],
            yerr=err_used[s],
            fmt='o-',
            color=side_colors[s],
            ecolor=side_colors[s],
            elinewidth=1.4,
            capsize=4,
            label=pl
        )

    if len(sides) > 1 and label is None:
        ax.legend()

    # Aesthetics
    ax.set_xticks(cues)
    ax.set_xticklabels([f"{a}°" for a in cues])
    ax.set_xlabel("Azimuth Angle")
    ax.set_ylabel(ylabel_text)

    if ylim:
        ax.set_ylim(ylim)
    if title:
        ax.set_title(title)

    plt.tight_layout()
    return ax, original_values if norm else ax

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
    xlim = None,
    ylim = None,
    color = 'b'):
    """
    Plot a Brian2Hears Sound object over time.
    Works for mono or stereo.
    
    sound: brian2hears.Sound
    """
    # Sound → numpy array: shape (samples, channels)
    if isinstance(sound, b2h.Sound):
        snd = sound
    else:
        # assume Tone, ToneBurst, etc.
        snd = sound.sound

    # Convert to numpy (samples, channels)
    wave = np.asarray(snd)
    fs = float(snd.samplerate)

    # time axis
    t = np.arange(wave.shape[0]) / fs
    if time_in_ms:
        t = t * 1000
        xlabel = "Time (ms)"
    else:
        xlabel = "Time (s)"

    fig, ax = plt.subplots(figsize=figsize)

    # mono
    if wave.ndim == 2 and wave.shape[1] == 1:
        ax.plot(t, wave[:, 0], color = color, linewidth=0.8)
    # stereo
    else:
        ax.plot(t, wave[:, 0], linewidth=0.8, label=f"left", color='m')
        ax.plot(t, wave[:, 1], linewidth=0.8, label=f"right", color='g')
        ax.legend()

    ax.set_xlabel(xlabel)
    ax.set_ylim(ylim)
    ax.set_ylabel("Amplitude")
    if title is None:
        # if original was Tone/ToneBurst, call your naming function
        if not isinstance(sound, b2h.Sound):
            title = create_sound_key(sound)
        else:
            title = None

    ax.set_title(title)
    if xlim:
        ax.set_xlim(xlim)
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
    bin_size=1,
    hist_rate=False,
    cf_bin_size=3,
    raster_dot_size=1,
    figsize=(14, 18)
):

    side_colors = {'L': 'm', 'R': 'g'}

    duration = res.get("simulation_time", res["sounds"]["base_sound"].sound.duration / b2.ms)
    if xlim is None:
        xlim = [0, duration]

    # Build ylim from center_cf + bw_neurons, or default to full range
    if center_cf is not None and bw_neurons is not None:
        _n_tmp = len(res["cue_to_rate"][cue]["L"][pop]["global_ids"])
        _cf_tmp = greenwood_cf_array(CFMIN / b2.Hz, CFMAX / b2.Hz, _n_tmp) / b2.Hz
        _, center_idx = take_closest(_cf_tmp, center_cf)
        low_idx  = max(0, center_idx - bw_neurons)
        high_idx = min(_n_tmp - 1, center_idx + bw_neurons)
        ylim = [_cf_tmp[low_idx], _cf_tmp[high_idx]]
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
    ax_sound0.set_ylabel("HRTF-R")
    ax_sound0.set_xlim(xlim)
    ax_sound0.grid(True, alpha=0.3)

    ax_raster_R = fig.add_subplot(gs[1, 0])
    ax_hist_R   = fig.add_subplot(gs[1, 1], sharey=ax_raster_R)

    ax_sound2 = fig.add_subplot(gs[2, 0])
    t2 = np.arange(len(L_hrtf_sound)) / L_hrtf_sound.samplerate * 1000
    ax_sound2.plot(t2, L_hrtf_sound, color='m', lw=2)
    ax_sound2.set_ylabel("HRTF-L")
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
            print(f"Avg firing rate ({side} side): {avg_value:.2f} Hz POP")
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

    # ===========================================================
    # PSTH
    # ===========================================================
    for side in ["L", "R"]:
        color = side_colors[side]
        spikes = res["cue_to_rate"][cue][side][pop]

        times_f, _, _, ymin_idx, ymax_idx, _ = filter_spikes(spikes, xlim, ylim)

        bins = np.arange(xlim[0], xlim[1] + bin_size, bin_size)
        counts, _ = np.histogram(times_f, bins=bins)

        if hist_rate:
            rates = (counts * 1000.0) / (bin_size * (ymax_idx - ymin_idx + 1))
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
    rate=True,
    cf_interval=None,
    time_interval=None,       # [t_start, t_end] in ms
    center_cf=None,           # NEW: central CF in Hz to define the band of interest
    bw_neurons=None,          # NEW: half-bandwidth in number of neurons around center_cf
    sides=None,
    color=None,
    figsize=[7,4],
    title=None,
    ylim=None,
    label=None,
    error='sem',
    shaded=True,
    cue_type="angle",
    xlim=None  # "angle", "itd", or "ild"
):
    """
    Unified function:
    - Handles single dataset or list of datasets
    - Supports all normalization options
    - Can plot single population or all
    - Adds SEM/STD error ONLY when multiple datasets are provided
    - time_interval: [t_start, t_end] in ms — restrict spike counting to this window
    - cf_interval:   [cf_min, cf_max] in Hz — restrict spike counting to this CF band
    - center_cf:     central CF in Hz — combined with bw_neurons to define the CF band
    - bw_neurons:    half-bandwidth in number of neurons around center_cf index.
                     The resulting band [center_idx - bw_neurons, center_idx + bw_neurons]
                     is converted to Hz and used as cf_interval (overrides cf_interval).
      Rate is always spikes / neuron_in_band / time_window (Hz).
    """

    # Accept list of datasets
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

    # ------------------------------------------------------------------
    # Resolve center_cf + bw_neurons → cf_interval
    # Mirrors the same logic used in draw_spikes_and_psth_bothside.
    # This overrides any explicitly passed cf_interval when both
    # center_cf and bw_neurons are provided.
    # ------------------------------------------------------------------
    if center_cf is not None and bw_neurons is not None:
        # Use the first available cue/side to get the global_ids length
        _first_angle = list(cue_to_rate.keys())[0]
        _n_tmp = len(cue_to_rate[_first_angle]["L"][pop if isinstance(pop, str) and pop != "all" else "LSO"]["global_ids"])
        _cf_tmp = greenwood_cf_array(CFMIN / b2.Hz, CFMAX / b2.Hz, _n_tmp) / b2.Hz

        _, center_idx = take_closest(_cf_tmp, center_cf)
        low_idx  = max(0, center_idx - bw_neurons)
        high_idx = min(_n_tmp - 1, center_idx + bw_neurons)

        # Convert neuron index bounds back to Hz for cf_interval
        cf_interval = [_cf_tmp[low_idx], _cf_tmp[high_idx]]
        print(
            f"[draw_rate_vs_angle] center_cf={center_cf} Hz → "
            f"neuron idx [{low_idx}, {high_idx}] → "
            f"cf_interval=[{cf_interval[0]:.1f}, {cf_interval[1]:.1f}] Hz"
        )

    # ------------------------------------------------------------------
    # Helper: filter spikes by time window only.
    # CF filtering is delegated to calculate_firing_rates so that the
    # neuron-count denominator is computed correctly there.
    # ------------------------------------------------------------------
    def _filter_spike_dict(spike_dict, time_interval):
        times   = spike_dict["times"]
        senders = spike_dict["senders"]
        gids    = spike_dict["global_ids"]

        if time_interval is None:
            return spike_dict

        mask = (times >= time_interval[0]) & (times <= time_interval[1])
        return {
            "times":      times[mask],
            "senders":    senders[mask],
            "global_ids": gids,   # unchanged — needed for neuron count
        }

    # Effective duration for rate denominator
    if time_interval is not None:
        effective_duration = (time_interval[1] - time_interval[0]) * b2.ms
    else:
        effective_duration = duration

    def _draw_single_pop_subplot(ax, pop_name):

        cues = list(cue_to_rate.keys())
        # Sort cues numerically to ensure the line plot makes sense
        cues = sorted(cues)
        sides_local = ["L", "R"] if sides is None else sides

        # Side colors
        if isinstance(color, dict):
            side_colors = color
        elif isinstance(color, str):
            side_colors = {sides_local[0]: color}
        else:
            side_colors = {"L": "m", "R": "g"}

        # Collect rates from all datasets
        all_tot = {side: [] for side in sides_local}
        all_avg = {side: [] for side in sides_local}

        for d in multi_data:
            angle_to_rate_d = d["cue_to_rate"]

            # Pre-filter by time only (CF is handled inside calculate_firing_rates)
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
            else:
                atr_to_use = angle_to_rate_d

            # Effective duration: shrink to time window if given
            if time_interval is not None:
                dur_d = effective_duration
            else:
                dur_d = (
                    d.get(
                        "simulation_time",
                        data["sounds"]["base_sound"].sound.duration / b2.ms,
                    )
                    * b2.ms
                )

            # Always forward cf_interval — calculate_firing_rates handles CF neuron count
            tot_d, avg_d, _ = calculate_firing_rates(
                atr_to_use,
                pop_name,
                sides_local,
                cues,
                dur_d,
                cf_interval,  # always passed; may now be derived from center_cf+bw_neurons
            )

            for side in sides_local:
                all_tot[side].append(tot_d[side])
                all_avg[side].append(avg_d[side])

        # Mean across datasets
        tot_spikes = {side: np.mean(all_tot[side], axis=0) for side in sides_local}
        avg_neuron_rate = {
            side: np.mean(all_avg[side], axis=0) for side in sides_local
        }

        # Error metric (ONLY meaningful in multi_mode)
        if multi_mode:
            if error == "sem":
                err_factor = lambda x: np.std(x, axis=0) / np.sqrt(len(multi_data))
            elif error == "std":
                err_factor = lambda x: np.std(x, axis=0)
            else:
                raise ValueError("error must be 'sem' or 'std'")
            print("ERROR", err_factor)
            tot_err = {side: err_factor(all_tot[side]) for side in sides_local}
            avg_err = {side: err_factor(all_avg[side]) for side in sides_local}
        else:
            tot_err = avg_err = None

        # Select which rate to plot
        if rate is True:
            plotted_rate = avg_neuron_rate
            plotted_err  = avg_err
            ylabel_text  = "Avg Firing Rate [Hz]"

        elif rate is False:
            plotted_rate = tot_spikes
            plotted_err  = tot_err
            ylabel_text  = "Population Firing Rate [Hz]"

        elif rate == "mm_norm":
            plotted_rate, _ = normalize_rates(avg_neuron_rate, sides_local)
            plotted_err      = avg_err
            ylabel_text      = "Min-Max Normalized Rate"

        elif rate == "max_norm":
            plotted_rate = {
                side: np.array(avg_neuron_rate[side])
                / np.max(avg_neuron_rate[side])
                for side in sides_local
            }
            plotted_err = avg_err
            ylabel_text = "Max Normalized Rate"

        elif rate == "diff":
            plotted_rate = {
                "L":     np.array(avg_neuron_rate["L"]) / np.max(avg_neuron_rate["L"]),
                "R":     np.array(avg_neuron_rate["R"]) / np.max(avg_neuron_rate["R"]),
                "L_pop": np.array(tot_spikes["L"])      / np.max(tot_spikes["L"]),
                "R_pop": np.array(tot_spikes["R"])      / np.max(tot_spikes["R"]),
            }
            plotted_err = None
            ylabel_text = "Diff Avg-Pop"

        else:
            raise ValueError("Invalid rate option.")

        # Plot
        if rate in ["diff", "max_norm"]:
            for key, clr, lbl in [
                ("L",     "m",           "Avg_L"),
                ("L_pop", "darkmagenta", "Pop_L"),
                ("R",     "g",           "Avg_R"),
                ("R_pop", "darkgreen",   "Pop_R"),
            ]:
                if key in plotted_rate:
                    ax.plot(cues, plotted_rate[key], "o-", color=clr, label=lbl)
            ax.legend()

        else:
            for side in sides_local:
                mean_curve = plotted_rate[side]

                ax.plot(
                    cues,
                    mean_curve,
                    "o-",
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
                            cues,
                            mean_curve,
                            yerr=err_curve,
                            fmt="none",
                            capsize=3,
                            color=side_colors.get(side, "k"),
                        )

            if label is None:
                ax.legend()

        ax.set_ylabel(ylabel_text)

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
            # Apply xlim first so we know the visible range
            if xlim:
                xlim_s = [xlim[0]/1e6, xlim[1]/1e6]
                ax.set_xlim(xlim_s)
                visible_cues = [c for c in cues if xlim_s[0] <= c <= xlim_s[1]]
            else:
                visible_cues = cues

            # Subsample ticks if too many (target max_ticks)
            max_ticks = 10
            if len(visible_cues) > max_ticks:
                step = math.ceil(len(visible_cues) / max_ticks)
                visible_cues = visible_cues[::step]

            ax.set_xticks(visible_cues)
            ax.set_xticklabels(
                [f"{int(c * 1e6)}" for c in visible_cues],
                rotation=45,
                ha='right'
            )
            ax.set_xlabel("ITD [µs]")

        elif cue_type == "ild":
            ax.set_xticks(cues)
            ax.set_xticklabels([f"{c}" for c in cues])
            ax.set_xlabel("ILD [dB]")
            if xlim:
                ax.set_xlim(xlim)

        if ylim:
            ax.set_ylim(ylim)

        # Title with active filter info
        base_title = (
            f"{pop_name} ({len(multi_data)} subjects)" if multi_mode else pop_name
        )
        filter_parts = []
        if time_interval is not None:
            filter_parts.append(f"t=[{time_interval[0]},{time_interval[1]}] ms")
        if center_cf is not None and bw_neurons is not None:
            filter_parts.append(f"CF={center_cf} Hz ±{bw_neurons} neurons")
        elif cf_interval is not None:
            filter_parts.append(f"CF=[{cf_interval[0]:.0f},{cf_interval[1]:.0f}] Hz")
        ax.set_title(
            base_title + ("  |  " + ", ".join(filter_parts) if filter_parts else "")
        )

    # ---- SINGLE POP ----
    if isinstance(pop, str) and pop != "all":
        fig, ax = plt.subplots(figsize=figsize)
        _draw_single_pop_subplot(ax, pop)

        if title:
            fig.suptitle(title)

        plt.tight_layout()
        plt.show()
        return ax

    # ---- ALL POPS ----
    pops = ["ANF", "SBC", "GBC", "LNTBC", "MNTBC", "MSO", "LSO"] if pop == "all" else list(pop)

    n_rows = math.ceil(len(pops) / 3)
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 4 * n_rows))
    axes = np.array(axes).flatten()

    for ax, p in zip(axes, pops):
        _draw_single_pop_subplot(ax, p)

    for j in range(len(pops), len(axes)):
        axes[j].axis("off")

    if title:
        fig.suptitle(title)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    return axes

def plot_single_neuron_psth(
    spikes_file,
    target_cf_hz=None,
    xlim=None,
    ylim = None,
    bin_size=1.0,
    hist_rate=True,
    figsize=(10, 4),
    n_neighbors=0,
    color = 'b'
):
    """
    Plot PSTH of one ANF neuron or a small CF neighborhood.

    spikes_file: list-like, length = n_reps
        Each element is a list (length = n_neurons) of spike-time lists (seconds).

    n_neighbors:
        Number of neighboring neurons on EACH side of the target CF.
        0  -> single neuron
        k  -> pool neurons [cf_idx-k : cf_idx+k]
    """

    # ------------------------------------------------------------
    # Number of repetitions / neurons
    # ------------------------------------------------------------
    n_reps = len(spikes_file)
    n_neurons = len(spikes_file[0])

    # ------------------------------------------------------------
    # Greenwood CF mapping

    cf_array = greenwood_cf_array(CFMIN / Hz, CFMAX / Hz, n_neurons) 


    _, cf_idx = take_closest(cf_array, target_cf_hz * Hz)

    # ------------------------------------------------------------
    # Select neuron indices
    # ------------------------------------------------------------
    idx_min = max(0, cf_idx - n_neighbors)
    idx_max = min(n_neurons - 1, cf_idx + n_neighbors)
    sel_indices = np.arange(idx_min, idx_max + 1)

    print("Selected neurons:")
    print(f"  Target CF      : {target_cf_hz:.1f} Hz")
    print(f"  Center idx     : {cf_idx}")
    print(f"  Index range    : [{idx_min}, {idx_max}]")
    print(f"  CF range [Hz]  : {cf_array[idx_min]:.1f} – {cf_array[idx_max]:.1f}")
    print(f"  # neurons used : {len(sel_indices)}")

    # ------------------------------------------------------------
    # Collect spikes across repetitions and neurons
    # ------------------------------------------------------------
    pooled_times_ms = []

    for rep in spikes_file:
        for idx in sel_indices:
            if len(rep[idx]) > 0:
                pooled_times_ms.append(np.asarray(rep[idx]) * 1000.0)

    if len(pooled_times_ms) == 0:
        raise RuntimeError("No spikes found for selected neurons.")

    pooled_times_ms = np.concatenate(pooled_times_ms)

    if xlim is None:
        xlim = (0, pooled_times_ms.max())

    # ------------------------------------------------------------
    # PSTH
    # ------------------------------------------------------------
    bins = np.arange(xlim[0], xlim[1] + bin_size, bin_size)
    counts, _ = np.histogram(pooled_times_ms, bins=bins)

    if hist_rate:
        # spikes / (bin * repetitions * neurons)
        y = counts * 1000.0 / (bin_size * n_reps * len(sel_indices))
        ylabel = "Firing rate [Hz]"
    else:
        y = counts
        ylabel = "Spike count"

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    plt.figure(figsize=figsize)
    plt.plot(bins[:-1], y, lw=2, color = color)
    plt.xlabel("Time [ms]")
    plt.ylabel(ylabel)


    if n_neighbors > 0:
        title = (
        f"PSTH (CF neighborhood)\n"
        f"CF = {cf_array[cf_idx]:.0f} Hz"
    )
        title += f" ± {n_neighbors} neurons"
    else:
        title = (
        f"Single-neuron PSTH\n"
        f"CF = {cf_array[cf_idx]:.0f} Hz"
    )
    title += f" | {n_reps} repetitions"

    plt.title(title)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    return

def calculate_single_neuron_vector_strength(
    spikes_file,
    target_cf_hz,
    n_neighbors=0,
    n_bins=7,
    x_ax="phase",          # "phase" or "time"
    y_ax="percent",        # "percent" or "ashida"
    center_at_peak=False,
    figsize=(7, 5),
    color="b",
    ylim=None,
    display=True
):
    """
    Vector strength + phase/time histogram for a single neuron
    across repeated simulations.

    spikes_file : list, length = n_reps
        Each element is a list (length = n_neurons) of spike-time lists (seconds)

    target_cf_hz : float
        CF of neuron of interest


    n_neighbors : int
        Pool neurons [cf_idx - n_neighbors : cf_idx + n_neighbors]
    """


    # ------------------------------------------------------------
    # Dimensions
    # ------------------------------------------------------------
    n_reps = len(spikes_file)
    n_neurons = len(spikes_file[0])

    # ------------------------------------------------------------
    # Greenwood CF mapping
    # ------------------------------------------------------------
    cf_array = greenwood_cf_array(CFMIN / Hz, CFMAX / Hz, n_neurons) / Hz
    _, cf_idx = take_closest(cf_array, target_cf_hz)

    idx_min = max(0, cf_idx - n_neighbors)
    idx_max = min(n_neurons - 1, cf_idx + n_neighbors)
    sel_indices = np.arange(idx_min, idx_max + 1)

    # ------------------------------------------------------------
    # Pool spikes across repetitions & neurons
    # ------------------------------------------------------------
    pooled_spike_times = []

    for rep in spikes_file:
        for idx in sel_indices:
            if len(rep[idx]) > 0:
                pooled_spike_times.append(np.asarray(rep[idx]))

    if len(pooled_spike_times) == 0:
        return 0 if not display else (0, None)

    spike_times_array = np.concatenate(pooled_spike_times)
    total_spikes = len(spike_times_array)

    # ------------------------------------------------------------
    # Vector strength (UNCHANGED logic)
    # ------------------------------------------------------------
    phases = get_spike_phases(spike_times_array, target_cf_hz)
    vs = calculate_vector_strength(spike_times_array, target_cf_hz)

    if not display:
        return vs

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
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
            values = np.cue(np.exp(1j * (phases - peak_center)))
            bins = np.linspace(-np.pi, np.pi, n_bins + 1)
            bin_centers = (bins[:-1] + bins[1:]) / 2
        else:
            values = phases
            bins = orig_bins
            bin_centers = (bins[:-1] + bins[1:]) / 2

        hist, _ = np.histogram(values, bins=bins)
        bin_width = bins[1] - bins[0]

        if y_ax == "percent":
            y = hist / total_spikes * 100
            ylabel = "Spikes / bin (% of total)"
        elif y_ax == "ashida":
            y = hist / (total_spikes * bin_width)
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

        period_ms = 1000 / target_cf_hz
        time_values = (phases / (2 * np.pi)) * period_ms

        orig_bins = np.linspace(0, period_ms, n_bins + 1)
        hist_raw, _ = np.histogram(time_values, bins=orig_bins)
        peak_bin_idx = np.argmax(hist_raw)

        if center_at_peak:
            bin_centers_orig = (orig_bins[:-1] + orig_bins[1:]) / 2
            peak_center = bin_centers_orig[peak_bin_idx]
            values = np.mod(
                time_values - peak_center + period_ms / 2,
                period_ms
            ) - period_ms / 2
            bins = np.linspace(-period_ms / 2, period_ms / 2, n_bins + 1)
            bin_centers = (bins[:-1] + bins[1:]) / 2
        else:
            values = time_values
            bins = orig_bins
            bin_centers = (bins[:-1] + bins[1:]) / 2

        hist, _ = np.histogram(values, bins=bins)
        bin_width = bins[1] - bins[0]

        if y_ax == "percent":
            y = hist / total_spikes * 100
            ylabel = "Spikes / bin (% of total)"
        elif y_ax == "ashida":
            y = hist / (total_spikes * bin_width)
            ylabel = "Probability density (ms$^{-1}$)"
        else:
            raise ValueError("y_ax must be 'percent' or 'ashida'")

        ax.bar(bin_centers, y, width=bin_width, alpha=0.7, color=color)
        ax.set_xlabel("Time [ms]")

    ax.set_ylabel(ylabel)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_title(
        f"Single-neuron VS = {vs:.3f}\n"
        f"CF = {cf_array[cf_idx]:.0f} Hz | "
        f"{n_reps} reps | "
        f"{len(sel_indices)} neuron(s)"
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.show()

    return vs, fig

def draw_mm_norm_multi_dataset(
    data_list,
    pop="LSO",
    sides=None,
    cf_interval=None,
    colors=None,
    labels=None,
    figsize=(7, 4),
    title=None,
    ylim=None,
):
    """
    Plot multiple datasets simultaneously (same population),
    all min–max normalized.

    - data_list: list of dataset dicts
    - NO SEM / STD
    - NO histogram
    - Pure comparison plot
    """

    if not isinstance(data_list, list) or len(data_list) < 2:
        raise ValueError("data_list must be a list with at least two datasets")

    sides_local = ["L", "R"] if sides is None else sides

    if colors is None:
        colors = plt.cm.tab10.colors
    if labels is None:
        labels = [f"Dataset {i+1}" for i in range(len(data_list))]

    def _extract_mm_norm(data):
        cue_to_rate = data["cue_to_rate"]
        cues = list(cue_to_rate.keys())

        duration = (
            data.get(
                "simulation_time",
                data["sounds"]["base_sound"].sound.duration / b2.ms,
            )
            * b2.ms
        )

        tot, avg, _ = calculate_firing_rates(
            cue_to_rate,
            pop,
            sides_local,
            cues,
            duration,
            cf_interval,
        )

        mm_norm = {}
        for side in sides_local:
            r = np.asarray(avg[side])
            mm_norm[side] = (r - r.min()) / (r.max() - r.min())

        return cues, mm_norm

    # --- extract & check cue consistency ---
    all_angles = []
    all_norm = []

    for data in data_list:
        cues, norm = _extract_mm_norm(data)
        all_angles.append(cues)
        all_norm.append(norm)

    for a in all_angles[1:]:
        if a != all_angles[0]:
            raise ValueError("All datasets must share the same cue grid")

    cues = all_angles[0]

    # --- plot ---
    fig, ax = plt.subplots(figsize=figsize)

    for i, norm in enumerate(all_norm):
        for side in sides_local:
            ax.plot(
                cues,
                norm[side],
                marker="o",
                linestyle="-",
                color=colors[i % len(colors)],
                alpha=0.85,
                label=f"{labels[i]} – {side}",
            )

    ax.set_xticks(cues)
    ax.set_xticklabels([f"{a}°" for a in cues])
    ax.set_xlabel("Azimuth Angle")
    ax.set_ylabel("Min–Max Normalized Rate")

    if ylim:
        ax.set_ylim(ylim)

    ax.set_title(title if title else f"{pop} – Min–Max Normalized Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return ax

def draw_multi_dataset_raw_rates(
    data_list,
    pop="LSO",
    sides=None,
    cf_interval=None,
    colors=None,
    labels=None,
    figsize=(7, 4),
    title=None,
    ylim=None,
):
    """
    Plot multiple datasets simultaneously (same population),
    using raw average firing rates.

    - data_list: list of dataset dicts
    - NO normalization
    - NO SEM / STD
    - NO histogram
    - Pure comparison plot
    """

    if not isinstance(data_list, list) or len(data_list) < 2:
        raise ValueError("data_list must be a list with at least two datasets")

    sides_local = ["L", "R"] if sides is None else sides

    if colors is None:
        colors = plt.cm.tab10.colors
    if labels is None:
        labels = [f"Dataset {i+1}" for i in range(len(data_list))]

    def _extract_rates(data):
        cue_to_rate = data["cue_to_rate"]
        cues = list(cue_to_rate.keys())

        duration = (
            data.get(
                "simulation_time",
                data["sounds"]["base_sound"].sound.duration / b2.ms,
            )
            * b2.ms
        )

        _, avg, _ = calculate_firing_rates(
            cue_to_rate,
            pop,
            sides_local,
            cues,
            duration,
            cf_interval,
        )

        return cues, avg

    # --- extract & consistency check ---
    all_angles = []
    all_avg = []

    for data in data_list:
        cues, avg = _extract_rates(data)
        all_angles.append(cues)
        all_avg.append(avg)

    for a in all_angles[1:]:
        if a != all_angles[0]:
            raise ValueError("All datasets must share the same cue grid")

    cues = all_angles[0]

    # --- plot ---
    fig, ax = plt.subplots(figsize=figsize)

    for i, avg in enumerate(all_avg):
        for side in sides_local:
            ax.plot(
                cues,
                avg[side],
                marker="o",
                linestyle="-",
                color=colors[i % len(colors)],
                alpha=0.85,
                label=f"{labels[i]} – {side}",
            )

    ax.set_xticks(cues)
    ax.set_xticklabels([f"{a}°" for a in cues])
    ax.set_xlabel("Azimuth Angle")
    ax.set_ylabel("Avg Firing Rate [Hz]")

    if ylim:
        ax.set_ylim(ylim)

    ax.set_title(title if title else f"{pop} – Raw Rate Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return ax

## Claude's SHIT

def _filter_spikes_to_cf_window(spikes, center_cf=None, bw_neurons=None):
    gids    = spikes["global_ids"]
    times   = spikes["times"]
    senders = spikes["senders"]
    n       = len(gids)

    if center_cf is None or bw_neurons is None:
        return {"times": times, "senders": senders,
                "global_ids": gids, "_n_window": n}

    cf_hz = greenwood_cf_array(CFMIN / b2.Hz, CFMAX / b2.Hz, n) / b2.Hz
    _, center_idx = take_closest(cf_hz, center_cf)
    low_idx  = max(0, center_idx - bw_neurons)
    high_idx = min(n - 1, center_idx + bw_neurons)

    cf_min_id = gids[0] + low_idx
    cf_max_id = gids[0] + high_idx
    mask = (senders >= cf_min_id) & (senders <= cf_max_id)

    return {"times": times[mask], "senders": senders[mask],
            "global_ids": gids[low_idx : high_idx + 1],
            "_n_window": high_idx - low_idx + 1}


def _get_psth_rates(spikes, xlim_peak, bin_size,
                    center_cf=None, bw_neurons=None,
                    psth_filtering=False, stim_fs=None, smooth_cutoff_hz=None):
    """
    Compute PSTH rates and bin centres.
    If psth_filtering=True, apply a zero-phase Butterworth low-pass:
      - cutoff = smooth_cutoff_hz if provided
      - else cutoff = 3 * stim_fs if stim_fs provided
      - else raises ValueError
    """
    from scipy.signal import butter, filtfilt

    spikes  = _filter_spikes_to_cf_window(spikes, center_cf, bw_neurons)
    n       = spikes.get("_n_window", len(spikes["global_ids"]))

    mask    = (spikes["times"] >= xlim_peak[0]) & (spikes["times"] <= xlim_peak[1])
    times_f = spikes["times"][mask]

    bins          = np.arange(xlim_peak[0], xlim_peak[1] + bin_size, bin_size)
    counts, edges = np.histogram(times_f, bins=bins)
    rates         = (counts * 1000.0) / (bin_size * n)
    centres       = 0.5 * (edges[:-1] + edges[1:])

    if psth_filtering:
        fs_psth = 1000.0 / bin_size
        nyq     = fs_psth / 2.0

        if smooth_cutoff_hz is None:
            if stim_fs is None:
                raise ValueError(
                    "psth_filtering=True requires either smooth_cutoff_hz or stim_fs."
                )
            smooth_cutoff_hz = 3.0 * stim_fs

        if smooth_cutoff_hz >= nyq:
            raise ValueError(
                f"smooth_cutoff_hz={smooth_cutoff_hz:.1f} Hz >= Nyquist={nyq:.1f} Hz. "
                f"Reduce bin_size or lower smooth_cutoff_hz."
            )

        b, a   = butter(N=4, Wn=smooth_cutoff_hz / nyq, btype='low')
        rates  = filtfilt(b, a, rates)

    return rates, centres


def _get_sorted_peaks(rates, centres, bin_size, n_peaks=None, prominence=50.0):
    """
    Return peak times sorted chronologically.
    Minimum spacing = 1/bin_size bins (1 ms). Minimum prominence filter applied.
    If n_peaks is not None, return the first n chronologically.
    """
    from scipy.signal import find_peaks

    peak_idxs, _ = find_peaks(
        rates,
        distance   = max(1, int(1.0 / bin_size)),
        prominence = prominence,
    )
    if len(peak_idxs) == 0:
        return np.array([]), np.array([])

    top = sorted(peak_idxs)
    if n_peaks is not None:
        top = top[:n_peaks]

    return centres[top], rates[top]


def _onset_peak(rates, centres, bin_size, prominence=50.0):
    """
    Highest-amplitude peak among the first 5 chronological peaks.
    """
    times, ampls = _get_sorted_peaks(rates, centres, bin_size,
                                      n_peaks=5, prominence=prominence)
    if len(times) == 0:
        return np.nan, np.nan
    idx = np.argmax(ampls)
    return times[idx], ampls[idx]


# ─────────────────────────────────────────────────────────────────────────────
# METRIC COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

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