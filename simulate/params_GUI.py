# ==========================================================
# PARAMS GUI FOR NEST BRAINSTEM MODEL
# Run with:
#     python params_GUI.py
# ==========================================================

import nest
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from models.BrainstemModel.params import Parameters as params

# ==========================================================
# LOAD PARAMETERS
# ==========================================================

P = params()

populations = ["SBC", "GBC", "MNTBC", "LNTBC", "LSO", "MSO", "SPN"]

all_params = {}
for pop in populations:
    all_params[pop] = {
        "C_m":          getattr(P.MEMB_CAPS,      pop),
        "g_L":          getattr(P.G_LEAK,          pop),
        "E_L":          getattr(P.E_L,             pop),
        "V_m":          getattr(P.E_L,             pop),
        "V_reset":      getattr(P.V_RESET,         pop),
        "V_th":         getattr(P.V_TH,            pop),
        "t_ref":        getattr(P.T_REF,           pop),
        "E_ex":         getattr(P.EXC_REV,         pop),
        "tau_rise_ex":  getattr(P.TAUS_EX_RISE,    pop),
        "tau_decay_ex": getattr(P.TAUS_EX_DECAY,   pop),
        "E_in":         getattr(P.INH_REV,         pop),
        "tau_rise_in":  getattr(P.TAUS_IN_RISE,    pop),
        "tau_decay_in": getattr(P.TAUS_IN_DECAY,   pop),
    }

SIM_DURATION = 50.0
VIEW_WINDOW  = 10.0

NEURON_RANGES = {
    "C_m":          (0,    500),
    "g_L":          (0,     50),
    "E_L":          (-90,  -50),
    "V_m":          (-90,  -50),
    "V_reset":      (-90,  -50),
    "V_th":         (-70,  -40),
    "t_ref":        (0,     10),
    "E_ex":         (-20,   20),
    "tau_rise_ex":  (0,     10),
    "tau_decay_ex": (0,     10),
    "E_in":         (-90,  +90),
    "tau_rise_in":  (0,     10),
    "tau_decay_in": (0,     10),
}

# ==========================================================
# HELPER
# ==========================================================

def make_pre_spike_times(base_times, num_inputs, delta=0.0):
    return [[t + i * delta for t in base_times] for i in range(num_inputs)]

# ==========================================================
# SIMULATION
# ==========================================================

def simulate(population, spike_string, n_ex, n_in,
             weights_ex, weights_in, delays_ex, delays_in,
             neuron_params):

    nest.ResetKernel()
    nest.SetKernelStatus(P.CONFIG.NEST_KERNEL_PARAMS)

    spike_times = [float(x.strip()) for x in spike_string.split(",")]

    post = nest.Create("iaf_cond_beta", 1, params=neuron_params)
    sr   = nest.Create("spike_recorder")
    mm   = nest.Create("multimeter", 1, {
        "record_from": ["V_m", "g_ex", "g_in"],
        "record_to":   "memory",
        "interval":    0.01
    })
    nest.Connect(post, sr)
    nest.Connect(mm,   post)

    if n_ex > 0:
        ex_times = make_pre_spike_times(spike_times, n_ex)
        for i, st in enumerate(ex_times):
            gen = nest.Create("spike_generator", 1, params={"spike_times": st})
            nest.Connect(gen, post, syn_spec={
                "weight": weights_ex[i],
                "delay":  delays_ex[i]
            })

    if n_in > 0:
        in_times = make_pre_spike_times(spike_times, n_in)
        for i, st in enumerate(in_times):
            gen = nest.Create("spike_generator", 1, params={"spike_times": st})
            nest.Connect(gen, post, syn_spec={
                "weight": weights_in[i],
                "delay":  delays_in[i]
            })

    nest.Simulate(SIM_DURATION)

    data        = nest.GetStatus(mm, "events")[0]
    times       = data["times"]
    V_m         = data["V_m"]
    g_ex        = data["g_ex"]
    g_in        = data["g_in"]
    I_ex        = g_ex * (V_m - neuron_params["E_ex"])
    I_in        = g_in * (V_m - neuron_params["E_in"])
    post_spikes = nest.GetStatus(sr, "events")[0]["times"]

    return times, V_m, g_ex, g_in, I_ex, I_in, post_spikes

# ==========================================================
# GUI
# ==========================================================

class App:
    def __init__(self, root):
        self.root = root
        root.title("NEST Brainstem Model")

        self._data   = None
        self._xstart = 0.0
        self._xwidth = VIEW_WINDOW
        self._yzoom  = [1.0, 1.0, 1.0]

        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)

        # --------------------------------------------------
        # Right side: plot + controls below
        # --------------------------------------------------
        plot_frame = ttk.Frame(root)
        plot_frame.grid(row=0, column=1, sticky="nsew")
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)

        self.fig, self.axes = plt.subplots(3, 1, figsize=(5, 4), sharex=True)
        self.fig.tight_layout(pad=1.5)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, columnspan=2, sticky="nsew")

        xscroll_frame = ttk.Frame(plot_frame)
        xscroll_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=4, pady=(0,2))
        xscroll_frame.columnconfigure(1, weight=1)
        ttk.Label(xscroll_frame, text="Scroll (ms)").grid(row=0, column=0, padx=(0,4))
        self._xscroll_var = tk.DoubleVar(value=0.0)
        self._xscroll = ttk.Scale(xscroll_frame, from_=0,
                                  to=SIM_DURATION - VIEW_WINDOW,
                                  variable=self._xscroll_var, orient="horizontal")
        self._xscroll.grid(row=0, column=1, sticky="ew")
        self._xscroll_lbl = ttk.Label(xscroll_frame, text="0.0 ms", width=8)
        self._xscroll_lbl.grid(row=0, column=2, padx=(4,0))
        self._xscroll_var.trace_add("write", self._on_xscroll)

        xwin_frame = ttk.Frame(plot_frame)
        xwin_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=4, pady=(0,2))
        xwin_frame.columnconfigure(1, weight=1)
        ttk.Label(xwin_frame, text="Window (ms)").grid(row=0, column=0, padx=(0,4))
        self._xwidth_var = tk.DoubleVar(value=VIEW_WINDOW)
        self._xwidth_sl  = ttk.Scale(xwin_frame, from_=1, to=SIM_DURATION,
                                     variable=self._xwidth_var, orient="horizontal")
        self._xwidth_sl.grid(row=0, column=1, sticky="ew")
        self._xwidth_lbl = ttk.Label(xwin_frame, text=f"{VIEW_WINDOW:.0f} ms", width=8)
        self._xwidth_lbl.grid(row=0, column=2, padx=(4,0))
        self._xwidth_var.trace_add("write", self._on_xwidth)

        yzoom_frame = ttk.Frame(plot_frame)
        yzoom_frame.grid(row=3, column=0, columnspan=2, sticky="ew", padx=4, pady=(0,4))
        for i, label in enumerate(["g zoom", "I zoom", "Vm zoom"]):
            ttk.Label(yzoom_frame, text=label).grid(row=0, column=i*3,   padx=(8,2))
            ttk.Button(yzoom_frame, text="−", width=2,
                       command=lambda idx=i: self._yzoom_change(idx, 1.5)
                       ).grid(row=0, column=i*3+1)
            ttk.Button(yzoom_frame, text="+", width=2,
                       command=lambda idx=i: self._yzoom_change(idx, 1/1.5)
                       ).grid(row=0, column=i*3+2)

        # spike times readout
        spike_readout_frame = ttk.Frame(plot_frame)
        spike_readout_frame.grid(row=4, column=0, columnspan=2, sticky="ew", padx=6, pady=(0,4))
        ttk.Label(spike_readout_frame, text="Post spikes (ms):",
                  font=("", 8, "bold")).pack(side="left")
        self._spike_readout = ttk.Label(spike_readout_frame, text="—",
                                        font=("", 8), foreground="red")
        self._spike_readout.pack(side="left", padx=(6, 0))

        # --------------------------------------------------
        # Left panel: scrollable controls
        # --------------------------------------------------
        ctrl_outer = ttk.Frame(root)
        ctrl_outer.grid(row=0, column=0, sticky="ns")

        self._canvas = tk.Canvas(ctrl_outer, width=400, highlightthickness=0)
        self._sb     = ttk.Scrollbar(ctrl_outer, orient="vertical",
                                     command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=self._sb.set)
        self._sb.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)

        ctrl = ttk.Frame(self._canvas, padding=(8, 6))
        self._win_id = self._canvas.create_window((0, 0), window=ctrl, anchor="nw")
        ctrl.bind("<Configure>", lambda e: self._canvas.configure(
            scrollregion=self._canvas.bbox("all")))
        self._canvas.bind("<Configure>", lambda e:
            self._canvas.itemconfig(self._win_id, width=e.width))
        self.canvas.get_tk_widget().bind("<Configure>", lambda e:
            self._canvas.configure(height=e.height))
        self._canvas.bind("<Enter>", lambda e:
            self._canvas.bind_all("<MouseWheel>", lambda ev:
                self._canvas.yview_scroll(int(-1*(ev.delta/120)), "units")))
        self._canvas.bind("<Leave>", lambda e:
            self._canvas.unbind_all("<MouseWheel>"))

        # --------------------------------------------------
        # Controls content
        # --------------------------------------------------
        ttk.Label(ctrl, text="Population").pack(anchor="w")
        self.pop_var = tk.StringVar(value=populations[0])
        pop_menu = ttk.Combobox(ctrl, textvariable=self.pop_var,
                                values=populations, state="readonly")
        pop_menu.pack(fill="x")
        pop_menu.bind("<<ComboboxSelected>>", self.on_pop_change)

        ttk.Label(ctrl, text="Spike times (ms)").pack(anchor="w", pady=(5,0))
        self.spike_var = tk.StringVar(value="2")
        spike_entry = ttk.Entry(ctrl, textvariable=self.spike_var)
        spike_entry.pack(fill="x")
        spike_entry.bind("<Return>",   lambda e: self.run())
        spike_entry.bind("<FocusOut>", lambda e: self.run())

        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", pady=5)

        self.n_ex = self._spinbox(ctrl, "N excitatory", default=2,
                                  callback=self._rebuild_syn_panel)
        self.n_in = self._spinbox(ctrl, "N inhibitory", default=0,
                                  callback=self._rebuild_syn_panel)

        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", pady=3)

        self.syn_frame = ttk.Frame(ctrl)
        self.syn_frame.pack(fill="x")
        self.syn_widgets = {}
        self._rebuild_syn_panel()

        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", pady=5)

        # neuron params header with restore button
        neuron_hdr = ttk.Frame(ctrl)
        neuron_hdr.pack(fill="x", pady=(0, 2))
        ttk.Label(neuron_hdr, text="Neuron parameters",
                  font=("", 9, "bold")).pack(side="left", anchor="w")
        ttk.Button(neuron_hdr, text="↺ Restore defaults",
                   command=self._restore_neuron_defaults).pack(side="right")

        self.neuron_frame = ttk.Frame(ctrl)
        self.neuron_frame.pack(fill="x")
        self.neuron_sliders = {}   # name -> tk.DoubleVar
        self._build_neuron_sliders(populations[0])

        self.run()

    # -------------------------------------------------------
    # Shared slider + entry factory (used for both syn and neuron params)
    # -------------------------------------------------------
    def _slider_with_entry(self, parent, label, lo, hi, default, label_width=13):
        """Slider + editable entry, bidirectionally synced. Returns tk.DoubleVar."""
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=1)

        ttk.Label(frame, text=label, width=label_width, anchor="w").pack(side="left")

        var = tk.DoubleVar(value=default)

        sl = ttk.Scale(frame, from_=lo, to=hi, variable=var, orient="horizontal")
        sl.pack(side="left", fill="x", expand=True)

        entry_var = tk.StringVar(value=f"{default:.3f}")
        entry = ttk.Entry(frame, textvariable=entry_var, width=7, justify="right")
        entry.pack(side="left", padx=(3, 0))

        _updating = [False]

        def _slider_moved(_=None):
            if _updating[0]:
                return
            _updating[0] = True
            entry_var.set(f"{var.get():.3f}")
            _updating[0] = False

        def _slider_released(_=None):
            _slider_moved()
            self.run()

        sl.bind("<Motion>",          _slider_moved)
        sl.bind("<ButtonRelease-1>", _slider_released)

        def _entry_commit(_=None):
            if _updating[0]:
                return
            try:
                v = float(entry_var.get().strip())
                v = max(lo, min(hi, v))
                _updating[0] = True
                var.set(v)
                entry_var.set(f"{v:.3f}")
                _updating[0] = False
                self.run()
            except ValueError:
                entry_var.set(f"{var.get():.3f}")

        entry.bind("<Return>",   _entry_commit)
        entry.bind("<FocusOut>", _entry_commit)

        return var

    def _spinbox(self, parent, label, default=0, callback=None):
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=1)
        ttk.Label(frame, text=label, width=13, anchor="w").pack(side="left")
        var = tk.IntVar(value=default)
        sb  = ttk.Spinbox(frame, from_=0, to=99, textvariable=var, width=5)
        sb.pack(side="left")
        def _on_change(*_):
            if callback:
                self.root.after_idle(callback)
            else:
                self.root.after_idle(self.run)
        var.trace_add("write", _on_change)
        return var

    # -------------------------------------------------------
    # Dynamic synaptic controls
    # -------------------------------------------------------
    def _rebuild_syn_panel(self):
        for w in self.syn_frame.winfo_children():
            w.destroy()
        self.syn_widgets.clear()

        try:
            n_ex = self.n_ex.get()
            n_in = self.n_in.get()
        except Exception:
            return

        if n_ex > 0:
            ttk.Label(self.syn_frame, text="— Excitatory —",
                      font=("", 8, "italic")).pack(anchor="w")
            if n_ex == 2:
                for i in (1, 2):
                    self.syn_widgets[f"w_ex{i}"] = self._slider_with_entry(
                        self.syn_frame, f"w_ex {i}", 0, 300, 5)
                    self.syn_widgets[f"d_ex{i}"] = self._slider_with_entry(
                        self.syn_frame, f"delay_ex {i}", 0.1, 5, 1)
            else:
                self.syn_widgets["w_ex"] = self._slider_with_entry(
                    self.syn_frame, "w_ex", 0, 300, 5)
                self.syn_widgets["d_ex"] = self._slider_with_entry(
                    self.syn_frame, "delay_ex", 0.1, 5, 1)

        if n_in > 0:
            ttk.Label(self.syn_frame, text="— Inhibitory —",
                      font=("", 8, "italic")).pack(anchor="w")
            if n_in == 2:
                for i in (1, 2):
                    self.syn_widgets[f"w_in{i}"] = self._slider_with_entry(
                        self.syn_frame, f"w_in {i}", -300, 0, -5)
                    self.syn_widgets[f"d_in{i}"] = self._slider_with_entry(
                        self.syn_frame, f"delay_in {i}", 0.1, 5, 1)
            else:
                self.syn_widgets["w_in"] = self._slider_with_entry(
                    self.syn_frame, "w_in", -300, 0, -5)
                self.syn_widgets["d_in"] = self._slider_with_entry(
                    self.syn_frame, "delay_in", 0.1, 5, 1)

        self.run()

    def _get_syn_lists(self):
        n_ex = self.n_ex.get()
        n_in = self.n_in.get()
        sw   = self.syn_widgets

        if n_ex == 2:
            weights_ex = [sw["w_ex1"].get(), sw["w_ex2"].get()]
            delays_ex  = [sw["d_ex1"].get(), sw["d_ex2"].get()]
        elif n_ex > 0:
            weights_ex = [sw["w_ex"].get()] * n_ex
            delays_ex  = [sw["d_ex"].get()] * n_ex
        else:
            weights_ex, delays_ex = [], []

        if n_in == 2:
            weights_in = [sw["w_in1"].get(), sw["w_in2"].get()]
            delays_in  = [sw["d_in1"].get(), sw["d_in2"].get()]
        elif n_in > 0:
            weights_in = [sw["w_in"].get()] * n_in
            delays_in  = [sw["d_in"].get()] * n_in
        else:
            weights_in, delays_in = [], []

        return weights_ex, delays_ex, weights_in, delays_in

    # -------------------------------------------------------
    # Neuron parameter controls
    # -------------------------------------------------------
    def _build_neuron_sliders(self, pop):
        for w in self.neuron_frame.winfo_children():
            w.destroy()
        self.neuron_sliders.clear()

        for name, val in all_params[pop].items():
            lo, hi = NEURON_RANGES[name]
            self.neuron_sliders[name] = self._slider_with_entry(
                self.neuron_frame, name, lo=lo, hi=hi, default=val)

    def _restore_neuron_defaults(self):
        """Reset all neuron sliders/entries to the params-file values for the current population."""
        pop = self.pop_var.get()
        for name, var in self.neuron_sliders.items():
            var.set(all_params[pop][name])
        # entry fields update automatically via the var trace in _slider_with_entry
        self.run()

    def on_pop_change(self, _=None):
        self._build_neuron_sliders(self.pop_var.get())
        self.run()

    # -------------------------------------------------------
    # View helpers
    # -------------------------------------------------------
    def _apply_view(self):
        if self._data is None:
            return
        x0 = self._xstart
        x1 = min(x0 + self._xwidth, SIM_DURATION)
        for i, ax in enumerate(self.axes):
            ax.set_xlim(x0, x1)
            ylo, yhi = ax.get_ylim()
            yctr  = (ylo + yhi) / 2
            yhalf = (yhi - ylo) / 2 * self._yzoom[i]
            ax.set_ylim(yctr - yhalf, yctr + yhalf)
        self.fig.tight_layout(pad=1.5)
        self.canvas.draw()

    def _on_xscroll(self, *_):
        self._xstart = self._xscroll_var.get()
        self._xscroll_lbl.config(text=f"{self._xstart:.1f} ms")
        self._apply_view()

    def _on_xwidth(self, *_):
        self._xwidth = self._xwidth_var.get()
        self._xwidth_lbl.config(text=f"{self._xwidth:.0f} ms")
        max_start = max(0.0, SIM_DURATION - self._xwidth)
        self._xscroll.config(to=max_start)
        if self._xstart > max_start:
            self._xscroll_var.set(max_start)
        self._apply_view()

    def _yzoom_change(self, axis_idx, factor):
        self._yzoom[axis_idx] *= factor
        self._apply_view()

    # -------------------------------------------------------
    # Simulation
    # -------------------------------------------------------
    def run(self):
        try:
            neuron_params = {k: v.get() for k, v in self.neuron_sliders.items()}
            weights_ex, delays_ex, weights_in, delays_in = self._get_syn_lists()

            times, V_m, g_ex, g_in, I_ex, I_in, post_spikes = simulate(
                population    = self.pop_var.get(),
                spike_string  = self.spike_var.get(),
                n_ex          = self.n_ex.get(),
                n_in          = self.n_in.get(),
                weights_ex    = weights_ex,
                weights_in    = weights_in,
                delays_ex     = delays_ex,
                delays_in     = delays_in,
                neuron_params = neuron_params,
            )

            self._data  = (times, V_m, g_ex, g_in, I_ex, I_in, post_spikes)
            self._yzoom = [1.0, 1.0, 1.0]

            ylabels = ["g (nS)", "I (pA)", "V_m (mV)"]
            for ax, lbl in zip(self.axes, ylabels):
                ax.clear()
                ax.set_ylabel(lbl, fontsize=8)
                ax.tick_params(labelsize=7)

            self.axes[0].plot(times, g_ex, color="tab:red",  label="g_ex")
            self.axes[0].plot(times, g_in, color="tab:blue", label="g_in")
            self.axes[0].legend(fontsize=6, loc="upper right")

            self.axes[1].plot(times, I_ex, color="tab:red",  label="I_ex")
            self.axes[1].plot(times, I_in, color="tab:blue", label="I_in")
            self.axes[1].legend(fontsize=6, loc="upper right")

            self.axes[2].plot(times, V_m, color="tab:green")
            if len(post_spikes):
                self.axes[2].scatter(post_spikes, [-40]*len(post_spikes),
                                     color="red", s=20, zorder=5)
            self.axes[2].set_ylim([-85, -35])
            self.axes[2].set_xlabel("Time (ms)", fontsize=8)

            self.fig.tight_layout(pad=1.5)
            self._xstart = self._xscroll_var.get()
            self._xwidth = self._xwidth_var.get()
            for ax in self.axes:
                ax.set_xlim(self._xstart, self._xstart + self._xwidth)

            self.canvas.draw()

            if len(post_spikes):
                spike_str = ",  ".join(f"{t:.2f}" for t in post_spikes)
                self._spike_readout.config(text=spike_str)
            else:
                self._spike_readout.config(text="—  (no spikes)")

        except Exception as e:
            messagebox.showerror("Simulation error", str(e))

# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1200x800")
    app = App(root)
    root.mainloop()