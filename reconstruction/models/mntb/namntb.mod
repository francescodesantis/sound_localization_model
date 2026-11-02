: Fast sodium channel (NaV) of the MNTB principal cell.
: Rothman & Manis 2003 kinetics (as models/mso/nax.mod), unique SUFFIX namntb.
:
: Gating rates scale with Q10 = 2.2 from 22 degC (tadj ~ 3.26 at 37 degC):
: NaV gating is less temperature-sensitive than K (Q10 2.2-2.5 vs 2.8-3.0).
: Conductance is not scaled (published gmax are functional 37 degC values).
:
: Difference from nax: hinf shifted +5 mV (NaV1.6-rich MNTB soma inactivates
: at a more depolarised V_1/2, keeping hinf > 0.7-0.8 at rest so the cell
: follows 200-400 Hz; Leao et al. 2005).

NEURON {
    SUFFIX namntb
    USEION na READ ena WRITE ina
    RANGE gbar
    GLOBAL q10, tau_scale
}

PARAMETER {
    q10 = 2.2
    tau_scale = 1.0
    gbar = 0.35 (S/cm2)
}

STATE { m h }

ASSIGNED {
    celsius (degC)
    tadj
    v   (mV)
    ena (mV)
    ina (mA/cm2)
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    ina = gbar * m^3 * h * (v - ena)
}

INITIAL {
    tadj = q10^((celsius - 22) / 10)
    m = minf(v)
    h = hinf(v)
}

DERIVATIVE states {
    m' = (minf(v) - m) * tadj / mtau(v)
    h' = (hinf(v) - h) * tadj / htau(v)
}

FUNCTION minf(v(mV)) {
    minf = 1 / (1 + exp(-(v + 38) / 7))
}

FUNCTION mtau(v(mV)) (ms) {
    mtau = tau_scale * (10 / (5*exp((v+60)/18) + 36*exp(-(v+60)/25)) + 0.04)
}

FUNCTION hinf(v(mV)) {
    hinf = 1 / (1 + exp((v + 60) / 6))
}

FUNCTION htau(v(mV)) (ms) {
    htau = tau_scale * (100 / (7*exp((v+60)/11) + 10*exp(-(v+60)/25)) + 0.6)
}
