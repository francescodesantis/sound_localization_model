: MNTB copy of models/mso/kht.mod (Kv3.1), unique SUFFIX kht_mntb.
:
: Gating rates scale with an effective Q10 = 2.0 from 22 degC (tadj ~ 2.83 at
: 37 degC); gmax is not scaled. Heterologous voltage-clamp data give 2.8-3.0,
: but during a free-running AP faster Kv3.1 kinetics shorten the depolarisation
: that opens it, a feedback a step protocol does not capture. This is the one
: difference from models/mso/kht.mod.

NEURON {
    SUFFIX kht_mntb
    USEION k READ ek WRITE ik
    RANGE gbar
    GLOBAL q10
}

PARAMETER {
    q10 = 2.0
    gbar = 0.015 (S/cm2)
}

STATE { n p }

ASSIGNED {
    celsius (degC)
    tadj
    v  (mV)
    ek (mV)
    ik (mA/cm2)
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    ik = gbar * (0.85*n^2 + 0.15*p) * (v - ek)
}

INITIAL {
    tadj = q10^((celsius - 22) / 10)
    n = ninf(v)
    p = pinf(v)
}

DERIVATIVE states {
    n' = (ninf(v) - n) * tadj / ntau(v)
    p' = (pinf(v) - p) * tadj / ptau(v)
}

FUNCTION ninf(v(mV)) {
    ninf = (1 + exp(-(v + 15) / 5))^(-0.5)
}

FUNCTION ntau(v(mV)) (ms) {
    ntau = 100 / (11*exp((v+60)/24) + 21*exp(-(v+60)/23)) + 0.7
}

FUNCTION pinf(v(mV)) {
    pinf = 1 / (1 + exp(-(v + 23) / 6))
}

FUNCTION ptau(v(mV)) (ms) {
    ptau = 100 / (4*exp((v+60)/32) + 5*exp(-(v+60)/22)) + 5
}
