: MNTB copy of models/mso/klt.mod, unique SUFFIX klt_mntb, same kinetics.
:
: Gating rates carry Q10 = 3.0 from the 22 degC the Rothman & Manis 2003
: kinetics were fitted at (tadj ~ 5.21 at 37 degC); gmax is not scaled.

NEURON {
    SUFFIX klt_mntb
    USEION k READ ek WRITE ik
    RANGE gbar
    GLOBAL q10
}

PARAMETER {
    q10 = 3.0
    gbar = 0.02 (S/cm2)
}

STATE { w h }

ASSIGNED {
    celsius (degC)
    tadj
    v   (mV)
    ek  (mV)
    ik  (mA/cm2)
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    ik = gbar * w^4 * h * (v - ek)
}

INITIAL {
    tadj = q10^((celsius - 22) / 10)
    w = winf(v)
    h = hinf(v)
}

DERIVATIVE states {
    w' = (winf(v) - w) * tadj / wtau(v)
    h' = (hinf(v) - h) * tadj / htau(v)
}

FUNCTION winf(v(mV)) {
    winf = (1 / (1 + exp(-(v + 48) / 6)))^0.25
}

FUNCTION wtau(v(mV)) (ms) {
    wtau = 100 / (6*exp((v+60)/6) + 16*exp(-(v+60)/45)) + 1.5
}

FUNCTION hinf(v(mV)) {
    hinf = 1 / (1 + exp((v + 43) / 6))
}

FUNCTION htau(v(mV)) (ms) {
    htau = 1000 / (exp((v+60)/20) + exp(-(v+60)/8)) + 50
}
