: MNTB copy of models/mso/ih.mod, unique SUFFIX ih_mntb, same kinetics.
:
: Gating rates carry Q10 = 3.0 from the 22 degC the Rothman & Manis 2003
: kinetics were fitted at (tadj ~ 5.21 at 37 degC); gmax is not scaled.

NEURON {
    SUFFIX ih_mntb
    NONSPECIFIC_CURRENT ih
    RANGE gbar, eh
    GLOBAL q10
}

PARAMETER {
    q10 = 3.0
    gbar = 0.002 (S/cm2)
    eh   = -43   (mV)
}

STATE { r }

ASSIGNED {
    celsius (degC)
    tadj
    v  (mV)
    ih (mA/cm2)
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    ih = gbar * r * (v - eh)
}

INITIAL {
    tadj = q10^((celsius - 22) / 10)
    r = rinf(v)
}

DERIVATIVE states {
    r' = (rinf(v) - r) * tadj / rtau(v)
}

FUNCTION rinf(v(mV)) {
    rinf = 1 / (1 + exp((v + 76) / 7))
}

FUNCTION rtau(v(mV)) (ms) {
    rtau = 100000 / (237*exp((v+60)/12) + 17*exp(-(v+60)/14)) + 25
}
