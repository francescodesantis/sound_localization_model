: Hyperpolarization-activated cation current (Ih / HCN)
: Rothman & Manis 2003, J Neurophysiol 89:3070-3087

NEURON {
    SUFFIX ih
    NONSPECIFIC_CURRENT ih
    RANGE gbar, eh
}

PARAMETER {
    gbar = 0.002 (S/cm2)
    eh   = -43   (mV)
}

STATE { r }

ASSIGNED {
    v  (mV)
    ih (mA/cm2)
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    ih = gbar * r * (v - eh)
}

INITIAL {
    r = rinf(v)
}

DERIVATIVE states {
    r' = (rinf(v) - r) / rtau(v)
}

FUNCTION rinf(v(mV)) {
    rinf = 1 / (1 + exp((v + 76) / 7))
}

FUNCTION rtau(v(mV)) (ms) {
    rtau = 100000 / (237*exp((v+60)/12) + 17*exp(-(v+60)/14)) + 25
}
