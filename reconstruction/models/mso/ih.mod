: Hyperpolarization-activated cation current (Ih / HCN)
: Rothman & Manis 2003, J Neurophysiol 89:3070-3087

: Temperature: gating RATES scale with Q10 = 3.0 from the 22 degC the kinetics
: were fitted at (Rothman & Manis 2003), tadj = 3^((celsius-22)/10) ~ 5.2 at
: 37 degC. The conductance is NOT scaled: published gmax are the functional
: 37 degC values (Wang & Colburn 2012, Brughera et al. 2018).

NEURON {
    SUFFIX ih
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
