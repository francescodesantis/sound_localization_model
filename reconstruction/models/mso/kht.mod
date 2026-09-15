: High-threshold potassium channel (Kht) - Rothman & Manis 2003, J Neurophysiol 89:3070-3087

NEURON {
    SUFFIX kht
    USEION k READ ek WRITE ik
    RANGE gbar
}

PARAMETER {
    gbar = 0.015 (S/cm2)
}

STATE { n p }

ASSIGNED {
    v  (mV)
    ek (mV)
    ik (mA/cm2)
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    ik = gbar * (0.85*n^2 + 0.15*p) * (v - ek)
}

INITIAL {
    n = ninf(v)
    p = pinf(v)
}

DERIVATIVE states {
    n' = (ninf(v) - n) / ntau(v)
    p' = (pinf(v) - p) / ptau(v)
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
