: High-threshold potassium channel (Kht) - Rothman & Manis 2003, J Neurophysiol 89:3070-3087

: Temperature: gating RATES scale with Q10 = 3.0 from the 22 degC the kinetics
: were fitted at (Rothman & Manis 2003), tadj = 3^((celsius-22)/10) ~ 5.2 at
: 37 degC. The conductance is NOT scaled: published gmax are the functional
: 37 degC values (Wang & Colburn 2012, Brughera et al. 2018).

NEURON {
    SUFFIX kht
    USEION k READ ek WRITE ik
    RANGE gbar
    GLOBAL q10
}

PARAMETER {
    q10 = 3.0
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
