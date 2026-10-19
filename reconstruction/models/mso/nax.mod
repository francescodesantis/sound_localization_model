: Fast sodium channel (NaV) - Rothman & Manis 2003, J Neurophysiol 89:3070-3087

NEURON {
    SUFFIX nax
    USEION na READ ena WRITE ina
    RANGE gbar
}

PARAMETER {
    gbar = 0.35 (S/cm2)
}

STATE { m h }

ASSIGNED {
    v   (mV)
    ena (mV)
    ina (mA/cm2)
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    ina = gbar * m^3 * h * (v - ena)
}

INITIAL {
    m = minf(v)
    h = hinf(v)
}

DERIVATIVE states {
    m' = (minf(v) - m) / mtau(v)
    h' = (hinf(v) - h) / htau(v)
}

FUNCTION minf(v(mV)) {
    minf = 1 / (1 + exp(-(v + 38) / 7))
}

FUNCTION mtau(v(mV)) (ms) {
    mtau = 10 / (5*exp((v+60)/18) + 36*exp(-(v+60)/25)) + 0.04
}

FUNCTION hinf(v(mV)) {
    hinf = 1 / (1 + exp((v + 65) / 6))
}

FUNCTION htau(v(mV)) (ms) {
    htau = 100 / (7*exp((v+60)/11) + 10*exp(-(v+60)/25)) + 0.6
}
