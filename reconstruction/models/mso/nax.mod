: Fast sodium channel (NaV) - Rothman & Manis 2003, J Neurophysiol 89:3070-3087

: Temperature: gating RATES scale with Q10 = 3.0 from the 22 degC the kinetics
: were fitted at (Rothman & Manis 2003), tadj = 3^((celsius-22)/10) ~ 5.2 at
: 37 degC. The conductance is NOT scaled: published gmax are the functional
: 37 degC values (Wang & Colburn 2012, Brughera et al. 2018).

NEURON {
    SUFFIX nax
    USEION na READ ena WRITE ina
    RANGE gbar
    GLOBAL q10
}

PARAMETER {
    q10 = 3.0
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
    mtau = 10 / (5*exp((v+60)/18) + 36*exp(-(v+60)/25)) + 0.04
}

FUNCTION hinf(v(mV)) {
    hinf = 1 / (1 + exp((v + 65) / 6))
}

FUNCTION htau(v(mV)) (ms) {
    htau = 100 / (7*exp((v+60)/11) + 10*exp(-(v+60)/25)) + 0.6
}
