: Low-threshold potassium channel (Klt / Kv1)
: Rothman & Manis 2003, J Neurophysiol 89:3070-3087
: Validated for auditory brainstem neurons (MSO, AVCN)

NEURON {
    SUFFIX klt
    USEION k READ ek WRITE ik
    RANGE gbar
}

PARAMETER {
    gbar = 0.02 (S/cm2)
}

STATE { w h }

ASSIGNED {
    v   (mV)
    ek  (mV)
    ik  (mA/cm2)
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    ik = gbar * w^4 * h * (v - ek)
}

INITIAL {
    w = winf(v)
    h = hinf(v)
}

DERIVATIVE states {
    w' = (winf(v) - w) / wtau(v)
    h' = (hinf(v) - h) / htau(v)
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
