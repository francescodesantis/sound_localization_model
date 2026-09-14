: Fast sodium channel (NaV) for MNTB principal cells.
: Based on Rothman & Manis 2003 (see models/mso/nax.mod) but with faster
: inactivation recovery, matching the MNTB's ability to phase-lock and fire
: high-frequency (200-400 Hz) trains to calyx-of-Held input (Srinivasan 2019;
: Kaczmarek/Kim MNTB Na literature). Unique SUFFIX (namntb) so it coexists in one
: NEURON process with the auto-loaded MSO/repo-root nax (no suffix collision).
:
: Differences from nax:
:   * htau scaled ~0.3x (faster recovery from inactivation)
:   * hinf shifted +5 mV depolarised (more Na available near rest / during trains)

NEURON {
    SUFFIX namntb
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
    hinf = 1 / (1 + exp((v + 60) / 6))
}

FUNCTION htau(v(mV)) (ms) {
    htau = 0.3 * (100 / (7*exp((v+60)/11) + 10*exp(-(v+60)/25))) + 0.3
}
