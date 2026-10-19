: Phenomenological Ca-dependent AHP K current
: Models slow afterhyperpolarization in LSO principal neurons.
: After Zhou & Colburn 2010 (J Neurophysiol 103:2987-3007).
:
: A proxy Ca variable accumulates during depolarization (sigmoid function
: of v above vhalf, activation at AP peak), decays with tau_ca.
: K-AHP conductance scales as ca/(ca+kd)  (Michaelis-Menten saturation).

NEURON {
    SUFFIX lso_ahp
    USEION k READ ek WRITE ik
    RANGE gbar, tau_ca, vhalf, kslope, kd
}

PARAMETER {
    gbar   = 0.002 (S/cm2)   : max K-AHP conductance
    tau_ca = 200.0 (ms)       : Ca proxy decay time constant (~200 ms slow AHP)
    vhalf  = -10.0 (mV)       : half-activation voltage (near AP peak)
    kslope =  10.0 (mV)       : activation slope (positive -> activates with depol)
    kd     =  0.5             : half-saturation [Ca] for K activation (dimensionless)
}

STATE { ca }

ASSIGNED {
    v  (mV)
    ek (mV)
    ik (mA/cm2)
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    ik = gbar * (ca / (ca + kd)) * (v - ek)
}

INITIAL {
    ca = 0
}

DERIVATIVE states {
    LOCAL ca_inf
    ca_inf = 1 / (1 + exp(-(v - vhalf) / kslope))
    ca' = (ca_inf - ca) / tau_ca
}
