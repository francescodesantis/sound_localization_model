from dataclasses import dataclass, field
from utils.cochlea_utils import ITD_REMOVAL_STRAT, NUM_CF, NUM_ANF_PER_HC

@dataclass
class Parameters:
    key: str = "default_params"

    cochlea: dict[str, dict[str, float]] = field(
        default_factory=lambda: (
            {
                "gammatone": {
                    "hrtf_params": {
                        "subj_number": 0,
                        "ild_only": False,
                        "itd_remove_strategy": ITD_REMOVAL_STRAT.COMPUTED,
                        "apply_gating": True,
                        "ramp_ms": 10,
                    },
                    "noise_factor": 0.3,
                    "refractory_period": 1,  # ms
                    "amplif_factor": 7,
                },
                "TanCarney": {
                    "hrtf_params": {
                        "subj_number": 0,
                        "itd_remove_strategy": ITD_REMOVAL_STRAT.ESTIMATE_FROM_HRTF,
                        "apply_gating": True,
                        "ramp_ms": 10,
                    },
                    "cochlea_params": None,
                    "rng_seed": 42,
                    "omni_noise_level": 0,
                },
                "Zilany": {
                    "hrtf_params": {
                        "subj_number": 0,
                        "cue_to_apply": "HRTF", # "HRTF", "itd_only", "ild_only"
                        "simulation_mode": "angle", # New: "angle", "artificial_itd", "artificial_ild"
                        "artificial_itd": 0.0,      # in seconds (e.g. 0.0005 for 500us)
                        "artificial_ild": 0.0,      # in dB
                    },
                    "cochlea_params": {
                        "anf_num": (6, 2, 2),            # Example fiber counts (HSR, MSR, LSR)
                        "species": "human",
                        "cohc": 1.0,
                        "cihc": 1.0,
                        "powerlaw": "approximate",
                        "ffGn": False
                    },
                    "rng_seed": 42,
                    "omni_noise_level": 0,
                },
                "CI": {
                    "hrtf_params": {
                        "subj_number": 0,
                        "cue_to_apply": "HRTF",
                    },
                    "rng_seed": 42,
                    "omni_noise_level": 0,
                }  
            }
        )
    )


    # 1. --- Network Parameters  -----------------

    n_ANFs: int = NUM_CF * NUM_ANF_PER_HC  # Total ANFs in model
    @dataclass
    class POP_NUM:
        n_SBCs: int = 28000
        n_GBCs: int = 3600
        n_MNTBCs: int = 3600
        n_LNTBCs: int = 3600
        n_LSOs: int = 5600
        n_MSOs: int = 15500
        n_SPNs: int = 3600
    
    @dataclass
    class POP_CONV:
        ANFs2SBCs: int = 3
        ANFs2GBCs: int = 20
        GBCs2MNTBCs: int = 1
        GBCs2LNTBCs: int = 1 
        SBCs2LSOs: int = 40
        MNTBCs2LSOs: int = 8
        SBCs2MSOs: int = 3
        MNTBCs2MSOs: int = 2    
        LNTBCs2MSOs: int = 1
        MNTBCs2SPNs: int = 4  


    # 2. --- Synaptic Parameters  -----------------
    # ------------------------------------------------------------

    @dataclass
    class SYN_WEIGHTS:
        ANFs2SBCs: float = 16.0      
        ANFs2GBCs: float = 7.0 #high
        #      
        GBCs2LNTBCs: float = 5.0
        GBCs2MNTBCs: float = 30.0 #high
        #
        SBCs2LSO: float = 0.5 #tuned for a single spike   
        MNTBCs2LSO: float = -2.3 #tuned for a single spike
        #
        SBCs2MSO: float = 10.0 
        MNTBCs2MSO: float = -10.0 
        LNTBCs2MSO: float = 0 
        #
        MNTBCs2SPN: float = -40.0 

    @dataclass
    class SYN_DELAYS:
        ANFs2SBCs: float = 0.5
        ANFs2GBCs: float = 0.5
        #
        GBCs2MNTBCs: float = 0.5
        GBCs2LNTBCs: float = 0.5
        #
        SBCs2LSO: float = 2.0
        MNTBCs2LSO: float = 1.28 #
        #
        SBCs2MSOipsi: float = 2.0
        SBCs2MSOcontra: float = 2.0
        LNTBCs2MSO: float = 0.465 
        MNTBCs2MSO: float = 1.28
        #
        MNTBCs2SPN: float = 1.0 #0.11 integration time at MNTB/LNTB

    # 3. --- Neuronal Parameters  -----------------
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # --- Membrane capacitances (pF) ------------------------------
    # ------------------------------------------------------------
    @dataclass
    class MEMB_CAPS:
        SBC: float = 26.0 
        GBC: float = 13.0
        LNTBC: float = 12.0   
        MNTBC: float = 11.7
        LSO: float = 12.0
        MSO: float = 70.0
        SPN: float = 75.0

    # ------------------------------------------------------------
    # --- Leak conductances (nS) ---------------------------------
    # ------------------------------------------------------------
    @dataclass
    class G_LEAK:
        SBC: float = 43.3   # tau = 0.6 ms    or. 20.0   # tau = 1.3 ms
        GBC: float = 86.6   # tau = 0.15 ms
        LNTBC: float = 3 # tau = 4 ms  
        MNTBC: float = 5 # tau = 4 ms 
        LSO: float = 24 # tau = 0.5 ms
        MSO: float = 70 # tau = 1 ms   
        SPN: float = 75 # tau = 1 ms  

    # ------------------------------------------------------------
    # --- Leak reversal potentials (mV) ------------------------------
    # ------------------------------------------------------------
    @dataclass
    class E_L:
        SBC: float = -66.0 
        GBC: float = -61.0
        LNTBC: float = -60.0   
        MNTBC: float = -67.0
        LSO: float = -63.0
        MSO: float = -55.0
        SPN: float = -65.0
    
    # ------------------------------------------------------------
    # --- Reset potentials (mV) ------------------------------
    # ------------------------------------------------------------
    @dataclass 
    class V_RESET:
        SBC: float = -68.0   
        GBC: float = -63.0   
        LNTBC: float = -62.0
        MNTBC: float = -69.0
        LSO: float = -65.0
        MSO: float = -57.0
        SPN: float = -67.0

    # ------------------------------------------------------------
    # --- POPULATION-SPECIFIC THRESHOLDS ---------------------
    # ------------------------------------------------------------
    @dataclass
    class V_TH:
        SBC: float = -45.0     
        GBC: float = -45.0     
        LNTBC: float = -45.0     
        MNTBC: float = -45.0     
        MSO: float = -45.0     
        LSO: float = -45.0 
        SPN: float = -45.0    

    # ------------------------------------------------------------
    # --- POPULATION-SPECIFIC REFRACTORY PERIODS (ms) -------
    # ------------------------------------------------------------
    @dataclass
    class T_REF:
        SBC: float = 0.6       
        GBC: float = 0.6       
        LNTBC: float = 0.6     
        MNTBC: float = 0.6     
        MSO: float = 0.6        
        LSO: float = 0.6 
        SPN: float = 0.6   

    # ------------------------------------------------------------
    # ---  Excitatory reversal potentials (mV) --------------------------
    # ------------------------------------------------------------
    @dataclass
    class EXC_REV:
        SBC: float = 0 
        GBC: float = 0
        LNTBC: float = 0   
        MNTBC: float = 0
        LSO: float = 0
        MSO: float = 0
        SPN: float = 0

    # ------------------------------------------------------------
    # --- Synaptic time constants (ms) ----------------------------
    # ------------------------------------------------------------
    @dataclass
    class TAUS_EX_RISE:
        SBC: float = 0.2       
        GBC: float = 0.2       
        LNTBC: float = 0.25     
        MNTBC: float = 0.1  
        LSO: float = 0.5    
        MSO: float = 0.15
        SPN: float = 0.5        
    # ------------------------------------------------------------ 
    @dataclass
    class TAUS_EX_DECAY:
        SBC: float = 0.5       
        GBC: float = 0.5       
        LNTBC: float = 3.8     
        MNTBC: float = 0.2    
        LSO: float = 1.0
        MSO: float = 0.3
        SPN: float = 1.0
    
    # ------------------------------------------------------------
    # ---  Inhibitory reversal potentials (mV) --------------------------
    # ------------------------------------------------------------
    @dataclass
    class INH_REV:
        SBC: float = -75.0 
        GBC: float = -75.0
        LNTBC: float = -75.0   
        MNTBC: float = -75.0
        LSO: float = -75.0
        MSO: float = -75.0
        SPN: float = -20.0

    # ------------------------------------------------------------
    # --- Synaptic time constants (ms) ----------------------------
    # ------------------------------------------------------------
    @dataclass
    class TAUS_IN_RISE:
        SBC: float = 2.0   
        GBC: float = 2.0      
        LNTBC: float = 2.0   
        MNTBC: float = 2.0   
        LSO: float = 0.2   
        MSO: float = 0.2       
        SPN: float = 0.15  
    # ------------------------------------------------------------  
    @dataclass
    class TAUS_IN_DECAY:
        SBC: float = 2.0       
        GBC: float = 2.0       
        LNTBC: float = 2.0
        MNTBC: float = 2.0      
        LSO: float = 1.76
        MSO: float = 1.76  
        SPN: float = 0.7  
    # ------------------------------------------------------------   


    # ------------------------------------------------------------
    # --- System / kernel configuration ---------------------------
    # ------------------------------------------------------------
    @dataclass
    class CONFIG:
        STORE_POPS: set = field(default_factory=lambda: set([]))
        NEST_KERNEL_PARAMS: dict = field(
            default_factory=lambda: {
                "resolution": 0.01,
                "rng_seed": 42,
                "total_num_virtual_procs": 14,
                "local_num_threads": 14
            }
        )

    # ------------------------------------------------------------
    # --- Post-init to instantiate nested dataclasses -------------
    # ------------------------------------------------------------
    def __post_init__(self):
        self.CONFIG = self.CONFIG()
        self.SYN_DELAYS = self.SYN_DELAYS()
        self.SYN_WEIGHTS = self.SYN_WEIGHTS()
        self.POP_CONV = self.POP_CONV()
        self.TAUS_EX_RISE = self.TAUS_EX_RISE()
        self.TAUS_EX_DECAY = self.TAUS_EX_DECAY()
        self.TAUS_IN_RISE = self.TAUS_IN_RISE()
        self.TAUS_IN_DECAY = self.TAUS_IN_DECAY()
        self.MEMB_CAPS = self.MEMB_CAPS()
        self.G_LEAK = self.G_LEAK()

"""
iaf_cond_alpha default params
{'C_m': 250.0, -> always too big. try with a sensible 10pF
 'Ca': 0.0,
 'E_L': -70.0,
 'E_ex': 0.0,
 'E_in': -85.0,
 'I_e': 0.0,
 'V_m': -70.0,
 'V_reset': -60.0,
 'V_th': -55.0, -> maybe we can try -57... 'might help maintain selectivity for coincident inputs in MSO while still allowing LSO to respond to intensity differences'???
 'archiver_length': 0,
 'available': (0,),
 'beta_Ca': 0.001,
 'capacity': (0,),
 'dg_ex': 0.0,
 'dg_in': 0.0,
 'element_type': 'neuron',
 'elementsize': 688,
 'frozen': False,
 'g_L': 16.6667, -> try 166.67 to compensate for higher C_m
 'g_ex': 0.0,
 'g_in': 0.0,
 'global_id': 0,
 'instantiations': (0,),
 'local': True,
 'model': 'iaf_cond_alpha',
 'model_id': 33,
 'node_uses_wfr': False,
 'post_trace': 0.0,
 'recordables': ('g_ex', 'g_in', 't_ref_remaining', 'V_m'),
 'synaptic_elements': {},
 't_ref': 2.0,
 't_spike': -1.0,
 'tau_Ca': 10000.0,
 'tau_minus': 20.0,
 'tau_minus_triplet': 110.0,
 'tau_syn_ex': 0.2,
 'tau_syn_in': 2.0,
 'thread': -1,
 'thread_local_id': -1,
 'type_id': 'iaf_cond_alpha',
 'vp': -1}
"""
