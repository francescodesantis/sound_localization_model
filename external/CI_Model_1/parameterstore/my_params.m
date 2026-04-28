%% Parameters for the CI_AN-Modell
ANParams.N_nervecells =[];          % number of auditory nerve cells
ANParams.X_EL =[];                         % position of the electrodes    
ANParams.TAUCHR =[];                     % chronaxie of the cell membrane
ANParams.EFFIRHEO =[];                 % rheobase of the cell memebrane
% global IREF;                         % reference current, BW: Not used anymore except for process_neural_model.m, which is not used anymore, too
% it is more like an in between variable, which calculates other
% parameters. It is itself calculated by other parameters
ANParams.V =[];                               % spatial spread function
ANParams.X_NZ=[];                         % position of the nerve cells
ANParams.ML50=[];                         % constant for the latencyx
ANParams.SIGMAL50=[];                 % constant for the jitter
ANParams.MT_ARP=[];                     % constant for the absolute refractory period
ANParams.MTAU_RRP=[];                 % constant for the relative refractory period
ANParams.RS0_ind=[];                   % relative spread indenpendent from the phase duration
% global tauM;                         % membrane time constant; BW: Not
% used anymore, is just an intermediate temporary variable for this file.
% Endparameter is ANPArams.R
ANParams.R=[];                               % membrane resistance
ANParams.Uth=[];                           % threshold 
ANParams.xGroup =[];                     % group limits
ANParams.indexGroup=[];             % index groups-> how many nerve cells are grouped in central auditory processing such that 46 "channels" come out
ANParams.Hd=[];                             % 2nd lowpass of central auditory processing
ANParams.hTP1=[];                         % 1rst lowpass of central auditory processing
ANParams.fsZAS=[];                       % sampling frequency of central auditory processing
% global Tph;                                     % phase duration
% global CSR;							% current stimulation rate [pulses
% per second] for auditory nerve model; BW: just used in ECAP-Sequence and
% reproduce-scripts
rand('twister',sum(100*clock))                  % init noise generator
randn('state',sum(100*clock))                  % init noise generator


%% Three CIParams are needed:
CIParams.TCL    = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]';
CIParams.Tph    = 25e-6; % rectangular pulse phase duration [s]
if ~isfield(CIParams, 'devicename')
    CIParams.devicename = 'nucleus'; % Needed for conversion from Clinical Level(CIParams.TCL) to Current 
end

%% Set parameters for auditory nerve model
ANParams.N_nervecells = 35000;

% distribution of the neurons along the basilar membrane
x_min = (1/2.1) * log10(0    / 165.4 + 1.0);  % but this gives -inf at f=0!
CFMIN = 125;    % Hz
CFMAX = 20000; % Hz

x_min = (1/2.1) * log10(CFMIN/165.4 + 1.0);
x_max = (1/2.1) * log10(CFMAX/165.4 + 1.0);
x_norm = linspace(x_min, x_max, ANParams.N_nervecells);  % normalized 0-1
cf_array = 165.4 * (10.^(2.1 * x_norm) - 1.0);          % Hz

% convert normalized position to mm for the spatial spread calculation
ANParams.X_NZ = x_norm * 35;

%ANParams.X_NZ = linspace(0,35,ANParams.N_nervecells);
ANParams.X_NZ = ANParams.X_NZ(:);
     
% position of electrodes in the cochlea in mm
ANParams.X_EL = 8.125:0.75:23.875;    

% spatial spread function
ANParams.lambda = 9;%9; % parameter that controls the spatial spread
ANParams.v0 = 1;%0.85;  % another parameter that controls the spatial spread
ANParams.V = calculate_AVF(ANParams.X_EL,ANParams.X_NZ,ANParams.lambda,ANParams.v0);
        
%%% auditory nerve cells
% chronaxie and rheobase
[ANParams.TAUCHR,ANParams.EFFIRHEO] = generateChronaxieRheobase(ANParams.N_nervecells); 
% relative spread of the membrane noise
ANParams.RS0_ind = calculateRelativeSpread(ANParams.N_nervecells);
% refractory constants T_ARP, TAU_RRP
[ANParams.MT_ARP,ANParams.MTAU_RRP] = generateRefractoryConstants(ANParams.N_nervecells);
% Neural latency
[ANParams.ML50,ANParams.SIGMAL50] = generateNeuralLatencyConstants(ANParams.N_nervecells);


%%% central auditory processing
ANParams.fsZAS = 5000;
[ANParams.xGroup, ANParams.indexGroup] = generateNeuralGroups(ANParams.X_NZ,ANParams.X_EL);
% 1. Lowpass ZAS
ANParams.hTP1 = generateImpRespPostMasking(ANParams.fsZAS);
% 2. Lowpass ZAS
ANParams.Hd = generateFilterObjLPMasking_2;

ANParams.Ith0 = conversion_CL2Iamp(CIParams.TCL,CIParams.devicename);
% calculate the membrane time constant tauM and the resistance of the
% membrane R. 

tauM = ANParams.TAUCHR/log(2);  % the membrane time constant     (eq. 6.10)
% I assume that C never occur in the thesis anymore, so I set it to one!
C     = 1;
ANParams.R    = tauM/C;         % the resistance of the membrane (eq. 6.10)

% reference current
% calculates the deteministic current for threshold for a given pulse
% with a defined phase CIParams.Tph
ANParams.IREF = calculateDeteminsticThresholdCurrent_2(mean(ANParams.EFFIRHEO),mean(ANParams.TAUCHR/log(2)),CIParams.Tph);


% hier empirische anpassung des CIs ans Hörnervenmodell, dabei soll die
% Höhe von V so variiert werden, dass SR, ca, den Wert von 30 annimmt.
% Entspricht in etwa 30 APs. 
for ii = 1:length(ANParams.Ith0)
    ANParams.V(:,ii) = ANParams.V(:,ii).*ANParams.IREF/ANParams.Ith0(ii,:);
end
ANParams.Uth  = ANParams.EFFIRHEO .* ANParams.R;