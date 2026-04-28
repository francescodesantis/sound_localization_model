%function set_global_constants_ACE_demo

global FS               , FS   = [];          % sampling frequency
global NFFT             , NFFT = [];          % fft length
global W                , W = [];             % hanning window
global G                , G = [];             % weights of the ACE filterbank
global NUMOFCHANNELS    , NUMOFCHANNELS = []; % number of channels
global Q_SUM            , Q_SUM = [];         % weights to summarize FFT-bins
global Tlevel           , Tlevel = [];        % threshold level
global Clevel           , Clevel = [];        % comfortable level
global indexOrg         , indexOrg = [];      
global pps              , pps = [];           % stimulation rate in pulses per second
global maxima           , maxima = [];        % maximal number of electrodes in n-of-m strategy
global Tph              , Tph = [];           % Duration of the phase of a pulse
global ipg              , ipg = [];           % Inter-phase-gap
global vActiveElectrodes, vActiveElectrodes = [];  %assortment of active electrodes

% sampling frequency 
FS = 16000;

% fft length
NFFT = 128;

% hanning window
W    = generate_hanning_window(NFFT);

% calculate the weights for the fft 
W1 = abs(fft(W));
W2 = abs(fft([W;zeros(NFFT,1)]));  % for G(2), which is derive from W(0.5)

G(1) = 2/W1(1);                 % for channels with 1 bin
G(2) = 2/(sqrt(2)*W2(2));       % for channels with 2 bins
G(3) = 2/78.38;                 % for channels with more than 2 bins.

clear W1;
clear W2;

vActiveElectrodes = [22 21 20 19 18 17 16 15 14 13 12 11 10  9  8  7  6  5  4  3  2  1]';

%Tlevel=THL;%[31;32;31;30;34; 36;34;31;31;54; 43;34;45;43;45; 36;34;36;37;38; 30;31];
%Clevel=MCL;%[91;92;91;90;94; 86;84;83;88;84; 91;94;95;93;95; 86;88;76;67;58; 80;91];

%compression characteristics
B = 0.0156; %base level
M = 0.5859; %saturation level
alpha_c = 415.96; % controls the steepness of the compression function

% Tlevel            = [50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50 50]';
% Clevel            = [99 99 99 99 99 99 99 99 99 99 99 99 99 99 99 99 99 99 99 99 99 99]';
Tph               = 25e-6; % Phasendauer eines Pulses
vGain             = zeros(size(vActiveElectrodes));

vLowerFreq = [ 188  313  438  563  688  813  938 1063 1188 1313 1563 1813 2063 2313 2688 3063 3563 4063 4688 5313 6063 6938]';
vUpperFreq = [ 313  438  563  688  813  938 1063 1188 1313 1563 1813 2063 2313 2688 3063 3563 4063 4688 5313 6063 6938 7938]';
pps        = 900;
maxima     = 8;   %maximal number of channels to stimulate
ipg        = 8e-006; 

% information, how to sum up the frequency bins
Q_SUM = [  ones(9,1); ...   
         2*ones(4,1); ...
         3*ones(2,1); ...
         4*ones(2,1); ...
         5*ones(2,1); 
         6;7;8];
indexOrg = 2;

% number of channels
NUMOFCHANNELS = length(vActiveElectrodes(~isnan(vActiveElectrodes)));  

% display of biphasic pulses
display_biphasic_pulses =1;% 1;
