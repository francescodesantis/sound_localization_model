function [IR, vTiDiscret,meanxGroup, SR, Zn, logicANpattern] = main_CI_AN_model(Iamp,i_el,tp,Tph,N_nervecells,xGroup,fsZAS,indexGroup,hTP1,Hd,ANParams)
%clear
%close all
%clc

% addpath(['.' filesep 'parameterstore']);
% addpath(['.' filesep 'ACE']);
% addpath(['.' filesep 'wavfilestore']);
% 
% % globale variablen laden
% devicename = 'nucleus';
% set_global_constants
% 
% %fsOrg                    = 44100;
% 
% %%% noise signal
% % signalOrg                = randn(fsOrg*0.05,1)/4;   
% %%% sine wave
% %  t         = 0:1/fsOrg:0.025-(1/fsOrg);
% %  signalOrg = 0.1*sin(2*pi*7000*t);signalOrg = signalOrg(:);
% % 
% % 
% %  signal                   = resample(signalOrg,CIParams.FS_ACE,fsOrg);     
% %  [AI,CFs,BWs,tp,Icl,i_el] = ACE_FilterBank_patients(signal, 1, N, CSR, CIParams.THL, CIParams.MCL, CIParams.Vol,CIParams.T_SPL,CIParams.C_SPL);
% %  tp  = tp';     % Zeitpunkt des elektrischen Stimulus
% %  Icl = Icl';    % Stromamplitude in CL
% %  i_el = i_el';  % aktive Elektrode
% 
% %%%%% REFERENCE CURRENT STIMULUS %%%%%%
% % N_electrodes = 1;
% % CSR  = 900;
% % TSR  = CSR*N_electrodes;
% % tp   = 0:1/CSR:99/CSR;
% % Icl  = ones(size(tp))*200;
% % i_el = ones(size(tp)) * 11;
% % Iamp = conversion_CL2Iamp(Icl,devicename);
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% % Tph  = ones(size(tp))*Tph; % hier die Phasendauer der elektrischen Stimuli ist
% % Iamp = conversion_CL2Iamp(Icl,'freedom'); 
% % Iamp(Icl==-2) = 0;  % Achtung, wenn kein elektrischer Puls generiert wird, dann wird die Amplitude auf null gesetzt
% % i_el(Icl==-2) = 1;  % dummy
% 
% %sinus signal generieren
% %    f     = 1000;
% %    fsOrg = 44100; 
% %    T     = 1;
% %    n     = 0:T*fsOrg-1;
% %    t     = n/fsOrg;
% %   
% %    signal = sin(2*pi*f*t);%.*10^(-40/20);
% % 
%  [signal,fsOrg] = wavread('S02M_L005_V6_M1_N2_CS0.wav');
%  %signal = resample(signal,fsOrg,sfsignal);
%  signal = signal(:);
% 
% %%% ATTENTION THIS LINE PUTS THE SIGNAL TO RMS = 1. IS THAT WANTED?
% % for speech, this is reasonable to put it optimally into the 30 dB that
% % the listener has as input dynamic range.
% signal = signal./sqrt(mean(signal.^2));
% signal = signal * 10^(-17/20); % Kalibrierung auf -17 dB


% Warum -17 dB? Wenn man sich die Kompressionskennlinie anschaut, erkennt
% man, dass oberhalb von -5 dB ein Limiter auftritt und unterhalb von -35
% ein Noise Gate. Zwischen -35 und -5 wird komprimiert. Unter der Annahme,
% dass Sprache einen Dynamikbereich von -18 und +12 dB um den Langzeitpegel
% hat, wird dieser genau in die Mitte dieser Kompressionkennlinie gesetzt,
% also bei -17 dB. 
% Da mir leider nicht bekannt ist, wie der Pegel am Eingang des CIs ist,
% wird die Pegelanpassung anhand dieser Kompressionskennlinie vorgenommen. 

% %ACE STRATEGY
% [mINew,pps,maxima] = ACE_signal_processing(signal,fsOrg,CIParams.THL,CIParams.MCL,devicename);
% tp   = 0:1/pps/maxima:(size(mINew,2)-1)/pps/maxima; %time vector
% figure,imagesc(tp,[],mINew)
% axis xy;
% xlabel('time (s)')
% ylabel('electrode number')
% %This translates the stimulation pattern to a form readable for the CI-AN
% %model
% [Iamp,i_el] = max(mINew,[],1); %�A   %this is a little tricky, because here, no stimulation results in electrode number 1 as being selected,
% %however, it should be ok, since the amplitude 0 is selected then...
% %Iamp = Iamp.*1e-0;  %A  here is a factor 1000 that I don't yet understand!


Tph = ones(size(tp)).*Tph;


%%%% calculate action potentials
% Init values
numberCycle = size(tp,2);
tLAP   = ones(N_nervecells,1)*-99/1000;  % arbtrary init value for the last action potential
tAPOut = ones(N_nervecells,1)*-99/1000;
tAPOut = [tAPOut zeros(N_nervecells,numberCycle)];
indexKeepRefracValues = [];         % init value for keeping the refractory values
refValues             = zeros(N_nervecells,2); % refractory values for keeping or not...

% prepare membrane noise
UN = membraneNoise(N_nervecells,1/tp(2),numberCycle); %1/tp(2) is the sampling rate of the stimulation pattern
UN = UN(1:N_nervecells,:);
%%% membrane noise with a length of 1 second
% load('noise_CSR900_N8_1sec');
APvec = [];
% Berechnung der Aktionspotentiale
for iPulse = 1:numberCycle
    Si = [tp(:,iPulse),Iamp(:,iPulse),Tph(:,iPulse),i_el(:,iPulse)]; %just a carrier, which electrode, phase etc.
    [tAP,indexKeepRefracValues,refValues,P_AP,index_AP] = auditoryNerve(Si,UN(:,iPulse),tLAP,indexKeepRefracValues,refValues,ANParams);
    APvec(end+1:end+length(index_AP),:)= [index_AP' tAP(index_AP)]; %line from Mathias [#nervecell spiketime]
    tAPOut(:,iPulse+1) = tAP;
    tLAP   = tAP(:,end);  %save last AP in vector tLAP
end
APvec = APvec(isfinite(APvec(:,2)), :);
logicANpattern = APtimes2logic(APvec,N_nervecells);
clear APvec

% central auditory processing
deltaAP    = diff(tAPOut,[],2);
meanxGroup = mean(xGroup,2);
isAP         = (deltaAP>0);
tAPOut(~isfinite(tAPOut)) = 0;
vTiDiscret = 0:1/fsZAS:((round(max(max(tAPOut))*fsZAS)/fsZAS)+200/fsZAS);
vTiDiscret   = 0:1/fsZAS:((round(max(max(tAPOut))*fsZAS)/fsZAS)+200/fsZAS); %time vector for internal rep
mtLAPGrouped = zeros(size(indexGroup,1),length(vTiDiscret));
convolution  = zeros(size(xGroup,1),length(hTP1)+length(vTiDiscret)-1); 
for iGroup = 1:size(indexGroup,1)  %loop over each group of nerve cells / no overlap
    for ii = indexGroup(iGroup,1):indexGroup(iGroup,2)  %loop over each nerve cell within a group
        isAPOneNeuron = find(isAP(ii,:)==1) + 1;
        actionPotentials = round(tAPOut(ii,isAPOneNeuron)*fsZAS)/fsZAS;  %times of APs in ms
                
        if ~isempty(actionPotentials)
            for jj = 1:length(actionPotentials)  %loop for every found AP
                [mini,indexAP_oneNeuron] = min(abs(vTiDiscret-actionPotentials(jj))); %matches spiking times to times of (low-sampled) IR
                mtLAPGrouped(iGroup,indexAP_oneNeuron) = mtLAPGrouped(iGroup,indexAP_oneNeuron) + 1; %simple addition
            end
        end
    end
    if any(mtLAPGrouped(iGroup,:)) == 1
        convolution(iGroup,:) = conv(mtLAPGrouped(iGroup,:),hTP1); %postmasking - convolution with Gaussian
    end
end

% SR ist die kurzzeit-gemittelte Gruppenentladungsrate. entspricht in etwa
% der anzahl der APs. 
SR      = convolution(:,round(size(hTP1,2)/2)+1:end-round(size(hTP1,2)/2));
mean(sum(SR,1)) %this is needed in order to calibrate the model:: SR(k) in the text
[Yn,Zn] = processIntegrate(SR,fsZAS); %forward masking simulation
IR = zeros(size(Yn));
for ii = 1:size(Yn,1)
    IR(ii,:) = filter(Hd,Yn(ii,:));  %elliptic lowpass with 5000 Hz cutoff frequency
end


