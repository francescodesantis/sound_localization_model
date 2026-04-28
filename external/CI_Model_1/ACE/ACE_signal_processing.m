function [mINew,pps,maxima] = ACE_signal_processing(signal,fsOrg,CIParams)
%% main_ACE_demo
% Input Parameters: signal = Signal, must be one column vector
% fsOrg = sampling frequency [HZ] of the Signal
% CIParams = struct with all Parameters to simulate a Cochlear Implant.
% This parameter ist optional, if empty it will use the default values in
% set_global_constants_ACE_Demo.m

CIParams = set_global_constants_ACE_demo(CIParams) % Set all needed CIParameters

% Check for any errors in the ACE-Parameters:
Check_CI_Parameters; 

W = hann(CIParams.NFFT);
Tlevel = CIParams.TCL;
Clevel = CIParams.MCL;

% 
% 
% [signal,sfsignal] = wavread('S02M_L005_V6_M1_N2_CS0.wav');
% signal = resample(signal,fsOrg,sfsignal);
% 
% signal = signal(:);
% signal = sqrt(2) * signal * 10^(-17/20); % Kalibrierung auf -17 dB



if CIParams.FS_ACE ~= fsOrg
    signal = resample(signal,CIParams.FS_ACE,fsOrg);
end
signal = process_preemphasis_CI(signal, CIParams.FS_ACE); % Hochpass

index     = 1:CIParams.NFFT;
lenSignal = length(signal);

iSequence = 1;
mT = [];
x = [];
S = [];
F = [];
A = [];
iElectrode = [];
C  = [];
cl = [];
I  = [];
mI = [];

switch CIParams.pps
    % muss zyklisch abgetastet werden
    case 900
        % vorschub = [18 17 18 18 18 17 18 18 18];  % wegen rundungsfehler
        vorschub = [18 18 17 18 18 18 17 18 18];  % wegen rundungsfehler (16000/900 = 17.7778; mean(vorschub) = 17.7778)
    otherwise
        vorschub = round(CIParams.FS_ACE/CIParams.pps); 
        disp('Warnung: Rundungsfehler möglich, da Abtastrate des Signales kein ganzzahliges Vielfaches der PPS ist!');
end

indexVorschub = 0;

Aplot = [];
Cplot = [];

while index(1) < lenSignal
    indexVorschub = indexVorschub + 1;
    if index(end) > lenSignal
        x = [signal(index(1):end); zeros(index(end) - lenSignal,1)];
    elseif index(1) <= 0
        index2 = index(index>0);
        x = [zeros(CIParams.NFFT-length(index2),1); signal(index2)];
    else
        x = signal(index);
    end

    % windowing
    x = W.*x;

    % spectra
    S = abs(fft(x,CIParams.NFFT));

    % "envelope" per time unit
    F = process_ACE_filterbank_demo(S, CIParams);

    % choose n of m
    [A,iElectrode] = process_nOfm(F,CIParams.maxima);
    Aplot = [Aplot A];
    % hier wird berechnet, wann zu welchen zeitpunkten die elektroden
    % stimulieren. 
    mT(:,iSequence) = generatePulseTrain_2(iElectrode,(index(1)-1)/CIParams.FS_ACE,CIParams.pps, CIParams);

    C = process_compression_ci(A,CIParams.B,CIParams.M,CIParams.alpha_c);
    Cplot = [Cplot C];
    % compression between Tlevels and Clevels
    cl = zeros(length(CIParams.vActiveElectrodes),1);
    
    logicalNoPulse = C(iElectrode) == 0;
    
    cl(iElectrode,:) = round(Tlevel(iElectrode,:) + (Clevel(iElectrode,:)-Tlevel(iElectrode,:)).*C(iElectrode,:));
    
    % current amplitude in micro-Ampere
    I = zeros(length(CIParams.vActiveElectrodes),1);
    
    %I(iElectrode,:) = 10.175.^(cl(iElectrode,:)/255); %TJ: verstehe ich als Art Kalibrierung von CL auf µA, agrees with formula for 'nucleus' in conversion_CL2Iamp, Formel von Laneau Diss.
    I(iElectrode,:) = conversion_CL2Iamp(cl(iElectrode,:),CIParams.devicename);
    %I(iElectrode,:) = cl(iElectrode,:);
    
    I(iElectrode(logicalNoPulse)) = 0;
    
    mI(:,iSequence) = I; % microAmpere!!!!

    index     = index + vorschub(indexVorschub); % vorschub
    iSequence = iSequence + 1;

    if indexVorschub == length(vorschub)
        indexVorschub = 0;
    end
end

%% Create Matrix for sequentiell electrode stimulation
% mI contains the current for each electrode per NFFT-Block. 
% If we want to stimulate the electrodes one after the other during one
% NFFT-block (sequentiell stimulation), each NFFT-Block must be divided in 
% n- of m- subblock, so that in the end, we have only one active electrode
% in each time-frame. If we would use mI directly, we would stimulate all
% n- of m-Electrodes at the same time. 
vT = [];
mINew = zeros(CIParams.NumOfChannels,length(find(mT~=Inf)));
indexTime = 1;
for tt = 1:size(mT,2)
    vT_OneCycle = mT(:,tt);
    index = find(vT_OneCycle~=Inf); %get numbers of active stimulation electrodes
    index = index(end:-1:1);    % keine gute lösung, in diesem fall gehts!
    vT = [vT (vT_OneCycle(index)')];
    for ii = 1:length(index)
        mINew(index(ii),indexTime) = mI(index(ii),tt);
        indexTime = indexTime + 1;
    end
end

%%%% DISPLAY with biphasic pulses %%%% THIS BLOCK IS NOT NECESSARY FOR THE
%%%% MODEL BUT IT IS NICE TO GET A GOOD DISPLAY
if CIParams.display_biphasic_pulses
    sfreq_pulsepattern = 1e6;  %usually 1MHz, to show the µs timing of the pulses
    
    for iCounter = 1:size(mINew,1)
        pulsetrain(iCounter,:) = pulsetrain_fordisplay(mINew(iCounter,:),CIParams.pps,CIParams.maxima,sfreq_pulsepattern,CIParams);
    end
    time_vector_pulsepattern = [0:1/sfreq_pulsepattern:(size(pulsetrain,2)-1)/sfreq_pulsepattern].*1000; %in ms;
    maximalamplitude = max(max(mINew));
    verticaldistance = maximalamplitude*1.1;
    plotposition = 0;
    %choose a location with high energy to plot
    [~, tmpindex] = max(mean(mINew,1));
    index_in_pulsetrain = round(tmpindex/size(mINew,2)*size(pulsetrain,2));
    %choose an 10ms inlet around the location of the highest energy to display
    startindex = round(max([1 index_in_pulsetrain-0.005*size(pulsetrain,2)]));
    endindex = round(startindex+0.01*size(pulsetrain,2));
    figure;
    for iCounter = 1:size(pulsetrain,1)
        plot(time_vector_pulsepattern(startindex:endindex), ...
            pulsetrain(iCounter,startindex:endindex)+plotposition);
        plotposition = plotposition + verticaldistance;
        hold on;
    end
    xlabel('Time (ms)')
    set(gca,'YTick',[0:2*verticaldistance:(size(pulsetrain,1)-1)*verticaldistance]);
    set(gca,'YTickLabel',{[1:2:21]});
    set(gca,'YLim',[-verticaldistance size(pulsetrain,1)*verticaldistance])
    ylabel('Electrode number');
    text(time_vector_pulsepattern(startindex+1),1,sprintf('maxAmpl.: %0.2f uA',maximalamplitude));
end

%% Ugly workaround (because pps must be returned):
pps = CIParams.pps;
maxima = CIParams.maxima;
