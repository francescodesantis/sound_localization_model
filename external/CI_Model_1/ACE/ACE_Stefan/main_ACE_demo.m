% main_ACE_demo
function [mINew,pps,maxima] = main_ACE_demo(signal,fsOrg)

%clear
% close all
%clc

global FS
global NFFT
global W
global Tlevel
global Clevel
global pps
global maxima
global vActiveElectrodes
global NUMOFCHANNELS


set_global_constants_ACE_demo % globale daten aufrufen

W = hann(NFFT);

% sinus signal generieren
% f     = 2000;
% fsOrg = 44100; 
% T     = 1;
% n     = 0:T*fsOrg-1;
% t     = n/fsOrg;
% 
% signal = sin(2*pi*f*t);
% 
% 
% [signal,sfsignal] = wavread('S02M_L005_V6_M1_N2_CS0.wav');
% signal = resample(signal,fsOrg,sfsignal);
% 
% signal = signal(:);
% signal = sqrt(2) * signal * 10^(-17/20); % Kalibrierung auf -17 dB



% Warum -17 dB? Wenn man sich die Kompressionskennlinie anschaut, erkennt
% man, dass oberhalb von -5 dB ein Limiter auftritt und unterhalb von -35
% ein Noise Gate. Zwischen -35 und -5 wird komprimiert. Unter der Annahme,
% dass Sprache einen Dynamikbereich von -18 und +12 dB um den Langzeitpegel
% hat, wird dieser genau in die Mitte dieser Kompressionkennlinie gesetzt,
% also bei -17 dB. 
% Da mir leider nicht bekannt ist, wie der Pegel am Eingang des CIs ist,
% wird die Pegelanpassung anhand dieser Kompressionskennlinie vorgenommen. 
if FS ~= fsOrg
    signal = resample(signal,FS,fsOrg);
end
signal = process_preemphasis_CI(signal); % Hochpass

index     = 1:NFFT;
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

switch pps
    % muss zyklisch abgetastet werden
    case 900
        % vorschub = [18 17 18 18 18 17 18 18 18];  % wegen rundungsfehler
        vorschub = [18 18 17 18 18 18 17 18 18];  % wegen rundungsfehler
    otherwise
        vorschub = round(FS/pps); 
        disp('Warnung: Rundungsfehler möglich!');
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
        x = [zeros(NFFT-length(index2),1); signal(index2)];
    else
        x = signal(index);
    end

    % windowing
    x = W.*x;

    % spectra
    S = abs(fft(x,NFFT));

    % "envelope" per time unit
    F = process_ACE_filterbank_demo(S);

    % choose n of m
    [A,iElectrode] = process_nOfm(F,maxima);
    Aplot = [Aplot A];
    % hier wird berechnet, wann zu welchen zeitpunkten die elektroden
    % stimulieren. 
    mT(:,iSequence) = generatePulseTrain_2(iElectrode,(index(1)-1)/FS,pps);

 
    C = process_compression_ci(A,B,M,alpha_c);
    Cplot = [Cplot C];
    % compression between Tlevels and Clevels
    cl = zeros(length(vActiveElectrodes),1);
    
    logicalNoPulse = C(iElectrode) == 0;
    
    cl(iElectrode,:) = round(Tlevel(iElectrode,:) + (Clevel(iElectrode,:)-Tlevel(iElectrode,:)).*C(iElectrode,:));
    
    % current amplitude in micro-Ampere
    I = zeros(length(vActiveElectrodes),1);
    
    I(iElectrode,:) = 10*175.^(cl(iElectrode,:)/255); %TJ: verstehe ich als Art Kalibrierung von CL auf µA, agrees with formula for 'nucleus' in conversion_CL2Iamp
    %I(iElectrode,:) = cl(iElectrode,:);
    
    I(iElectrode(logicalNoPulse)) = 0;
    
    mI(:,iSequence) = I; % microAmpere!!!!

    index     = index + vorschub(indexVorschub); % vorschub
    iSequence = iSequence + 1;

    if indexVorschub == length(vorschub)
        indexVorschub = 0;
    end
end

vT = [];
mINew = zeros(NUMOFCHANNELS,length(find(mT~=Inf)));
indexTime = 1;
for tt = 1:size(mT,2)
    vT_OneCycle = mT(:,tt);
    index = find(vT_OneCycle~=Inf); % stimulation electrodes
    index = index(end:-1:1);    % keine gute lösung, in diesem fall gehts!
    vT = [vT (vT_OneCycle(index)')];
    for ii = 1:length(index)
        mINew(index(ii),indexTime) = mI(index(ii),tt);
        indexTime = indexTime + 1;
    end
end

%%%% DISPLAY with biphasic pulses %%%% THIS BLOCK IS NOT NECESSARY FOR THE
%%%% MODEL BUT IT IS NICE TO GET A GOOD DISPLAY
if display_biphasic_pulses
    sfreq_pulsepattern = 1e6;  %usually 1MHz, to show the µs timing of the pulses
    
    for iCounter = 1:size(mINew,1)
        pulsetrain(iCounter,:) = pulsetrain_fordisplay(mINew(iCounter,:),pps,maxima,sfreq_pulsepattern);
    end
    time_vector_pulsepattern = [0:1/sfreq_pulsepattern:(size(pulsetrain,2)-1)/sfreq_pulsepattern].*1000; %in ms;
    verticaldistance = max(max(mINew))*1.1;
    plotposition = 0;
    %choose a location with high energy to plot
    [tmp, tmpindex] = max(mean(mINew,1));
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
end
