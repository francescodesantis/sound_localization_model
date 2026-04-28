function [mINew,logicANpattern,IR]  = main_SP_CI_AN(signalfilename, paramfilename, varargin)

% main_SP_CI_AN script

%% display progress bar
progressbar = waitbar(0,'Starting processing');
% globale variablen laden
CIParams.devicename = 'nucleus';
% get parameter file,
% set_global_constants
run(paramfilename);

% Read signal
[signal,fsOrg] = audioread(signalfilename);

% Make signal mono signal
 signal = signal(:);

%%% ATTENTION THIS LINE PUTS THE SIGNAL TO RMS = 1. IS THAT WANTED?
% for speech, this is reasonable to put it optimally into the 30 dB that
% the listener has as input dynamic range.
signal = signal./sqrt(mean(signal.^2));
signal = signal * 10^(-17/20); % Kalibrierung auf -17 dB

% Warum -17 dB? Wenn man sich die Kompressionskennlinie anschaut, erkennt
% man, dass oberhalb von -5 dB ein Limiter auftritt und unterhalb von -35
% ein Noise Gate. Zwischen -35 und -5 wird komprimiert. Unter der Annahme,
% dass Sprache einen Dynamikbereich von -18 und +12 dB um den Langzeitpegel
% hat, wird dieser genau in die Mitte dieser Kompressionkennlinie gesetzt,
% also bei -17 dB. 
% Da mir leider nicht bekannt ist, wie der Pegel am Eingang des CIs ist,
% wird die Pegelanpassung anhand dieser Kompressionskennlinie vorgenommen. 
%% Plot of the acoustic signal
t_acoustic = [0:1/fsOrg:(length(signal)-1)/fsOrg];
figure, plot(t_acoustic,signal);
xlabel('Time (s)');
ylabel('Amplitude');
title('Waveform of the unprocessed Signal');

waitbar(0.2,progressbar, 'Simulate CI processing');
%% CI signal processing
%%ACE STRATEGY
[mINew,CIParams.pps,maxima] = ACE_signal_processing(signal,fsOrg,CIParams);
tp   = 0:1/CIParams.pps/maxima:(size(mINew,2)-1)/CIParams.pps/maxima; %time vector
figure,imagesc(tp,[],mINew)
axis xy;
xlabel('time (s)')
ylabel('electrode number')
title('Electrode stimulation pattern');
%This translates the stimulation pattern to a form readable for the CI-AN
%model
[Iamp,i_el] = max(mINew,[],1); %µA   %this is a little tricky, because here, no stimulation results in electrode number 1 as being selected,
%however, it should be ok, since the amplitude 0 is selected then...
%Iamp = Iamp.*1e-0;  %A  here is a factor 1000 that I don't yet understand!


waitbar(0.4,progressbar, 'Simulate Auditory nerve');
%% CI-AN-model
[IR, vTiDiscret, meanxGroup,~,~,logicANpattern] = main_CI_AN_model(Iamp,i_el,tp,CIParams.Tph,ANParams.N_nervecells,ANParams.xGroup,ANParams.fsZAS,ANParams.indexGroup,ANParams.hTP1,ANParams.Hd,ANParams);

%% plot spiking pattern, same sampling frequency as the stimulation
tAN   = 0:1/10000:(size(logicANpattern,2)-1)/10000; %time vector, sampling rate of the AN is 10000.
figure, imagesc(tAN,[0 36],~logicANpattern);
xlabel('Time (s)');
ylabel('Cochlear location (mm)');
texthandle = text(0,1,'Apex');
texthandle = text(0,34,'Base');
colormap gray
axis xy;
title('Spiking pattern of auditory nerve cells');
waitbar(0.7,progressbar, 'Plot Internal Representation');
%% Plot IR
figure,imagesc(vTiDiscret,meanxGroup(:,1),IR);
set(gca,'ydir','normal')
set(gca,'FontSize',14)
xlabel('Time (s)');
ylabel('Cochlear location (mm)');
title('Internal representation after auditory nerve modeling');
ylim([0 35])
texthandle = text(0,1,'Apex');
set(texthandle,'Color','white');
texthandle = text(0,34,'Base');
set(texthandle,'Color','white');
colorbar;
disp('finished')
% update Progressbar
waitbar(1,progressbar, 'Finished');
delete(progressbar);
