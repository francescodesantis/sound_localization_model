% figureS4reproduce

% global V
% global R
% global TAUCHR
% global EFFIRHEO
% global Uth
% global RS0_ind;
% global MT_ARP;
% global MTAU_RRP;
% global N_nervecells;
% global ML50;
% global SIGMAL50;


ANParams.MT_ARP = 0.5e-3; %s                     % constant for the absolute refractory period
ANParams.MTAU_RRP = 1.7e-3; %s
ANParams.ML50 = 0.607e-3;
ANParams.SIGMAL50 = 0.106e-3;
ANParams.EFFIRHEO = 100e-6;
ANParams.TAUCHR = 0.25e-3;
ANParams.RS0_ind = 0.1153;
ANParams.V = 1;
ANParams.C = 1;
ANParams.N_nervecells = 1;
tauM = ANParams.TAUCHR/log(2);  % the membrane time constant     (eq. 6.10)
ANParams.R    = tauM/ANParams.C;         % the resistance of the membrane (eq. 6.10)

ANParams.Uth  = ANParams.EFFIRHEO .* ANParams.R;


%tLAP   = ones(N_nervecells,1)*-99/1000;  % arbitrary init value for the last action potentialfor jCounter = 1:length(CSR) %different stimulation rates
CSR = [200 400 600 800 1000 1200 1500 1800];
%Threshold is dependent on stimulation rate (from Fig. 3)
IdB = [54 56 58 63 65 67 69 72];
I = 10.^(IdB./20).*1e-6; %current in A 

plotposition = [1 3 5 7 2 4 6 8]; %the positions in the plot to match it with Fredelake and Hohmann 2012 - Suppl.Mat.
figure;
for jCounter = 1:length(CSR) %different stimulation rates
    tp = 0:1/CSR(jCounter):1999/CSR(jCounter);    %time vector
    i_el = ones(size(tp)) * 1; %which electrode -> does not really matter here, because only one nerve cell
    numberCycle = size(tp,2);
    UN = membraneNoise(ANParams.N_nervecells,CSR(jCounter),numberCycle); %membrane noise
    UN = UN(1:ANParams.N_nervecells,:);
    Tph = ones(size(tp)) * 100e-6; %phase duration
    Iamp = ones(size(tp))*I(jCounter);
    indexKeepRefracValues = [];         % init value for keeping the refractory values
    refValues             = zeros(ANParams.N_nervecells,2); % refractory values for keeping or not...
    tLAP   = ones(ANParams.N_nervecells,1)*-99/1000;  % arbitrary init value for the last action potential
    tAPOut = ones(ANParams.N_nervecells,1)*-99/1000;
    tAPOut = [tAPOut zeros(ANParams.N_nervecells,numberCycle)];
    
    for iPulse = 1:numberCycle %for every time step
        Si = [tp(iPulse),Iamp(iPulse),Tph(iPulse),i_el(iPulse)]; %just a carrier, which electrode, phase etc.
        [tAP,indexKeepRefracValues,refValues] = auditoryNerve(Si,UN(:,iPulse),tLAP,indexKeepRefracValues,refValues,ANParams);
        tAPOut(:,iPulse+1) = tAP;
        tLAP   = tAP(:,end);  %save last AP in vector tLAP
    end
    %collect histogram
    histogram{jCounter} = nonzeros(diff(tAPOut));
    %plot histograms in a subfigure
    X = [0.001:5e-5:0.01];
    N = hist(histogram{jCounter},X);
    subplot(4,2,plotposition(jCounter)), bar(X.*1000,N);
    xlim([0 11]);
    ylabel('N_{AP}');
    xlabel('ISI (ms)')
    titlestring = [num2str(CSR(jCounter)) ' pps'];
    title(titlestring);
end




