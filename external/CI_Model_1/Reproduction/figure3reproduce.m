%spiking_probability as a function of current level
% clear
% close
% addpath(['.' filesep 'parameterstore']);

%% set model parameters
ANParams.N_nervecells = 1;
ANParams.TAUCHR = 125e-6; %s
ANParams.EFFIRHEO = 87.5e-6; %A
ANParams.RS0_ind = 0.0774;
ANParams.MT_ARP = 0.6e-3; %s                     % constant for the absolute refractory period
ANParams.MTAU_RRP = 1.2e-3; %

ANParams.V = ones(1,22);

C =1;
%[ANParams.ML50,ANParams.SIGMAL50] = generateNeuralLatencyConstants(ANParams.N_nervecells);
ANParams.ML50 = 0;
ANParams.SIGMAL50 = 0;

tauM = ANParams.TAUCHR/log(2);  % the membrane time constant     (eq. 6.10)
ANParams.R    = tauM/C;         % the resistance of the membrane (eq. 6.10)

ANParams.Uth  = ANParams.EFFIRHEO .* ANParams.R;
CSR = [100 200 300 400 600 800]; % stimulation rates

IdB = [48:0.5:60]'; %dB values ref. 1µA
I = 10.^(IdB./20).*1e-6; %currents in A 

tLAP   = ones(ANParams.N_nervecells,1)*-99/1000;  % arbitrary init value for the last action potential
P_AP = [];
meanP_AP = [];

%% display progress bar
progressbar = waitbar(0,'Starting processing')
for jCounter = 1:length(CSR) %different stimulation rates
    tp = 0:1/CSR(jCounter):499/CSR(jCounter);    %time vector
    i_el = ones(size(tp)) * 1; %which electrode -> does not really matter here, because only one nerve cell
    numberCycle = size(tp,2);
    UN = membraneNoise(ANParams.N_nervecells,CSR(jCounter),numberCycle); %membrane noise
    UN = UN(1:ANParams.N_nervecells,:);
    Tph = ones(size(tp)) * 50e-6; %phase duration
    for iCounter = 1:size(I) %for every current to be tested
        Iamp = ones(size(tp))*I(iCounter);
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
        meanspikerate(jCounter,iCounter) = sum(sign(diff(tAPOut,[],2)))/numberCycle; %cumbersome, but correct!
    end
    %meanP_AP(jCounter,:) = mean(P_AP,2); %average probability over many
    %stimulation samples THE ALTERNATIVE!
    waitbar(jCounter/length(CSR),progressbar, 'Processing...');
end
% delete progressbar
delete(progressbar);
%plot the results
figure;
plot(IdB,meanspikerate);
ylabel('spike probability / spikes per pulse');
xlabel('I (dB ref. 1 \muA)');
title('Comparison of observed and modeled spike probabilities as a function of the stimulus intensity with different and pulse rates')
legend([num2str(CSR(1)) ' pps'],[num2str(CSR(2)) ' pps'],[num2str(CSR(3)) ' pps'],[num2str(CSR(4)) ' pps'],[num2str(CSR(5)) ' pps'],[num2str(CSR(6)) ' pps'],'Location','SouthEast')
hold on;

%plot literature data of Javel(1990)
%get the colors from preceding plot
modelhandles = get(gca,'Children');
Ix = [50.8 51.2 51.8 52.3 52.7 53.2]; Spy = [0.0166 0.487 0.808 0.947 0.97 .976];
plot(Ix,Spy,'o','Color',get(modelhandles(end),'Color'));
Ix = [51.3 51.8 52.3 52.7 53.2 53.8]; Spy = [0.158 0.417 0.663 0.852 0.958 .974];
plot(Ix,Spy,'o','Color',get(modelhandles(end-1),'Color'));
Ix = [51 51.8 52.3 52.8 53.2 53.8]; Spy = [.0741 .313 .493 .647 .828 .992];
plot(Ix,Spy,'o','Color',get(modelhandles(end-2),'Color'));
Ix = [51.1 51.9 52.6 53.2 53.9 54.5 55]; Spy = [.0364 .198 .381 .503 .645 .798 .942];
plot(Ix,Spy,'o','Color',get(modelhandles(end-3),'Color'));
Ix = [52.1 52.7 53.4 54.1 54.7 55.2 55.7 56.1 56.6]; Spy = [.133 .284 .442 .505 .558 .651 .793 .917 .983];
plot(Ix,Spy,'o','Color',get(modelhandles(end-4),'Color'));
Ix = [51.3 52.3 53.2 54 54.8 55.5 56.2 56.8 57.3 57.8]; Spy = [.0202 .105 .256 .342 0.448 .543 .631 .712 .786 .836];
plot(Ix,Spy,'o','Color',get(modelhandles(end-5),'Color'));
grid on;