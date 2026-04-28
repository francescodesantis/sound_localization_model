% figureS2reproduce
MT_ARP = 0.7e-3; %s                     % constant for the absolute refractory period
MTAU_RRP = 1.6e-3; %

N_nervecells = 1;
deltaTLAP = [0:0.1:13].*1e-3; %time difference to last AP
indexKeepRefracValues = [];         % init value for keeping the refractory values
refValues             = zeros(N_nervecells,2); % refractory values for keeping or not...
number_of_refractoryfunctions = 20; %how many (stochastic) refractory functions?
figure;

for iCounter = 1:number_of_refractoryfunctions
    % refractory function
    [T_ARP,tau_RRP,refValues] = calculate_refracConstants(MT_ARP,MTAU_RRP,indexKeepRefracValues,refValues);
    r = calculateRefractoryFunction(deltaTLAP,T_ARP,tau_RRP);
    
    %plot
    plothandle = plot(deltaTLAP.*1000,20.*log10(r),'k-'); %ms vs dB
    set(plothandle,'Color',[.8 .8 .8])
    
    hold on;
end
%one non-stochastically distributed refractory function
r = calculateRefractoryFunction(deltaTLAP,MT_ARP,MTAU_RRP); 
meanplothandle = plot(deltaTLAP.*1000,20.*log10(r),'k-'); %ms vs dB
ylim([-2 15]);
xlim([0 13]);
xlabel('\Delta t_{LAP} (ms)');
ylabel('20 log_{10}(r) (dB)');

%plot data of Dynes (1996)
Dynes_x = [1.89 2.13 2.39 2.71 3.05 3.43 3.86 4.38  4.91 5.52  6.19  6.98  7.84  8.81 9.91  11.1 12.5];
Dynes_y = [2.72 2.32 2.00 2.34 1.12 1.01 0.217 .535 .517 -.083 .0229 -.0653 .341 .235 .323 .0229 .0229];
Dynesplothandle = plot(Dynes_x,Dynes_y,'ko');

legend([Dynesplothandle, meanplothandle, plothandle],'probe threshold (Dynes, 1996)','mean refractory function','example stochastic refr. func.','Location','NorthEast');