% neuronal forward masking
% clear
% close all;
% addpath(['.' filesep 'parameterstore']);


devicename = 'freedom';
%set_global_constants;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ANParams.N_nervecells = 5000;
ANParams.V = ones(ANParams.N_nervecells,22);
C     = 1;
 Tph = 100e-6; %phase duration
[ANParams.TAUCHR,ANParams.EFFIRHEO] = generateChronaxieRheobase(ANParams.N_nervecells); 
% relative spread of the membrane noise
ANParams.RS0_ind = calculateRelativeSpread(ANParams.N_nervecells);
% refractory constants T_ARP, TAU_RRP
[ANParams.MT_ARP,ANParams.MTAU_RRP] = generateRefractoryConstants(ANParams.N_nervecells);
% Neural latency
[ANParams.ML50,ANParams.SIGMAL50] = generateNeuralLatencyConstants(ANParams.N_nervecells);


%%% central auditory processing
fsZAS = 5000;
% 1. Lowpass ZAS
hTP1 = generateImpRespPostMasking(fsZAS);
% 2. Lowpass ZAS
Hd = generateFilterObjLPMasking_2;

tauM = ANParams.TAUCHR/log(2);  % the membrane time constant     (eq. 6.10)
ANParams.R    = tauM/C;         % the resistance of the membrane (eq. 6.10)

ANParams.Uth  = ANParams.EFFIRHEO .* ANParams.R;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 CSR = 1000; % stimulation rate

 xGroup = [0.25 34.75];   %only one group of nerve fibers
 indexGroup = [1 ANParams.N_nervecells];%  all nerve cells in one group
 
%masker
 tmasker = 100e-3; %s
 Imasker = 200e-6; %A
 tgap = 50e-3; %s
 tprobe = 20e-3; %s
 tafterprobe = 10e-3; %s
 tp   = 0:1/CSR:(tmasker+tgap+tprobe+tafterprobe)-1/CSR;
 i_el = ones(size(tp))*11; % which electrode
 
%left figure
 Iprobe = 180e-6; %A
 Iamp = [repmat(Imasker,1,tmasker*CSR) zeros(1,tgap*CSR) ...
     repmat(Iprobe,1,tprobe*CSR) zeros(1,tafterprobe*CSR)];
[IR,~,~,SR, Zn] = main_CI_AN_model(Iamp,i_el,tp,Tph,ANParams.N_nervecells,xGroup,fsZAS,indexGroup,hTP1,Hd,ANParams);
t_IR = [0:1/fsZAS:(size(IR,2)-1)/fsZAS];
%plot figure
figure;
plot(t_IR,sum(IR,1)), hold on;
plot(t_IR,sum(SR,1),'--');
plot(t_IR,sum(Zn,1),'-.');
xlabel('Time (s)');
xlim([0 0.25]);
ylabel('IR(k), SR(k), Z(k)');
legend('IR(k)','SR(k)','Z(k)','Location','NorthEast');
clear IR SR Zn Iamp

%right figure
Iprobe = 125e-6; %A
Iamp = [repmat(Imasker,1,tmasker*CSR) zeros(1,tgap*CSR) ...
    repmat(Iprobe,1,tprobe*CSR) zeros(1,tafterprobe*CSR)];
[IR,~,~,SR, Zn] = main_CI_AN_model(Iamp,i_el,tp,Tph,ANParams.N_nervecells,xGroup,fsZAS,indexGroup,hTP1,Hd,ANParams);
t_IR = [0:1/fsZAS:(size(IR,2)-1)/fsZAS];
%plot figure
figure;
plot(t_IR,sum(IR,1)), hold on;
plot(t_IR,sum(SR,1),'--');
plot(t_IR,sum(Zn,1),'-.');
xlabel('Time (s)');
xlim([0 0.25]);
ylabel('IR(k), SR(k), Z(k)');
legend('IR(k)','SR(k)','Z(k)','Location','NorthEast');
