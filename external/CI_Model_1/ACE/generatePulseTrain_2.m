function vT = generatePulseTrain_2(iElectrode,vTBegin,pps, CIParams)


% generatePulseTrain
% global CIParams.vActiveElectrodes
% global Tph
% global ipg


iElectrode = sort(iElectrode,1,'descend');

nChns = length(iElectrode);

stimulationsequence = 1:length(CIParams.vActiveElectrodes);

period           = 1/pps;
lenBiphasicPulse = 2*CIParams.Tph + CIParams.ipg;


% auch diese funktion ist nach meinen eigenen überlegungen entstanden. mir
% fehlen infos darüber, wann exakt die elektroden stimulieren. 
% dazu war meine überlegung, dass eine periode eine länge von 1/pps hat.
% wir haben z.B. 8 Pulse mit einer pulsdauer von 2*phasendauer+interphasegap 
% an verschiedenen elektroden. (1/pps-8*pulsdauer)/8 ergibt einen rest, den ich
% als deltaT bezeichne und dieser sei der zeitliche abstand zwischen zwei
% aufeinanderfolgende pulse
%
% Nachtrag: In einem Paper von Laneau und Wouters habe ich nun gelesen,
% dass diese Idee richtig ist. Laneau und Wouters 2004 JARO
%     _
% ___| |  _______
%      |_|
%
%         <-- deltaT -->
%                        _
% ______________________| |  _______
%                         |_|



deltaT = (period-lenBiphasicPulse*nChns)/nChns; 


vTtmp = 0:lenBiphasicPulse+deltaT:(nChns-1)*(lenBiphasicPulse+deltaT);

vTtmp = vTtmp + vTBegin;

vT = Inf*ones(length(CIParams.vActiveElectrodes),1);
vT(stimulationsequence(iElectrode),:) = vTtmp;
