function [mTLap,mtLAPGrouped] = process_neural_model(mTi,mIdB,mTph)

% inputs: mTi  = a matrix, with information on the time and electrode, when 
%                and where a stimulus is present
%         mIdB = a matrix, with information on current amplitude in dB and 
%                electrode, the corresponding time is stored in mTi
%         mTph = a matrix, with information on phase length in seconds and 
%                electrode, the corresponding time is stored in mTi
%       
%         Note: 
%                     ---Time------>
%             i 1   | t11 t12 t13...t1T |
%             E 2   | t21 t22 t23...t2T |
%             l 3   | t31 t32 t33...t3T |
%             e 4   | t41 t42 t43...t4T |
%             c .   | t.1 t.2 t.3...t.T |
%             t .   | t.1 t.2 t.3...t.T |
%             r .   | t.1 t.2 t.3...t.T |
%             o E-2 | t.1 t.2 t.3...t.T |
%             d E-1 | t.1 t.2 t.3...t.T |
%             e E   | tE1 tE2 tE3...tET |
%
%
% outputs: mTLap = a matrix, containing the time point of the last action
%                  potential of each neuron
%          mtLAPGrouped = a matrix,containing the time point of the last action
%                         potential of each neuron
%
%
%

% global V;








global IREF;
global N;
global ML50
global SIGMAL50
global indexGroup
global % CIParams.vActiveElectrodes
global mUN
global V
global RS0_ind
global Tph
% iElectrode = find(~isnan(% CIParams.vActiveElectrodes));

% mTi  = mTi(iElectrode,:);
% mIdB = mIdB(iElectrode,:);
% mTph = mTph(iElectrode,:);



vTi  = mTi(:); % all columns are arranged into one row
vIdB = mIdB(:);
vTph = mTph(:);

% it must be possible to finde the electrodes to the time vector
numberElectrode = length(% CIParams.vActiveElectrodes);
iElectrode      = 1:numberElectrode;
iElectrode      = iElectrode(:);
iElectrodeTime  = repmat(iElectrode,size(mTi,2),1);


[vTiSorted,vIndexTiSorted] = sort(vTi,'ascend');


% hier muss man nun die Vektoren zu Reihenvektoren machen und dabei
% achten, dass die Zuordnungen von Strom, und Phasendauer zum Zeitpunkt
% stimmen, ACHTUNG Hier die Option offen lassen für SPEAK, dass man da
% irgendwie die effektiven Amplituden addiert... vielleicht kann man dies
% mit einer Matrix machen, die summiert wird... 
vTi  = vTiSorted(:).';
vIdB = vIdB(vIndexTiSorted).';
vTph = vTph(vIndexTiSorted).';
iElectrodeTime = iElectrodeTime(vIndexTiSorted).';
% später für SPEAK erweitern... 

index = find(vTi~=Inf);
vTi  = vTi(index);
vIdB = vIdB(index);
vTph = vTph(index);
iElectrodeTime = iElectrodeTime(index);




numberStimuli = length(vTi);

mUN = mUN(:,1:numberStimuli);
mUN = mUN(randperm(N),:);

% calculate the current levels in dB into linear amplitude values in A
vIamp = IREF * 10.^(vIdB/20);

% Init values
tLAPOld    = ones(N,1)*-99/1000;  % init value for the last action potential
tLAPOut    = ones(N,1)*-99/1000;

indexKeepRefracValues = []; % init value for keeping the refractory values
refValues             = zeros(N,2); % refractory values for keeping or not...


mtLAPGrouped = zeros(size(indexGroup,1),1);

% calculate the relative spread
RS0 = RS0_ind * (1 + 792*Tph - 65833*Tph^2);

for iStimulus = 1:numberStimuli
%     if mod (iStimulus,100) == 0
%         disp(['Stimulus ' int2str(iStimulus) '. of ' int2str(numberStimuli) ' in process.'])
%     end
    v = V(:,iElectrodeTime(iStimulus));
    [tLAPNew,indexKeepRefracValues,refValues,P_AP,index_AP] = auditoryNerve_constAVF_fast(vTi(:,iStimulus),vTph(:,iStimulus),vIamp(:,iStimulus),v,tLAPOld,mUN(:,iStimulus),indexKeepRefracValues,refValues,RS0);
    % find the last action potential of each neuron for the keeping the
    % refractory values or not. 
    tLAPCalc = tLAPNew(:,end);
    indexInf = find(tLAPCalc==-Inf);
    indexNotInf = find(tLAPCalc~=-Inf);
    tLAPCalc(indexInf) = tLAPOld(indexInf,:);    
    tLAPOld = tLAPCalc;

    % calculate the propagation latency 
    [dj] = calculate_latency_dj(ML50,SIGMAL50,P_AP);
    
    tLAPTmp(indexNotInf,:) = tLAPCalc(indexNotInf);
    tLAPTmp(indexInf,:)    = tLAPOut(indexInf,end);
    tLAPTmp(index_AP,:)    = tLAPCalc(index_AP) + dj(index_AP);
    tLAPOut = [tLAPOut tLAPTmp];

    deltaAP = tLAPOut(:,end) - tLAPOut(:,end-1);
    isAP = (deltaAP>0);
    for iGroup = 1:size(indexGroup,1)
        mtLAPGrouped(iGroup,iStimulus) = sum(isAP(indexGroup(iGroup,1):indexGroup(iGroup,2)));
    end
    
end
mTLap = tLAPOut;
clear tLAP tLAPOut;
