% function [tLAPNew,indexKeepRefracValues,refValues,P_AP,index_AP] = auditoryNerve_constAVF_fast(ti,Tph,Iamp,v,tLAP,UN,indexKeepRefracValues,refValues,RS0)
function [tAP,indexKeepRefracValues,refValues,P_AP,index_AP] = auditoryNerve(Si,UN,tLAP,indexKeepRefracValues,refValues,ANParams)% ;ti,Tph,Iamp,v,tLAP,UN,indexKeepRefracValues,refValues,RS0)
%% Function to simulate the auditory Nerve output. 
% used Variables from the ANParams struct:
% ANParams.V
% ANParams.R
% ANParams.TAUCHR
% ANParams.EFFIRHEO
% ANParams.Uth
% ANParams.RS0_ind
% ANParams.MT_ARP;
% ANParams.MTAU_RRP;
% ANParams.N_nervecells;
% ANParams.ML50;
% ANParams.SIGMAL50;

% Si = [tp,Iamp,Tph,i_el];
tp   = Si(:,1);
Iamp = Si(:,2);
Tph  = Si(:,3);
i_el = Si(:,4);
   

% get the spatial spread function for the active electrode
v = ANParams.V(:,i_el);
% calculate effective current amplitude (eq. 6.1).
if numel(Iamp) == 1;
    effIamp = Iamp*v;
else
    effIamp = repmat(Iamp',ANParams.N_nervecells,1).*v;
    effIamp = effIamp(:,1)+effIamp(:,2);
    Tph     = Tph(1,:);
    tp      = tp(1,:);
end

% calculate depolarisation potential of the cell membrane (eq. 6.4).
UD = effIamp .* ANParams.R .* (1-exp(-Tph./(ANParams.TAUCHR/log(2))));

% refractory function
[T_ARP,tau_RRP,refValues] = calculate_refracConstants(ANParams.MT_ARP,ANParams.MTAU_RRP,indexKeepRefracValues,refValues);
r = calculateRefractoryFunction(tp-tLAP,T_ARP,tau_RRP); 


% phase dependent relative spread
RS0 = ANParams.RS0_ind * (1 + 792*Tph - 65833*Tph^2);
% correct the relative spread with the refractory function. the higher r,
% the less is the relative spread
RS0     = RS0./r;      % (eq. 6.30)

% calculate the std of the noise
sigma_Tph = ANParams.Uth.*RS0;  
% adjust the noise samples with the standard deviation. 
UN        = UN .* sigma_Tph;

% calculate the probability of firing (eq. 6.28)
P_AP = 1/2 + 1/2 * erf(  (UD - r.*ANParams.Uth) ./ (sqrt(2)*sigma_Tph)  );

% find for all neurons the depolarization current greater than the 
% threshold current, multiplied with the refractory function and noised the 
% noise samples (eq. 6.27)
indexNeuron = 1:ANParams.N_nervecells;
bool_AP   = (UD + UN >= (r .* ANParams.Uth) );
bool_noAP = ~bool_AP;

index_AP   = indexNeuron(bool_AP);
index_noAP = indexNeuron(bool_noAP);


% the time of the new action potentials... 
tp = tp*ones(ANParams.N_nervecells,1);

tAP(index_AP,:) =  tp(index_AP) + ...
         ANParams.TAUCHR(index_AP) .* log2(effIamp(index_AP)./(effIamp(index_AP) ...
        - (1-UN(index_AP)./ANParams.Uth(index_AP)).*ANParams.EFFIRHEO(index_AP))); % eq. 6.17


index_complex = find(imag(tAP));
% if no AP is generated tLAPNew is set to -2. It will be corrected in the
% function above... (not good programmed, I know)
tAP([index_noAP(:);index_complex(:)],:) = tLAP([index_noAP(:);index_complex(:)],:);


ifLatency = 1;  % if latency and jitter is include, otherwise 0
if ifLatency == 1
    [dj] = calculate_latency_dj(ANParams.ML50,ANParams.SIGMAL50,P_AP);


    tAPTmpAP(index_AP,:)  = tAP(index_AP,end);
    tLAPTmp(index_noAP,:) = tAP(index_noAP,end);
    if ~isempty(tAPTmpAP)
        tLAPTmp(index_AP,:)   = tAPTmpAP(index_AP) + dj(index_AP);
    end
    tAP = tLAPTmp;

else 
    dj = zeros(ANParams.N_nervecells,1);
    tLAPTmp(index_AP,:)    = tAP(index_AP) + dj(index_AP);
    tAP = tLAPTmp;
end

% if no AP was generated the refractory values should be kept.
indexKeepRefracValues = index_noAP;