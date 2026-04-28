function [Yn,ZnPlot] = processIntegrate(SR,fs)


Ta  = 1/fs;

% Zn ist ein internes Zustandssignal, welches durch Integration über das
% Eingangssignal gewonnen wird und somit Informationen über die 
% vorangegangene Stimulation verfügt, d.h. Zn ist eine Art Gedächtnis. 
Zn  = zeros(size(SR,1),1);

% Zeitkonstanten für die Integration
tauEin = 70e-3; % page 147
tauAus = 70e-3; % page 147

% c1 und c2 sind Konstanten
c1 = zeros(size(Zn));
c2 = zeros(size(Zn));

% Ist das Ausgangssignal ais der Integration und ist durch die 
% Maximalbildung entweder die kurzzeit-gemittelte Gruppenentladungsrate SR,
% wenn SR >= Zn oder Zn, wenn SR kleiner ist als Zn
Yn = zeros(size(SR));

ZnPlot = [];
for iSample = 1:size(SR,2)
    SRn = SR(:,iSample);
    
    boolSRGreaterThanZnOld = (SRn >= Zn);
    boolSRLessThanZnOld    = (SRn <  Zn);
    
    c1(boolSRGreaterThanZnOld) = exp(-Ta/tauEin);
    c2(boolSRGreaterThanZnOld) = 1-c1(boolSRGreaterThanZnOld);

    c1(boolSRLessThanZnOld) = exp(-Ta/tauAus);
    c2(boolSRLessThanZnOld) = 0;
    
    Zn = c1.*Zn + c2.*SRn;
    ZnPlot = [ZnPlot Zn];
    
    Yn(:,iSample) = max([SRn Zn],[],2);

end




