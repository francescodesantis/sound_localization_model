function [xGroup, indexGroup] = generateNeuralGroups(X_NZ, X_EL)


%global X_NZ
%global X_EL

ERB = 0.9; % 1 ERB is 0.9 mm on the basilarmembrane

nElectrodes = length(X_EL);
if nElectrodes == 1
    xgBorders   = (X_EL-ERB/2):ERB:X_NZ(end);
    xgBorders   = [fliplr(xgBorders(1)-ERB:-ERB:0) xgBorders];
else
    d_el  = (diff(X_EL(1:2))); % ich nehme an, dass der Abstand zwischen den 
                             % Elektroden bei 
                             % einem CI konstant ist. 
    % Gruppengrenzen zwischen den Elektroden...
    xgBorders = (X_EL(1)-d_el/2):d_el:X_NZ(end);
    xgBorders = [fliplr(xgBorders(1)-d_el:-d_el:0) xgBorders];
end


xGroup = [xgBorders(1:end-1)' xgBorders(2:end)'];


indexGroup = zeros(size(xGroup,1),1);
for ii = 1:size(xGroup,1)
    
    [mini,index] = min(abs(X_NZ - xGroup(ii,1)));
    indexGroup(ii,:) = index;
end

[mini,index] = min(abs(X_NZ - xGroup(end,2)));
clear mini;
indexGroup = [indexGroup+1 [indexGroup(2:end);  index]];

