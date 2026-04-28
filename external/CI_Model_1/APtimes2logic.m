function logicANpattern = APtimes2logic(APvec,number_nerve_cells,FS)
if nargin < 3
    FS = 10000; %sampling frequency of the AN firing pattern
end

if isempty(APvec)
    logicANpattern = zeros(number_nerve_cells,1);
else
    %% generates a logical AN spiking pattern from the variable APvec (Boston notation)
   
    %initialize the AN pattern with zeros (if there are no INFs involved)
    if isfinite(round(max(APvec(:,2))*FS))
        logicANpattern = zeros(number_nerve_cells,round(max(APvec(:,2))*FS));
    end
    
    %fill the AN pattern with ones at those positions where APs are
    for iCounter = 1:size(APvec,1)
        timeindex = max([round(APvec(iCounter,2)*FS) 1]);
        logicANpattern(APvec(iCounter,1),timeindex) = 1;
    end
    
    logicANpattern = logical(logicANpattern); %make it really logical
end