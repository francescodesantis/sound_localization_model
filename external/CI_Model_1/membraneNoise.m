function n=membraneNoise(N_nervecells,fs,N_lenNoise)
% function n=membraneNoise(N,fs)
%
% Generate membrane noise of N elements at a rate of fs [1/s].
%
% The output n will be a row-vector.
% (Calculation is speeded up for fs=7200 and fs=9000)

% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
% fc: Butterworth corner frequency in Hz
fc= 200;

% fo: Butterworth filter order
fo= 1;
% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
% generate Gaussian noise with given noise power and length
% n= wgn(1,N,10.5);
n = randn(N_nervecells,N_lenNoise);


% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
if fs > 400 % only filter the Gaussian noise if its sampling frequency is more than twice the desired corner frequency
    % - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    % design the filter coefficients + speed up with default parameters
    if (fc == 200)
        if (fs == 7200)
            b=[0.0805,0.0805];
            a=[1.0000,-0.8391];
        elseif (fs == 9000)
            b=[0.0654,0.0654];
            a=[1.0000,-0.8693];
        else
            [b,a]= butter(1,fc/(fs/2));
        end;
    end;
    
    
    if (size(n,1)*size(n,2)) > 5e7 %this would be a too large matrix for the filter command
        %split it up into 10 packs of cells and filter them separately
        number_of_packs = 10;
        newvec = zeros(N_nervecells,N_lenNoise);
        for iCounter = 1:number_of_packs
            newvec(1+(iCounter-1)*size(n,1)/number_of_packs:iCounter*size(n,1)/number_of_packs,:) ...
                = filter(b,a,n(1+(iCounter-1)*size(n,1)/number_of_packs:iCounter*size(n,1)/number_of_packs,:),[],2);
        end
    else
        % filter noise
        n= filter(b,a,n,[],2);
    end
end

