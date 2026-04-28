function [A,iElectrode] = process_nOfm(F,nChns)

% usage:  A     = process_nOfm(F,nChns)
% input:  F          = filterbank output (22 filters)
% output: nChns      = number of channels to simulate
%         iElectrode = index of the stimulating electrodes
% 
% This function chooses the channels to stimulate. The channels with the
% nChns maximal values are stimulated
%
% Reference: Laneau, 2005 PhD-thesis
%
% Author: Stefan Fredelake
% Date: 23-10-2008
% 

if nargin < 2
    nChns             = 8; % number of channels to stimulate
end

A               = zeros(size(F));
[Fs,iElectrode] = sort(F,1,'descend');
iElectrode      = iElectrode(1:nChns);
A(iElectrode)   = F(iElectrode);      % eq 5.16


% Copyright (C) 2008 Stefan Fredelake, Oldenburg University
% This program is free software; you can redistribute it and/or modify it 
% under the terms of the GNU General Public License as published by the 
% Free Software Foundation; either version 3 of the License, or (at your 
% option) any later version.
% This program is distributed in the hope that it will be useful, but 
% WITHOUT ANY WARRANTY; without even the implied warranty of 
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General 
% Public License for more details.
% You should have received a copy of the GNU General Public License 
% along with this program; if not, see http://www.gnu.org/licenses/.
% 