function [RS0_ind] = calculateRelativeSpread(N)

% usage:  RS0 = calculateRelativeSpread(Tph,N)
% input:  Tph = phase duration in seconds
%         N   = number of neurons
% 
% output: RS0 = relative spread accoring to Bruce et al
% 
% calculates the relative spread accorind to Bruce et al. Note Bruce et al
% assumes at this function is only applicable to 100us<=Tph<=5000us and 
% bipolar stimulation. (p. 38) But Hamacher used it for lower Tph values 
% and also for monopolar stimulation. (p. 79)
% 
% Reference: Hamacher 2003 Ph.D.-thesis
%
% Author: Stefan Fredelake
% Date: 02-12-2008
% 
%


% RS0_ind = 0.12;   % phase duration independent part of the relative spread

meanRS0_ind = 0.095; % from Hamacher p. 100
stdRS0_ind  = 0.04;  % from Hamacher p. 100

RS0_ind     = (randn(N,1)*stdRS0_ind)+meanRS0_ind;




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