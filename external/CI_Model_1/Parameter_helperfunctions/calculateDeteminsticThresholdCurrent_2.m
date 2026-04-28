function Ith0 = calculateDeteminsticThresholdCurrent_2(IRheo,tauM,Tph)

% usage:  Ith0 = calculateDeteminsticThresholdCurrent(IRheo,C,Tph)
% input:  IRheo = the rheobase 
%         tauM  = time constant of the membrane given with RC
%         Tph   = phase of the stimulus
%
% output: Ith0 = the deterministic current for threshold
% 
% calculates the deteministic current for threshold for a given pulse
% with a defined phase Tph
%
% 
% Reference: Hamacher 2003 Ph.D.-thesis
%
% Author: Stefan Fredelake
% Date: 01-12-2008
% 
%

% global tauM

Ith0 = IRheo./(1-exp(-Tph./tauM));


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