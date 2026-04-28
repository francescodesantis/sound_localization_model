function [tauChr,effIRheo] = generateChronaxieRheobase(N)

% usage:  [tauChr,IRheo] = generateChronaxieRheobase(N)
% input:  N = number of neurons
%       
% output: tauChr = chronaxie values
%         IRheo  = effecitve rheobase values
%
% Generates random values for chronaxie and rheobase according to Hamacher 
% pp 97-100.
% 
% 
% Reference: Hamacher 2003 Ph.D.-thesis
%
% Author: Stefan Fredelake
% Date: 02-12-2008
% 
%


if nargin < 1
    N = 10000;
end

meanTauChr = 255e-6;    % seconds
stdTauChr  =  57e-6;    % seconds

meanIRheo  =   32e-6;    % ampere
stdIRheo   =    6.5e-6;  % ampere

tauChr     = (randn(N,1)*stdTauChr) + meanTauChr;
effIRheo   = (randn(N,1)*stdIRheo)  + meanIRheo;


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