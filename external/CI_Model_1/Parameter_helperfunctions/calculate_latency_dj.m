function [dj, md, sd] = calculate_latency_dj(mL50,sigmaL50,P_AP)

% usage:  [dj] = calculate_latency_dj(mL50,sigmaL50,P_AP)
% input:  mL50     = 
%         sigmaL50 = 
%         P_AP     = 
% 
% output: dj = latency
%         
% 
% calculates the latency
% 
% Reference: Hamacher 2003 Ph.D.-thesis
%
% Author: Stefan Fredelake
% Date: 17-09-2009
% 
%

N = size(mL50,1);

s1 = -248e-6;
s2 = - 79e-6;


md = mL50     + s1*(P_AP-0.5);
sd = sigmaL50 + s2*(P_AP-0.5); 

dj = randn(N,1) .* sd + md;

% Copyright (C) 2009 Stefan Fredelake, Oldenburg University
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