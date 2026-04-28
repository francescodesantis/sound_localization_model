function [mL50,sigmaL50] = generateNeuralLatencyConstants(N)

% usage:  [mL50,sigmaL50] = generateNeuralLatency(N)
% input:  N   = number of neurons
% 
% output: mL50 =     random number for the calculation of the latency, each
%                    element stands for a neuron
%         sigmaL50 = random number for the calculation of the latency, each
%                    element stands for a neuron
%         
% 
% generates a random vector with numbers for the calculation of the
% propagation latency
% 
% Reference: Hamacher 2003 Ph.D.-thesis
%
% Author: Stefan Fredelake
% Date: 17-09-2009
% 
%

mean_mL50     = 607e-6;  % p.102
mean_sigmaL50 = 106e-6;  % p.102

std_mL50     = 142e-6;   % p.102
std_sigmaL50 =  87e-6;   % p.102


x0 = randn(N,1);
x1 = (x0*std_mL50)     + mean_mL50;
x2 = (x0*std_sigmaL50) + mean_sigmaL50;

xnew = (randn(N,1)*std_sigmaL50) + mean_sigmaL50;

rho  = my_corr(x1,x2);
while rho > 0.5
    index = 0;
    while index == 0
        index = round(N*rand(1));
    end  
    x2(index) = xnew(index);
    rho = my_corr(x1,x2);
end

mL50     = x1;
sigmaL50 = xnew;

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