function [mT_ARP,mTau_RRP] = generateRefractoryConstants(N)

% usage:  [mT_ARP,mTau_RRP] = generateRefractoryConstants(N)
% input:  N   = number of neurons
% 
% output: T_ARP   = absolute refractory phase
%         tau_RRP = relative refractory phase time constant
% 
% generates a random vector with absoulte and relative refractory phases.
% both vectors are correlation with rho = 0.75
% 
% Reference: Hamacher 2003 Ph.D.-thesis
%
% Author: Stefan Fredelake
% Date: 02-12-2008
% 
%

mean_mT_ARP   = 0.7e-3; % mittlere absolute Refraktärphase
mean_mtau_RRP = 1.6e-3; % mittlere relative Refraktätphase

std_mT_ARP   = 0.15 * mean_mT_ARP;      
std_mtau_RRP = 0.15 * mean_mtau_RRP;

x0   = randn(N,1);
xnew = (randn(N,1)*std_mtau_RRP) + mean_mtau_RRP;

x1 = (x0*std_mT_ARP)   + mean_mT_ARP;
x2 = (x0*std_mtau_RRP) + mean_mtau_RRP;


rho  = my_corr(x1,x2);
while rho > 0.75
    index = 0;
    while index == 0
        index = round(N*rand(1));
    end  
    x2(index) = xnew(index);
    rho = my_corr(x1,x2);
end

mT_ARP   = x1;
mTau_RRP = x2;

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