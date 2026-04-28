function [T_ARP,tau_RRP,refValues] = calculate_refracConstants(MT_ARP,MTAU_RRP,indexKeepRefracValues,refValues)

% usage:  [T_ARP,tau_RRP] = calculateT_ARP_tauRRP(N)
% input:  N   = number of neurons
% 
% output: T_ARP   = absolute refractory phase
%         tau_RRP = relative refractory phase
% 
% generates a random vector with absoulte and relative refractory phases.
% both vectors are correlated with rho = 0.75
% 
% Reference: Hamacher 2003 Ph.D.-thesis
%
% Author: Stefan Fredelake
% Date: 02-12-2008
% 
%

N = size(MT_ARP,1);

std_mT_ARP   = 0.15 * MT_ARP;
std_mtau_RRP = 0.15 * MTAU_RRP;

x0   = randn(N,1);
xnew = randn(N,1) .* std_mtau_RRP;

x1 = x0 .* std_mT_ARP;
x2 = x0 .* std_mtau_RRP;
xOrg = x2;  % falls wieder Werte benötigt werden!

index     = ceil(N*rand(round(3/10*N),1));

x2(index) = xnew(index);
rho       = my_corr(x1,x2);
if round(rho*100) ~= 0.75
    if rho > 0.75
        while rho > 0.75
            index = ceil(N*rand(1));
            x2(index) = xnew(index);
            rho = my_corr(x1,x2);
        end
    elseif rho < 0.75
        runs = 0;
        while rho < 0.75
            runs = runs + 1;
            x2(index(runs)) = xOrg(index(runs));
            rho = my_corr(x1,x2);
        end
    end
end

N_ARP   = x1;
N_RRP   = x2; 


N_ARP(indexKeepRefracValues) = refValues(indexKeepRefracValues,1);
N_RRP(indexKeepRefracValues) = refValues(indexKeepRefracValues,2);

T_ARP   = MT_ARP   + N_ARP;
tau_RRP = MTAU_RRP + N_RRP;


refValues = [N_ARP N_RRP];

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