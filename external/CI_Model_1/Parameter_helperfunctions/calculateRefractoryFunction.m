function r = calculateRefractoryFunction(t,T_ARP,tau_RRP)

% usage:  r = calculateRefractoryFunction(t)
% input:  t       = time after the last action potential
%         T_ARP   = absolute refractory phase
%         tau_RRP = relative refractory phase
%
% output: r = refractory factor, which will be multiplied with Uth
% 
% calculates the refractroy factor, which is needed for the raise of the
% threshold Uth. 
% 
% Reference: Hamacher 2003 Ph.D.-thesis
%
% Author: Stefan Fredelake
% Date: 02-12-2008
% 
%

% T_ARP   = 0.7e-3;  % absolute refractory phase from Hamacher p. 79
% tau_RRP = 1.6e-3;  % relative refractory phase from Hamacher p. 79

q = 0.1; % constant from Hamacher p. 79
p = 0.68;  % constant from Hamacher p. 79

t = ones(size(T_ARP)).*t;





r = (1-exp(-(t-T_ARP)./(q*tau_RRP))).*(1-p*exp(-(t-T_ARP)./(tau_RRP)));
r = 1./r;

% index = find(t>T_ARP);
r(t<=T_ARP) = Inf;


% if t > T_ARP
%     r = (1-  exp( -(t-T_ARP)/(q*tau_RRP) )) * ...
%         (1-p*exp( -(t-T_ARP)/(tau_RRP)   ));
%     r = 1/r;
% else
%     r = Inf;
% end


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