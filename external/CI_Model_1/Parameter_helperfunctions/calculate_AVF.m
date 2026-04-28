function v = calculate_AVF(x_el,x_nz,lambda,v0)

% usage:  v = calculate_AVF(x_el,x_nz,szWhichConfig)
% input:  x_el = position of first electrode [mm]
%         x_nz = position of neuron cell [mm]
%         lambda = constant of the spatial spread function
%           v0 = constant factor of the spatial spread
%
% output: v = spreading function
% 
% calculates the spreading function of an electrical field according for
% monopolar stimulation
%
% 
% 
% monopola 
%
%
%                  el1
%
%
% -o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o----->
%        x_nz                                          x
%
%
% Reference: Hamacher 2003 Ph.D.-thesis
%
% Author: Stefan Fredelake
% Date: 17-10-2008
% 
%

%v0 = 1; % from Hamacher: default value p. 95
v  = v0 * sf_mono(x_el,x_nz,lambda);
       

function v = sf_mono(x_el,x_nz,lambda)
for iElectrode = 1:length(x_el)
    v(:,iElectrode) = abs(exp(-abs(x_nz-x_el(iElectrode))/lambda));
end



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