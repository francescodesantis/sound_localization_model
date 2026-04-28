function w = generate_hanning_window(L)

% usage:  w = generate_hanning_window(L)
% input:  L = length of the window
% output: w = hanning window
% 
% This function generates a hanning window of the length L.
%
% Reference: Laneau, 2005 PhD-thesis
%
% Author: Stefan Fredelake
% Date: 22-10-2008
% 

k = 0:L-1;
k = k(:);
w = 0.5 * (  1-cos(2*pi*k/L)  );    % eq. 5.3


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