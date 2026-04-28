function F = process_ACE_filterbank_demo(S)

% usage:  F = process_ACE_filterbank(S)
% input:  S = complex spectrum
% output: F = fíltered output spectrum
% 
% This filterbank is only applicable for a complex spectrum S with a length
% of 128 bins, and a sampling frequency of 16 kHz. 
%
% Reference: Laneau, 2005 PhD-thesis
%
% Author: Stefan Fredelake
% Date: 22-10-2008
% 

global NUMOFCHANNELS
global Q_SUM
global G
global indexOrg;
global vActiveElectrodes
% information about the frequency bins to sum 
% Q_SUM = [  ones(9,1); ...   
%          2*ones(4,1); ...
%          3*ones(2,1); ...
%          4*ones(2,1); ...
%          5*ones(2,1); 
%          6;7;8];
     
% Q_SUM = [     3     2     2     2     1     2     3     4     4     6     5     4     5     5    11]'; % bu600127
% Q_SUM = [1     1     1     1     1     1     1     2     2     2     2     3     4     5     4     7     7     8     9]'; % fj821217


S2 = abs(S).^2; % square the amplitude spectrum


F  = zeros(NUMOFCHANNELS,1);
for n = 1:NUMOFCHANNELS
    if n == 1
        index2 = indexOrg;  % only for the first loop
    end
    index1 = index2 + 1;
    index2 = index1 + Q_SUM(n)-1;
    index  = index1:index2;
    if Q_SUM(n) == 1
        weight = G(1);
    elseif Q_SUM(n) == 2
        weight = G(2);
    else
        weight = G(3);
    end
    F(n,:) = weight*sqrt(sum(S2(index)));   % eq. 5.4
end

% Information about the filterbank. 
% Chn.No. | num. f-Bins | fc, low bin | fc, upp bin  
%      1  |           1 |         250 |         250
%      2  |           1 |         375 |         375
%      3  |           1 |         500 |         500
%      4  |           1 |         625 |         625
%      5  |           1 |         750 |         750
%      6  |           1 |         875 |         875
%      7  |           1 |        1000 |        1000
%      8  |           1 |        1125 |        1125
%      9  |           1 |        1250 |        1250
%     10  |           2 |        1375 |        1500
%     11  |           2 |        1625 |        1750
%     12  |           2 |        1875 |        2000
%     13  |           2 |        2125 |        2250
%     14  |           3 |        2375 |        2625
%     15  |           3 |        2750 |        3000
%     16  |           4 |        3125 |        3500
%     17  |           4 |        3625 |        4000
%     18  |           5 |        4125 |        4625
%     19  |           5 |        4750 |        5250
%     20  |           6 |        5375 |        6000
%     21  |           7 |        6125 |        6875
%     22  |           8 |        7000 |        8000

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