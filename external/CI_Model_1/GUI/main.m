%% run_CI_model_extract_spikes.m
% Standalone script - no GUI needed

clear all;

% --- add model to path ---
addpath(genpath('/Users/francescodesantis/Desktop/CI_Model_1'));
cd('/Users/francescodesantis/Documents/repos/sound_localization/external/CI_Model_1/GUI');

% --- set your paths ---
paramfilename  = '../parameterstore/my_params.m';
signalfilename = '../wavfilestore/tone.wav';

% --- run the model ---
[mINew, logicANpattern, IR] = main_SP_CI_AN(signalfilename, paramfilename);

% --- extract spike timestamps per fiber ---
fs_AN = 10000;  % Hz, hardcoded in main_SP_CI_AN
N_fibers = size(logicANpattern, 1);
spike_times = cell(N_fibers, 1);  % cell array, one entry per fiber

for i = 1:N_fibers
    spike_idx = find(logicANpattern(i, :));      % sample indices
    spike_times{i} = spike_idx / fs_AN;          % convert to seconds
end

% --- save ---
save('spike_times.mat', 'spike_times');
disp('Done. Spike times saved.')