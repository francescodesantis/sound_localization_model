%% run_CI_model_extract_spikes.m
% Standalone script - no GUI needed
clear all;

% --- add model to path ---
addpath(genpath('/Users/francescodesantis/Desktop/CI_Model_1'));
cd('/Users/francescodesantis/Desktop/CI_Model_1/GUI');

% --- set your paths ---
paramfilename       = '../parameterstore/my_params.m';
%paramfilename       = '../parameterstore/example_nervecells_35000.m';
signalfilename_left = '../wavfilestore/binaural_left_wn.wav';
signalfilename_right = '../wavfilestore/binaural_right_wn.wav';
savepath = '/Users/francescodesantis/Documents/repos/sound_localization_model/data/ANF_SPIKETRAINS/CImodel/wn/';

% create output folder if it doesn't exist
if ~exist(savepath, 'dir')
    mkdir(savepath);
end

% --- helper: extract spike times from logicANpattern ---
fs_AN = 10000;

% --- LEFT ---
[mINew_left, logicANpattern_left, IR_left] = main_SP_CI_AN(signalfilename_left, paramfilename);


N_fibers = size(logicANpattern_left, 1);
spike_times_left = cell(N_fibers, 1);
for i = 1:N_fibers
    spike_idx = find(logicANpattern_left(i, :));
    spike_times_left{i} = spike_idx / fs_AN;
end
save(fullfile(savepath, 'spike_times_left.mat'), 'spike_times_left');
disp('Left ear done.')

% --- RIGHT ---
[mINew_right, logicANpattern_right, IR_right] = main_SP_CI_AN(signalfilename_right, paramfilename);
spike_times_right = cell(N_fibers, 1);
for i = 1:N_fibers
    spike_idx = find(logicANpattern_right(i, :));
    spike_times_right{i} = spike_idx / fs_AN;
end
save(fullfile(savepath, 'spike_times_right.mat'), 'spike_times_right');
disp('Right ear done.')

disp('All done.')