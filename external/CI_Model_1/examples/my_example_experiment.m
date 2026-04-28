%% Example experiment file for the CINM-Model
% This example experiments runs different parameterfiles for one .wav-file
% The only changed parameter in the model is the number of nervecells. 

% clean up workspace
clear; close all; clc;

% add all needed paths (Note: Upper Directory mus be in Matlabs-Path):
CImodel_start; 

% define a custom parameterfile and wavfile

% parameterfile, which does not set the number of nervecells

wavfile = 'mixed_44kHz.wav';

N_nervecells = [500; 250; 50]; %Calculate for three different Nerve cell numbers

for myloop = 1:length(N_nervecells)
    parameterfile = strcat('example_nervecells_', num2str(N_nervecells(myloop))); %
    %run the model
    main_SP_CI_AN(wavfile, parameterfile); 
end