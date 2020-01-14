%% script to simulate Poisson Process data and run sv-ppGPFA 
clear all; close all;
rng(2)
addpath(genpath('~/dev/research/programs/src/matlab/svGPFA_savingTestData'))
%% make simulated data
dy = 50; % number of neurons
ntr = 5; % number of trials
% generate dataset with three latents
[Y,prs,rates,fs,dx,dy,ntr,trLen,tt] = generate_toy_data(dy,ntr);

ngtest = 2000;
testTimes = linspace(0,max(trLen),ngtest)';

trueLatents = {};
for nn = 1:ntr
    for ii = 1:dx
        trueLatents{nn, ii} = fs{ii,nn}(testTimes);
    end
end

simFilename = 'results/pointProcessSimulation.mat';
save(simFilename, 'Y', 'prs', 'rates', 'fs', 'dx', 'dy', 'ntr', 'trLen', 'tt', 'testTimes', 'trueLatents')
