%% script to simulate Poisson Process data and run sv-ppGPFA 
clear all; close all;
rng(2)
% addpath(genpath('~/dev/research/programs/src/matlab/svGPFA_savingTestData'))
addpath(genpath('../src'))
%% make simulated data
dy = 10; % number of neurons
ntr = 5; % number of trials
simulationDuration = 20;

% generate dataset with three latents
[Y,prs,rates,fs,dx,dy,ntr,trLen,tt] = generate_toy_data(dy,ntr,simulationDuration);

ngtest = 2000;
testTimes = linspace(0,max(trLen),ngtest)';

trueLatents = {};
for nn = 1:ntr
    for ii = 1:dx
        trueLatents{nn, ii} = fs{ii,nn}(testTimes);
    end
end

exit = false;
while ~exit
    rNum = randi([0, 10^8-1], 1);
    simFilename = sprintf('results/%08d-pointProcessSimulation.mat', rNum);
    if ~isfile(simFilename)
        exit = true;
        save(simFilename, 'Y', 'prs', 'rates', 'fs', 'dx', 'dy', 'ntr', 'trLen', 'tt', 'testTimes', 'trueLatents')
    end
end
