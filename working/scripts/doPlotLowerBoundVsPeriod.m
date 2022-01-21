
clear all; close all;
addpath(genpath('../src'))

% mEstNumber=30428568; % epsilon 1e-3, period0=5.0
% mEstNumber=48660016; % epsilon 1e-3, period0=7.0
% mEstNumber=58183141; % epsilon 1e-3, period0=3.0

% mEstNumber=06156497; % epsilon 1e-2, period0=5.0
% mEstNumber=70992717; % epsilon 1e-2, period0=7.0
% mEstNumber=23891803; % epsilon 1e-2, period0=3.0

% mEstNumber=17913026; % epsilon 1e-5, period0=5.0
% mEstNumber=70338114; % epsilon 1e-5, period0=7.0
% mEstNumber=37865224; % epsilon 1e-5, period0=3.0

mEstNumber=79220732; % epsilon 1e-5, period0=3.0
partialDesc = 'initial';

% periodValueStart = 0.01;
periodValueStart = 0.84;
periodValueEnd = 2.0;
periodValueStep = 0.01;
nIter = 1;

periodValues = periodValueStart:periodValueStep:periodValueEnd;
lowerBoundValues = zeros(length(periodValues),1);

if length(partialDesc)>0
    estimationResFilename = sprintf('results/%08d-%s-pointProcessEstimationRes.mat', mEstNumber, partialDesc);
    figFilename = sprintf('figures/%08d-%s-lowerBoundVsPeriod.png', mEstNumber, partialDesc);
else 
    estimationResFilename = sprintf('results/%08d-pointProcessEstimationRes.mat', mEstNumber);
    figFilename = sprintf('figures/%08d-lowerBoundVsPeriod.png', mEstNumber);
end

loadRes = load(estimationResFilename, 'm');
m = loadRes.m;
plotLowerBoundVsPeriod(m, nIter, periodValues, figFilename)
