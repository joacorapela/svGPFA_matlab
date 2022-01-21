
%% script to simulate Poisson Process data and run sv-ppGPFA 
% clear all; close all;
% addpath(genpath('../src'))
addpath(genpath('~/dev/research/programs/src/matlab/iniconfig'))

mEstNumber=70109875;
% mEstNumber=30145494;

estimationParamsFilename = sprintf('results/%08d-pointProcessEstimationParams.ini', mEstNumber);
ini = IniConfig();
ini.ReadFile(estimationParamsFilename);
pEstNumber = ini.GetValues('data', 'pEstNumber');


pythonDataFilenamePattern = '../../pythonCode/scripts/results/%08d_estimationDataForMatlab.mat';
pythonDataFilename = sprintf(pythonDataFilenamePattern, pEstNumber);
pythonData = load(pythonDataFilename);

mEstResFilename = sprintf('results/%08d-pointProcessEstimationRes.mat', mEstNumber);
mEstRes = load(mEstResFilename);
m = mEstRes.m;

latentsTimes = pythonData.(sprintf("latentsTrialsTimes_0"));
latentsTimes = reshape(latentsTimes, [length(latentsTimes), 1]);
pred = predictNew_svGPFA(m, latentsTimes);

lowerBound = m.FreeEnergy;
elapsedTime = m.elapsedTime;
meanEstimatedLatents = pred.latents.mean;
varEstimatedLatents = pred.latents.variance;

estimationResFilename = sprintf('results/%08d-pointProcessEstimationRes.mat', mEstNumber);
save(estimationResFilename, 'lowerBound', 'elapsedTime', 'latentsTimes', 'meanEstimatedLatents', 'varEstimatedLatents', 'm');
