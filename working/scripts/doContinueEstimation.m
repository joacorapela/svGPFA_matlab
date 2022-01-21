close all; clear all;
addpath(genpath('../src'))
addpath(genpath('~/dev/research/programs/src/matlab/iniconfig'))

emMaxIter=200; prevEstResNumber=49242119;
prevEstResFilenamePattern = 'results/%8d-pointProcessEstimationRes.mat';

prevEstResFilename = sprintf(prevEstResFilenamePattern, prevEstResNumber);
prevEstRes = load(prevEstResFilename);

m = prevEstRes.m;
latentsTimes = prevEstRes.latentsTimes;
m.opt.maxiter.EM = emMaxIter;

estimationParamsINI = IniConfig();
estimationParamsINI.AddSections({'data'});
estimationParamsINI.AddKeys('data', 'prevEstResNumber', prevEstResNumber);

exit = false;
while ~exit
    estResNumber = randi([0, 10^8-1], 1);
    estimationParamsFilename = sprintf('results/%08d-pointProcessEstimationParams.ini', estResNumber);
    if exist(estimationParamsFilename, "file")~=2
        estimationParamsINI.WriteFile(estimationParamsFilename);
        exit = true;
    end
end

m.savePartial = false;

m = variationalEM(m, @getKernelsParams);

pred = predictNew_svGPFA(m, latentsTimes);

lowerBound = m.FreeEnergy;
meanEstimatedLatents = pred.latents.mean;
varEstimatedLatents = pred.latents.variance;

estimationResFilename = sprintf('results/%08d-pointProcessEstimationRes.mat', estResNumber);
save(estimationResFilename, 'lowerBound', 'latentsTimes', 'meanEstimatedLatents', 'varEstimatedLatents', 'm');

function params = getKernelsParams(m)
    params = [m.kerns{1}.hprs, m.kerns{2}.hprs]
end

