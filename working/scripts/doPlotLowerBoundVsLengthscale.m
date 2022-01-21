
clear all; close all;
addpath(genpath('../src'))

pEstNumber = 19940935;
partialDesc = 'mstep_embedding008';

nIter = 1;
lengthscaleValueStart = 0.01;
lengthscaleValueEnd = 2.00;
lengthscaleValueStep = 0.01;

finalPythonDataFilenamePattern = '../../pythonCode/scripts/shenoy/results/%08d_estimationDataForMatlab.mat';
intermediatePythonDataFilenamePattern = '../../pythonCode/scripts/shenoy/results/%08d_%s_estimationDataForMatlab.mat';
finalFigFilenamePattern = 'figures/%08d-lowerBoundVsLengthscale.png';
intermediateFigFilenamePattern = 'figures/%08d-%s-lowerBoundVsLengthscale.png';

if length(partialDesc)>0
    pythonDataFilename = sprintf(intermediatePythonDataFilenamePattern, pEstNumber, partialDesc);
    figFilename = sprintf(intermediateFigFilenamePattern, pEstNumber, partialDesc);
else 
    pythonDataFilename = sprintf(finalPythonDataFilenamePattern, pEstNumber, partialDesc);
    figFilename = sprintf(finalFigFilenamePattern, pEstNumber, partialDesc);
end

m = buildMatlabFromPythonModel(pythonDataFilename);
lengthscaleValues = lengthscaleValueStart:lengthscaleValueStep:lengthscaleValueEnd;
plotLowerBoundVsLengthscale(m, nIter, lengthscaleValues, figFilename)
