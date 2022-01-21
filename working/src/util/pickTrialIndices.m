function idx = pickTrialIndices(numtrials,moveLens,RTs);
% function to select trials with variable movement durations
% this is done to make aligning to MO onset worse

% sort times
ntr = length(moveLens);
assert(ntr == length(RTs));

[~,iiML] = sort(moveLens);
% [~,iiRT] = sort(RTs);

% select trials from tails and center of distribution
numMax = floor(numtrials/3);
numMin = floor(numtrials/3);
numMean = numtrials - numMax - numMin;
idx = [iiML(1:numMax); iiML(end-numMin+1:end); iiML(floor(ntr/2) -floor(numMean/2) + 1:floor(ntr/2)-floor(numMean/2)+numMean)];
% assert(length(idxML) == numtrials);
% idxRT = [iiRT(1:numMax); iiRT(end-numMin+1:end); iiRT(floor(ntr/2) -floor(numMean/2) + 1:floor(ntr/2)-floor(numMean/2)+numMean)];
% assert(length(idxRT) == numtrials);
% 
% idx = unique([idxRT; idxML]);
% idx = idx(randperm(length(idx)));
% idx = idx(1:numtrials);
