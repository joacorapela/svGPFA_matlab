function [idx,idxMat] = time2ind(tt,timesMat);
% compute indices in tt vector that correspond closest in time to particular event times
% stored in Numtrials x numEvents matrix timesMat

numEval = length(tt);
idx = 1:numEval;

[nTrials,nEvents] = size(timesMat);

FirstAll = bsxfun(@(x,y) x >= y,tt,permute(timesMat,[1 3 2]));

idxMat = zeros(nTrials,nEvents);
for ii = 1:nTrials
    for jj = 1:nEvents
        idxMat(ii,jj) = find(FirstAll(ii,:,jj),1,'first');
    end
end