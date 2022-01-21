function [liksum,lik] = evaluatePredictiveLogLik_PointProcess(dat,T,pred,dt);
% function to evaluate the log likelihood for continuous time (dt = 0) or
% discretised time (dt > 0) poisson process observation.
% dat are the observations used for computing pred

ntr = max(size(dat));
for nn = 1:ntr
    [cnts{nn},dt] = discretiseSpikeTrain(dat{nn},T(nn),dt);
    cntmat(:,:,nn) = cell2mat(permute(cellfun(@(x) x',cnts{nn},'uni',0),[1 3 2]));
end
lambda = pred*dt;
lik = log(lambda).*cntmat - lambda - log(factorial(cntmat));
% sum over neurons and trials
liksum = sum(lik(:))./sum(cntmat(:));
