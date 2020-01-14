function loglik = evaluatePredictiveLogLik_PointProcess(mu_h_Quad,mu_h_Spikes,link,wwQuad);
% function to evaluate the log likelihood for continuous time (dt = 0) or
% discretised time (dt > 0) poisson process observation.
% dat are the observations used for computing pred

intval = link(mu_h_Quad); % numQuad x N x length(trEval)

% lik_pp1 = sum(vec(mtimesx(wwQuad,'T',intval)));
% lik_pp2 = sum(cellvec(mu_h_Spikes));

lik_pp1 = vec(mtimesx(wwQuad,'T',intval)); % trEval x 1 vector of normaliser term
lik_pp2 = cell2mat(cellfun(@(x)sum(log(link(x))),mu_h_Spikes,'uni',0))';

loglik = -lik_pp1 + lik_pp2;