function m = hyperMstep_Update(m)
% function to update hyperparameters of svGPFA model

hprs0 = cellfun(@(struct)struct.hprs, m.kerns,'uni',0)';

% extract hyperparameters for each GP kernel
prs0 = m.EMfunctions.extract_hyperParams(m);

% make objective function
fun = @(prs,nIter) m.EMfunctions.HyperMstep_Objective(m,prs,nIter);

% check gradients numerically
if m.opts.verbose == 2 % extra level of verbosity
    fprintf('hyper M step grad check:\n')
    DerivCheck(@(x)fun(x,1),prs0);
end

optimopts = optimset('Gradobj','on','display', 'none');

optimopts.MaxIter = m.opts.maxiter.hyperMstep;

% ADAM option values
DEF_stepSize = 0.001;
DEF_beta1 = 0.9;
DEF_beta2 = 0.999;
DEF_epsilon = sqrt(eps);

[prs, nLowerBound, exitfag, output] = fmin_adam(fun,prs0,DEF_stepSize, DEF_beta1, DEF_beta2, DEF_epsilon, 1, optimopts);

% update kernel hyperparameters in model structure
m = m.EMfunctions.updateHyperParams(m,prs);

% begin debug

% warning('Debug code is running on hyperMstep_Update.m');
% 
% q_mu = m.q_mu;
% q_sqrt = m.q_sqrt;
% q_diag = m.q_diag;
% C = m.prs.C;
% b = m.prs.b;
% index = m.index;
% ttQuad = m.ttQuad;
% wwQuad = m.wwQuad;
% xxHerm = m.xxHerm;
% wwHerm = m.wwHerm;
% Y = m.Y;
% Z = m.Z;
% epsilon = n.epsilon;
% 
% kernelNames = {};
% for k=1:length(m.kerns)
%     kernelNames{k} = func2str(m.kerns{k}.K);
% end
% 
% maxIter = optimopts.MaxIter;
% 
% filename = '~/dev/research/gatsby-swc/gatsby/svGPFA/pythonCode/ci/data/hyperMstep_Update.mat';
% save(filename, 'epsilon', 'q_mu', 'q_sqrt', 'q_diag', 'C', 'b', 'index', 'ttQuad', 'wwQuad', 'xxHerm', 'wwHerm', 'Z', 'Y', 'hprs0', 'kernelNames', 'maxIter', 'prs', 'nLowerBound', 'exitfag', 'output');
% 
% keyboard

% end debug
