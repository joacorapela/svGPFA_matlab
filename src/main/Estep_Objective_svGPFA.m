function [obj,grad] = Estep_Objective_svGPFA(m,prs,Kmats,ntr);

if nargin < 4
    trEval = 1:m.ntr;
elseif length(ntr) == 1
    trEval = ntr;
else
    trEval = ntr;
end

[q_mu, q_sqrt, q_diag, q_sigma, idx, idx_sig,idx_sigdiag] = extract_variationalPrs_svGPFA(m,prs,trEval);

[mu_h,var_h] = m.EMfunctions.predict_MultiOutputGP(Kmats,q_mu,q_sigma,m.prs.C,m.prs.b);

% get expected log-likelihood and gradient
Elik = m.EMfunctions.likelihood(m,mu_h,var_h,trEval);
gradElik = m.EMfunctions.gradLik_variationalPrs(m,Kmats,q_sqrt, q_diag,idx,idx_sig,idx_sigdiag,trEval,mu_h,var_h);

% get KL divergence and gradient
KLd = build_KL_divergence(m,Kmats,q_mu,q_sigma);
gradKLd = grad_variationalPrs_KL_divergence(m,Kmats,q_mu,q_sqrt,q_diag,idx,idx_sig,idx_sigdiag,trEval);

obj = -Elik + KLd; % negative free energy
grad = -gradElik + gradKLd; % gradients

% begin debug

% tt= m.tt;
% 
% C = m.prs.C;
% b = m.prs.b;
% 
% hprs = cellfun(@(struct)struct.hprs, m.kerns,'uni',0)';
% 
% kernelNames = {};
% for k=1:length(m.kerns)
%     kernelNames{k} = func2str(m.kerns{k}.K);
% end
% 
% xxHerm = m.xxHerm;
% wwHerm = m.wwHerm;
% Z = m.Z;
% Y = m.Y;
% BinWidth = m.BinWidth;
% 
% filename = '~/dev/research/gatsby/svGPFA/code/test/data/Estep_Objective_svGPFA.mat';
% save(filename, 'q_mu', 'q_sqrt', 'q_diag', 'tt', 'C', 'b', 'kernelNames', 'hprs', 'xxHerm', 'wwHerm', 'Z', 'Y', 'BinWidth', 'Elik', 'KLd', 'obj', 'grad');
% 
% keyboard

% end debug

