function [obj,grad] = Estep_Objective_PointProcess_svGPFA(m,prs,Kmats_Quad,Kmats_Spikes,ntr);

if nargin < 5
    trEval = 1:m.ntr;
elseif length(ntr) == 1
    trEval = ntr;
else
    trEval = ntr;
end

[q_mu, q_sqrt, q_diag, q_sigma, idx, idx_sig,idx_sigdiag] = extract_variationalPrs_svGPFA(m,prs,trEval);

% get multi output prediction for quad and from spikes
[mu_h_Quad,var_h_Quad, mu_k_Quad, var_k_Quad] = m.EMfunctions.predict_MultiOutputGP(Kmats_Quad,q_mu,q_sigma,m.prs.C,m.prs.b);
[mu_h_Spikes,var_h_Spikes, mu_k_Spikes, var_k_Spikes] = m.EMfunctions.predict_MultiOutputGP_fromSpikes(Kmats_Spikes,Kmats_Quad.Kzzi,Kmats_Quad.Kzz,q_mu,q_sigma,m.prs.C,m.prs.b,trEval,m.index);

% get expected log-likelihood and gradient
Elik = m.EMfunctions.likelihood(m,mu_h_Quad,var_h_Quad,mu_h_Spikes,var_h_Spikes,trEval);
gradElik = m.EMfunctions.gradLik_variationalPrs(m,Kmats_Quad,Kmats_Spikes,q_sqrt, q_diag,idx,idx_sig,idx_sigdiag,trEval,...
    mu_h_Quad,var_h_Quad);

% get KL divergence and gradient
KLd = build_KL_divergence(m,Kmats_Quad,q_mu,q_sigma);
gradKLd = grad_variationalPrs_KL_divergence(m,Kmats_Quad,q_mu,q_sqrt,q_diag,idx,idx_sig,idx_sigdiag,trEval);

% assemble objective function and gradients 
obj = -Elik + KLd; % negative free energy
grad = -gradElik + gradKLd; % gradients

% debug section start

% warning('Debug code is running on Estep_Objective_PointProcess_svGPFA.m');

% q_mu = m.q_mu;
% q_sqrt = m.q_sqrt;
% q_diag = m.q_diag;

% ttQuad = m.ttQuad;
% wwQuad = m.wwQuad;
% xxHerm = m.xxHerm;
% wwHerm = m.wwHerm;

% Kzz = Kmats_Quad.Kzz;
% Kzzi = Kmats_Quad.Kzzi;
% quadKtz = Kmats_Quad.Ktz;
% quadKtt = Kmats_Quad.Ktt;
% spikeKtz = Kmats_Spikes.Ktz;
% spikeKtt = Kmats_Spikes.Ktt;

% C = m.prs.C;
% b = m.prs.b;

% varRnk = m.opts.varRnk;

% index = m.index;

% Z = m.Z;
% Y = m.Y;

% hprs = cellfun(@(struct)struct.hprs, m.kerns,'uni',0)';

% kernelNames = {};
% for k=1:length(m.kerns)
%     kernelNames{k} = func2str(m.kerns{k}.K);
% end

% filename = '~/dev/research/gatsby-swc/gatsby/svGPFA/code/ci/data/Estep_Objective_PointProcess_svGPFA.mat';

% save(filename, 'q_mu', 'q_sqrt', 'q_diag', 'ttQuad', 'wwQuad', 'xxHerm', 'wwHerm', 'Z', 'Y', 'kernelNames', 'hprs', 'index', 'C', 'b', 'varRnk', 'obj', 'grad', 'KLd', 'mu_h_Quad','var_h_Quad', 'mu_h_Spikes','var_h_Spikes', 'mu_k_Quad','var_k_Quad', 'mu_k_Spikes','var_k_Spikes', 'Elik');

% keyboard

% debug section end
