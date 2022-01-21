function [obj,grad] = inducingPointMstep_Objective_svGPFA(m,prs,ntr);

if nargin < 3
    trEval = 1:m.ntr;
else
    trEval = ntr;
end

% extract induincing points from parameter vector
Z = cell(m.dx,1);
idxZ = cell(m.dx,1);

istrt = [1 cumsum(m.numZ(1:end-1))+1];
iend  = cumsum(m.numZ);

prs = reshape(prs,[],1, length(trEval));
for kk = 1:m.dx 
    Z{kk} = prs(istrt(kk):iend(kk),:,:);
    idxZ{kk} = (istrt(kk):iend(kk));
end

% build kernel matrices
current_hprs = cellfun(@(struct)struct.hprs, m.kerns,'uni',0)'; % extract kernel hyperparams
Kmats = BuildKernelMatrices(m,m.tt,Z,current_hprs,2);

[q_mu,q_sigma] = extract_VariatinalPrs_single(m.q_mu,m.q_sigma,trEval);

[mu_h,var_h] = m.EMfunctions.predict_MultiOutputGP(Kmats,q_mu,q_sigma,m.prs.C,m.prs.b);


% get expected log-likelihood and gradient
Elik = m.EMfunctions.likelihood(m,mu_h,var_h,trEval);
gradElik = m.EMfunctions.gradLik_inducingPoints(m,Kmats,idxZ,trEval,mu_h,var_h,q_mu,q_sigma);

% get KL divergence and gradient
KLd = build_KL_divergence(m,Kmats,q_mu,q_sigma);
gradKLd = grad_inducingPoints_KL_divergence(m,Kmats,idxZ,q_mu,q_sigma,trEval);

obj = - Elik + KLd; % negative free energy
grad = - gradElik + gradKLd; % gradients
