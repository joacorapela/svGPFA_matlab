function [obj,grad] = inducingPointMstep_Objective_PointProcess_svGPFA(m,prs,ntr);

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

Z0 = m.Z;
prs = reshape(prs,[],1, length(trEval));
for kk = 1:m.dx 
    Z{kk} = prs(istrt(kk):iend(kk),:,:);
    idxZ{kk} = (istrt(kk):iend(kk));
end

% build kernel matrices
current_hprs = cellfun(@(struct)struct.hprs, m.kerns,'uni',0)'; % extract kernel hyperparams

% buil kernel matrices
KmatsQuad = BuildKernelMatrices(m,m.ttQuad(:,:,trEval),Z,current_hprs,2);
KmatsObs = BuildKernelMatrices_fromSpikes(m,Z,current_hprs,2,trEval);

[q_mu,q_sigma] = extract_VariatinalPrs_single(m.q_mu,m.q_sigma,trEval);

% get multioutput GP prediction
[mu_hQuad,var_hQuad] = m.EMfunctions.predict_MultiOutputGP(KmatsQuad,q_mu,q_sigma,m.prs.C,m.prs.b);
[mu_hObs,var_hObs] = m.EMfunctions.predict_MultiOutputGP_fromSpikes(KmatsObs,KmatsQuad.Kzzi,KmatsQuad.Kzz,q_mu,q_sigma,m.prs.C,m.prs.b,trEval,m.index);

% get likelihood and gradient
Elik = m.EMfunctions.likelihood(m,mu_hQuad,var_hQuad,mu_hObs,var_hObs,trEval);
gradElik = m.EMfunctions.gradLik_inducingPoints(m,KmatsQuad,KmatsObs,idxZ,trEval,mu_hQuad,var_hQuad,q_mu,q_sigma);

% get KL divergence and gradient
KLd = build_KL_divergence(m,KmatsQuad,q_mu,q_sigma);
gradKLd = grad_inducingPoints_KL_divergence(m,KmatsQuad,idxZ,q_mu,q_sigma,trEval);

% assemble objective and gradients 
obj = - Elik + KLd; % negative free energy
grad = - gradElik + gradKLd; % gradients

% disp(sprintf('*** obj = %f ***', obj));
