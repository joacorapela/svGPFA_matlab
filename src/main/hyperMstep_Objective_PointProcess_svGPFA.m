function [obj,grad] = hyperMstep_Objective_PointProcess_svGPFA(m,prs,nIter)

% extract hprs from parameter vector
hprs = cell(m.dx,1);
idxhprs = cell(m.dx,1);

hprsidx = cumsum(cell2mat(cellfun(@(c)c.numhprs, m.kerns,'uni',0)'));
istrthprs = [1; hprsidx(1:end-1)+1];
iendhprs = hprsidx;

for kk = 1:m.dx
    hprs{kk} = prs(istrthprs(kk):iendhprs(kk));
    idxhprs{kk} = (istrthprs(kk):iendhprs(kk));
end

% get minibatch
batchidx = m.minibatches(nIter,:);

% extract Z and q's and for minibatch
Zs = cellfun(@(x) x(:,:,batchidx),m.Z,'uni',0);
q_mu = cellfun(@(x) x(:,:,batchidx),m.q_mu,'uni',0);
q_sigma = cellfun(@(x) x(:,:,batchidx),m.q_sigma,'uni',0);

% buil kernel matrices
KmatsQuad = BuildKernelMatrices(m,m.ttQuad(:,:,batchidx),Zs,hprs,1);
KmatsObs = BuildKernelMatrices_fromSpikes(m,Zs,hprs,1,batchidx);

% get multioutput GP prediction
[mu_hQuad,var_hQuad] = m.EMfunctions.predict_MultiOutputGP(KmatsQuad,q_mu,q_sigma,m.prs.C,m.prs.b);
[mu_hObs,var_hObs] = m.EMfunctions.predict_MultiOutputGP_fromSpikes(KmatsObs,KmatsQuad.Kzzi,KmatsQuad.Kzz,q_mu,q_sigma,m.prs.C,m.prs.b,batchidx,m.index);

% get likelihood and gradient
Elik = m.EMfunctions.likelihood(m,mu_hQuad,var_hQuad,mu_hObs,var_hObs,batchidx);
gradElik = m.EMfunctions.gradLik_hprs(m,KmatsQuad,KmatsObs,idxhprs,mu_hQuad,var_hQuad,q_mu,q_sigma,batchidx);

% get KL divergence and gradient
KLd = build_KL_divergence(m,KmatsQuad,q_mu,q_sigma);
gradKLd = grad_hprs_KL_divergence(m,KmatsQuad,idxhprs,q_mu,q_sigma);

obj = -Elik + KLd; % negative free energy
grad = -gradElik + gradKLd; % gradients

% disp(sprintf('*** obj = %f ***', obj));
