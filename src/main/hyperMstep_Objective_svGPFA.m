function [obj,grad] = hyperMstep_Objective_svGPFA(m,prs,nIter)

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

% build kernel matrices for current minibatch
Kmats = BuildKernelMatrices(m,m.tt,Zs,hprs,1);

% get multioutput GP prediction
[mu_h,var_h] = m.EMfunctions.predict_MultiOutputGP(Kmats,q_mu,q_sigma,m.prs.C,m.prs.b);

% get likelihood and gradient
Elik = m.EMfunctions.likelihood(m,mu_h,var_h,batchidx);
gradElik = m.EMfunctions.gradLik_hprs(m,Kmats,idxhprs,batchidx,mu_h,var_h,q_mu,q_sigma);

% get KL divergence and gradient
KLd = build_KL_divergence(m,Kmats,q_mu,q_sigma);
gradKLd = grad_hprs_KL_divergence(m,Kmats,idxhprs,q_mu,q_sigma);

obj = -Elik + KLd; % negative free energy
grad = -gradElik + gradKLd; % gradients
