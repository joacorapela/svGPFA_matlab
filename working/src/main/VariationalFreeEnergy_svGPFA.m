function fe = VariationalFreeEnergy_svGPFA(m)

% compute kernel matrices
current_hprs = cellfun(@(struct)struct.hprs, m.kerns,'uni',0)'; % extract kernel hyperparams
Kmats = BuildKernelMatrices(m,m.tt,m.Z,current_hprs,0);
% compute the variational free energy
[mu_h,var_h] = m.EMfunctions.predict_MultiOutputGP(Kmats,m.q_mu,m.q_sigma,m.prs.C,m.prs.b);

Elik = m.EMfunctions.likelihood(m,mu_h,var_h);

% get KL divergence 
KLd = build_KL_divergence(m,Kmats,m.q_mu,m.q_sigma);

fe = Elik - KLd;