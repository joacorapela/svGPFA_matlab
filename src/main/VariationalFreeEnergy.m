function fe = VariationalFreeEnergy(m,Kmats,varargin);
% compute the variational free energy
[mu_h,var_h] = predict_MultiOutputGP(Kmats,m.q_mu,m.q_sigma,m.prs.C,m.prs.b);

Elik = m.EMfunctions.likelihood(m,mu_h,var_h);

% get KL divergence 
KLd = build_KL_divergence(m,Kmats,m.q_mu,m.q_sigma);

fe = Elik - KLd;